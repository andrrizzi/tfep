#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Modules and functions to compute energies and gradients with OpenMM.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

# DO NOT IMPORT OpenMM HERE! OpenMM is an optional dependency of tfep.
from typing import Optional

from MDAnalysis.lib.mdamath import triclinic_vectors
import numpy as np
import pint
import torch

from tfep.utils.misc import (
    atom_to_flattened, flattened_to_atom,
    energies_array_to_tensor, forces_array_to_tensor
)
from tfep.potentials.base import PotentialBase


# =============================================================================
# TORCH MODULE API
# =============================================================================

class OpenMMPotential(PotentialBase):
    """Potential energy and forces with OpenMM.

    Wraps OpenMM to provides a differentiable potential energy function for
    training.

    .. warning::
        Currently double-backpropagation is not supported, which means force
        matching cannot be performed during training.

    """

    #: The default energy unit.
    DEFAULT_ENERGY_UNIT : str = 'kJ/mol'

    #: The default positions unit.
    DEFAULT_POSITIONS_UNIT : str = 'nanometer'

    def __init__(
            self,
            openmm_context,
            positions_unit: Optional[pint.Unit] = None,
            energy_unit: Optional[pint.Unit] = None,
            precompute_gradient: bool = False,
    ):
        """Constructor.

        Parameters
        ----------
        openmm_context : openmm.Context or openmm.System
            The OpenMM ``Context`` used to compute energies and forces. If a
            ``System`` object is passed instead, a ``Context`` is created with
            the OpenMM default ``Platform``.
        positions_unit : pint.Unit, optional
            The unit of the positions passed to the class methods. Since input
            ``Tensor``s do not have units attached, this is used to appropriately
            convert ``batch_positions`` to OpenMM units. If ``None``, no conversion
            is performed, which assumes that the input positions are in the same
            units used internally by OpenMM (nanometer).
        energy_unit : pint.Unit, optional
            The unit used for the returned energies (and as a consequence forces).
            Since ``Tensor``s do not have units attached, this is used to
            appropriately convert OpenMM energies into the desired units. If ``None``
            no conversion is performed, which means that energies and forces will be
            returned in OpenMM units (kJ/mol).
        precompute_gradient : bool, optional
            If ``True``, the gradient is computed in the forward pass and saved
            to be consumed during the backward pass. This speeds up the training,
            but should be deactivated if gradients are not needed. Setting this
            to ``False`` (default) will cause an exception if a backward pass is
            attempted.

        See Also
        --------
        :class:`.OpenMMPotentialEnergyFunc`
            More details on input parameters and implementation details.

        """
        from openmm import Context, System, VerletIntegrator

        super().__init__(positions_unit=positions_unit, energy_unit=energy_unit)

        if isinstance(openmm_context, System):
            # Cache a Context with the default platform. We don't care about the
            # integrator since we perform only single-point calculations.
            openmm_context = Context(openmm_context, VerletIntegrator(0.001))

        #: The OpenMM Context used to compute energies and forces.
        self.openmm_context = openmm_context

        #: Whether to compute the gradients in the forward pass to speed up the backward pass.
        self.precompute_gradient = precompute_gradient

    def forward(
            self,
            batch_positions: torch.Tensor,
            batch_cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute a differential potential energy for a batch of configurations.

        Parameters
        ----------
        batch_positions : torch.Tensor
            Shape ``(batch_size, 3*n_atoms)``. The atoms positions in units of
            ``self.positions_unit``.
        batch_cell : torch.Tensor, optional
            Shape ``(batch_size, 6)``. Unitcell dimensions. For each data point,
            the first 3 elements represent the vector lengths in units of
            ``self.positions_unit`` and the last 3 their respective angles (in
            degrees) in the following order:
            ``[len(a), len(b), len(c), angle(b,c), angle(a,c), angle(a,b)]``.
            The first vector will lie in x-direction, second in xy-plane, and the
            third one in the z-positive subspace.

        Returns
        -------
        potential_energy : torch.Tensor
            ``potential_energy[i]`` is the potential energy of configuration
            ``batch_positions[i]`` in units of ``self.energy_unit``.

        """
        return openmm_potential_energy(
            batch_positions,
            openmm_context=self.openmm_context,
            batch_cell=batch_cell,
            positions_unit=self._positions_unit,
            energy_unit=self._energy_unit,
            precompute_gradient=self.precompute_gradient,
        )


# =============================================================================
# TORCH FUNCTIONAL API
# =============================================================================

class OpenMMPotentialEnergyFunc(torch.autograd.Function):
    """PyTorch-differentiable potential energy using ASE.

    Wraps OpenMM to provides a differentiable potential energy function for
    training.

    .. warning::
        Currently double-backpropagation is not supported, which means force
        matching cannot be performed during training.

    By default, the perform the batch of energy/gradient calculations serially.
    This scheme is, however, not embarassingly parallel. Thus, the module supports
    batch parallelization schemes through :class:``tfep.utils.parallel.ParallelizationStrategy``s.

    Parameters
    ----------
    ctx : torch.autograd.function._ContextMethodMixin
        A PyTorch context to save information for the gradient.
    batch_positions : torch.Tensor
        Shape ``(batch_size, 3*n_atoms)``. The atoms positions in units of
        ``positions_unit`` (or OpenMM units if ``positions_unit`` is not provided).
    openmm_context : openmm.Context
        The OpenMM ``Context`` used to compute energies and forces.
    batch_cell : torch.Tensor or None, optional
        Shape ``(batch_size, 6)``. Unitcell dimensions. For each data point,
        the first 3 elements represent the vector lengths in units of
        ``positions_unit`` (or OpenMM units if ``positions_unit`` is not provided)
        and the last 3 their respective angles (in degrees) in the following
        order: ``[len(a), len(b), len(c), angle(b,c), angle(a,c), angle(a,b)]``.
        The first vector will lie in x-direction, second in xy-plane, and the
        third one in the z-positive subspace.
    positions_unit : pint.Unit, optional
        The unit of the positions passed. This is used to appropriately convert
        ``batch_positions`` to OpenMM units. If ``None``, no conversion is performed,
        which assumes that the input positions are in the same units used by
        OpenMM (nanometer).
    energy_unit : pint.Unit, optional
        The unit used for the returned energies (and as a consequence forces).
        This is used to appropriately convert OpenMM energies into the desired
        units. If ``None``, no conversion is performed, which means that energies
        and forces will use OpenMM units (kJ/mol).
    precompute_gradient : bool, optional
        If ``True``, the gradient is computed in the forward pass and saved to
        be consumed during the backward pass. This speeds up the training, but
        should be deactivated if gradients are not needed. Setting this to
        ``False`` (default) will cause an exception if a backward pass is
        attempted.

    Returns
    -------
    potentials : torch.Tensor
        ``potentials[i]`` is the potential energy of configuration
        ``batch_positions[i]``.

    See Also
    --------
    :class:`.OpenMMPotential`
        ``Module`` API for computing potential energies with OpenMM.

    """

    @staticmethod
    def forward(
            ctx,
            batch_positions: torch.Tensor,
            openmm_context,
            batch_cell: Optional[torch.Tensor] = None,
            positions_unit: Optional[pint.Unit] = None,
            energy_unit: Optional[pint.Unit] = None,
            precompute_gradient: bool = False,
    ):
        """Compute the potential energy of the molecule with ASE."""
        batch_size = batch_positions.shape[0]

        # Convert tensor with shape (batch_size, n_atoms*3) to numpy array with
        # shape (batch_size, n_atoms, 3) in OpenMM units (nanometer).
        batch_positions_arr_nm = flattened_to_atom(batch_positions.detach().cpu().numpy())
        if positions_unit is not None:
            batch_positions_arr_nm = _to_openmm_units(batch_positions_arr_nm, positions_unit)

        # Make sure there's a list of box vectors.
        if batch_cell is None:
            box_vectors_nm = [None for _ in range(batch_size)]
        else:
            # To nm. The last 3 entires of each batch cell are angles, not lengths.
            batch_cell_arr_nm = batch_cell.detach().cpu().numpy()
            if positions_unit is not None:
                batch_cell_arr_nm[:, :3] = _to_openmm_units(batch_cell_arr_nm[:, :3], positions_unit)

            # From lengths+angles to box vectors.
            box_vectors_nm = [triclinic_vectors(x) for x in batch_cell_arr_nm]

        # Compute energies (and optionally forces).
        energies = [_run_single_point_calculation(
            openmm_context,
            batch_positions_arr_nm[i],
            box_vectors_nm[i],
            return_forces=precompute_gradient,
        ) for i in range(batch_size)]

        # Unpack the results. From [(energy, forces), ...] to ([energy, ...], [forces, ...]).
        if precompute_gradient:
            energies, forces = list(zip(*energies))
            # From Tuple[ndarray] to ndarray.
            forces = np.array(forces)

        # Convert energies to unitless tensors.
        if energy_unit is None:
            energies = torch.tensor(energies)
        else:
            energies *= OpenMMPotential.default_energy_unit(energy_unit._REGISTRY)
            energies = energies_array_to_tensor(energies, energy_unit)
        energies = energies.to(batch_positions)

        # Save the forces for backward propagation. We do not support backward
        # passes with precompute_gradient=False.
        if precompute_gradient:
            ctx.forces = forces
            ctx.energy_unit = energy_unit
            ctx.positions_unit = positions_unit
        else:
            ctx.forces = None

        return energies

    @staticmethod
    def backward(ctx, grad_output):
        """Compute the gradient of the potential energy."""
        # We still need to return a None gradient for each
        # input of forward() beside batch_positions.
        n_input_args = 6
        grad_input = [None for _ in range(n_input_args)]

        # Compute gradient w.r.t. batch_positions.
        if ctx.needs_input_grad[0]:
            # Check that forces are available.
            if ctx.forces is None:
                raise ValueError('precompute_gradient must be set to True for backward pass.')

            # Convert to unitless tensors.
            if (ctx.energy_unit is None) and (ctx.positions_unit is None):
                forces = torch.from_numpy(atom_to_flattened(ctx.forces))
            else:
                ureg = ctx.energy_unit._REGISTRY
                default_positions_unit = OpenMMPotential.default_positions_unit(ureg)
                default_energy_unit = OpenMMPotential.default_energy_unit(ureg)
                forces = ctx.forces * default_energy_unit / default_positions_unit
                forces = forces_array_to_tensor(forces, ctx.positions_unit, ctx.energy_unit)
            forces = forces.to(grad_output)

            # Accumulate gradient
            grad_input[0] = -forces * grad_output[:, None]

        return tuple(grad_input)


def openmm_potential_energy(
        batch_positions: torch.Tensor,
        openmm_context,
        batch_cell: Optional[torch.Tensor] = None,
        positions_unit: Optional[pint.Unit] = None,
        energy_unit: Optional[pint.Unit] = None,
        precompute_gradient: bool = False,
):
    """PyTorch-differentiable potential energy using OpenMM.

    PyTorch ``Function``s do not accept keyword arguments. This function wraps
    :func:`.OpenMMPotentialEnergyFunc.apply` to enable standard functional notation.
    See the documentation on the original function for the input parameters.

    See Also
    --------
    :class:`.OpenMMPotentialEnergyFunc`
        More details on input parameters and implementation details.

    """
    # apply() does not accept keyword arguments.
    return OpenMMPotentialEnergyFunc.apply(
        batch_positions,
        openmm_context,
        batch_cell,
        positions_unit,
        energy_unit,
        precompute_gradient,
    )


# =============================================================================
# RUNNING UTILITY FUNCTIONS
# =============================================================================

def _to_openmm_units(x, positions_unit):
    """Convert x from positions_unit to nanometers."""
    default_positions_unit = OpenMMPotential.default_positions_unit(positions_unit._REGISTRY)
    return (x * positions_unit).to(default_positions_unit).magnitude


def _run_single_point_calculation(openmm_context, positions, box_vectors, return_forces):
    """Run a single-point calculation.

    This does not support batch dimensions.

    positions and box_vectors (if not None) must be numpy arrays in units of
    nanometers with shapes (n_atoms, 3) and (3, 3) respectively.

    Returns energy and forces in OpenMM units (stripped of OpenMM units).

    """
    # Temporarily enable grad to support energy/force calculations with OpenMM-ML.
    with torch.enable_grad():
        if box_vectors is not None:
            openmm_context.setPeriodicBoxVectors(*box_vectors)
        openmm_context.setPositions(positions)
        state = openmm_context.getState(getEnergy=True, getForces=return_forces)

    energy = state.getPotentialEnergy()._value
    if return_forces:
        return [energy, state.getForces(asNumpy=True)._value]
    return energy
