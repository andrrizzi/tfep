#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Modules and functions to compute QM energies and gradients with ASE.

The function/classes in this module wrap an Atomistic Simulation Environment
(ASE) ``Calculator``s and make them compatible with PyTorch.

See Also
--------
ASE: https://wiki.fysik.dtu.dk/ase/index.html

"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

# DO NOT IMPORT ASE HERE! ASE is an optional dependency of tfep.
import copy
import functools

import numpy as np
import torch

from tfep.utils.misc import (
    atom_to_flattened, flattened_to_atom,
    energies_array_to_tensor, forces_array_to_tensor
)
from tfep.utils.parallel import SerialStrategy
from tfep.potentials.base import PotentialBase


# =============================================================================
# TORCH MODULE API
# =============================================================================

class ASEPotential(PotentialBase):
    """Potential energy and forces with ASE.

    This ``Module`` wraps :class:``.ASEPotentialEnergyFunc`` to provide a
    differentiable potential energy function for training.

    .. warning::
        Currently double-backpropagation is not supported, which means force
        matching cannot be performed during training.

    """

    #: The default energy unit.
    DEFAULT_ENERGY_UNIT : str = 'eV'

    #: The default positions unit.
    DEFAULT_POSITIONS_UNIT : str = 'angstrom'

    def __init__(
            self,
            calculator,
            symbols=None,
            numbers=None,
            pbc=None,
            positions_unit=None,
            energy_unit=None,
            parallelization_strategy=None,
            **atoms_kwargs,
    ):
        """Constructor.

        Parameters
        ----------
        calculator : ase.calculators.calculator.Calculator
            The ASE calculator used to compute energies and forces.
        symbols : str or List[str]
            The symbols of the atoms elements used to initialize the ``ase.Atoms``
            object. It can be a string formula, a list of symbols, or a list of
            ``ase.Atom`` objects.  Examples: ``'H2O'``, ``'COPt12'``, ``['H', 'H', 'O']``,
            ``[Atom('Ne', (x, y, z)), ...]``.
        numbers: List[int]
            Atomic numbers (use only one between symbols and numbers).
        pbc: bool or three bool
            Periodic boundary conditions flags.  Examples: ``True``, ``False``,
            ``0``, ``1``, ``(1, 1, 0)``, ``(True, False, False)``.
        positions_unit : pint.Unit, optional
            The unit of the positions passed to the class methods. Since input
            ``Tensor``s do not have units attached, this is used to appropriately
            convert ``batch_positions`` to ASE units. If ``None``, no conversion
            is performed, which assumes that the input positions are in the same
            units used by ASE.
        energy_unit : pint.Unit, optional
            The unit used for the returned energies (and as a consequence forces).
            Since ``Tensor``s do not have units attached, this is used to
            appropriately convert ASE energies into the desired units. If ``None``
            is performed, which means that energies and forces will be returned
            in ASE units.
        parallelization_strategy : tfep.utils.parallel.ParallelizationStrategy, optional
            The parallelization strategy used to distribute batches of energy and
            gradient calculations. By default, these are executed serially.
        **atoms_kwargs
            Other keyword arguments for ``ase.Atoms``.

        See Also
        --------
        :class:`.ASEPotentialEnergyFunc`
            More details on input parameters and implementation details.

        """
        from ase import Atoms

        super().__init__(positions_unit=positions_unit, energy_unit=energy_unit)

        # Handle mutable default arguments.
        if parallelization_strategy is None:
            parallelization_strategy = SerialStrategy()

        self.atoms = Atoms(
            symbols=symbols,
            numbers=numbers,
            pbc=pbc,
            calculator=calculator,
            **atoms_kwargs,
        )
        self.parallelization_strategy = parallelization_strategy

    def forward(self, batch_positions, batch_cell=None):
        """Compute a differential potential energy for a batch of configurations.

        Parameters
        ----------
        batch_positions : torch.Tensor
            Shape ``(batch_size, 3*n_atoms)``. The atoms positions in units of
            ``self.positions_unit``.
        batch_cell : torch.Tensor, optional
            Shape ``(batch_size, 3, 3)`` or ``(batch_size, 3)`` or ``(batch_size, 6)``.
            Unit cell vectors.  Can also be given as just three numbers for
            orthorhombic cells, or 6 numbers, where first three are lengths of
            unit cell vectors (in units of ``self.positions_unit``, and the other
            three are angles between them (in degrees), in following order:
            ``[len(a), len(b), len(c), angle(b,c), angle(a,c), angle(a,b)]``.
            First vector will lie in x-direction, second in xy-plane, and the
            third one in z-positive subspace.

        Returns
        -------
        potential_energy : torch.Tensor
            ``potential_energy[i]`` is the potential energy of configuration
            ``batch_positions[i]`` in units of ``self.energy_unit``.

        """
        return ase_potential_energy(
            batch_positions,
            atoms=self.atoms,
            batch_cell=batch_cell,
            positions_unit=self._positions_unit,
            energy_unit=self._energy_unit,
            parallelization_strategy=self.parallelization_strategy,
        )


# =============================================================================
# TORCH FUNCTIONAL API
# =============================================================================

class ASEPotentialEnergyFunc(torch.autograd.Function):
    """PyTorch-differentiable potential energy using ASE.

    This wraps an ASE calculator to perform batchwise energy and forces calculation
    used for the forward pass and backpropagation.

    .. warning::
        Currently double-backpropagation is not supported, which means force
        matching cannot be performed during training.

    By default, the perform the batch of energy/gradient calculations serially.
    This scheme is, however, not embarassingly parallel. Thus, the module supports
    batch parallelization schemes through :class:``tfep.utils.parallel.ParallelizationStrategy``s.

    Parameters
    ----------
    ctx : torch.autograd.function._ContextMethodMixin
        A context to save information for the gradient.
    batch_positions : torch.Tensor
        Shape ``(batch_size, 3*n_atoms)``. The atoms positions in units of
        ``positions_unit`` (or ASE units if ``positions_unit`` is not provided).
    atoms : ase.Atoms or list[ase.Atoms]
        The ASE ``Atoms`` object containing the ASE calculator.
    batch_cell : None or torch.Tensor
        Shape ``(batch_size, 3, 3)`` or ``(batch_size, 3)`` or ``(batch_size, 6)``.
        Unit cell vectors.  Can also be given as just three numbers for orthorhombic
        cells, or 6 numbers, where first three are lengths of unit cell vectors
        (in units of ``positions_unit`` or ASE units if ``positions_unit`` is not
        provided), and the other three are angles between them (in degrees), in
        following order: ``[len(a), len(b), len(c), angle(b,c), angle(a,c), angle(a,b)]``.
        First vector will lie in x-direction, second in xy-plane, and the third
        one in z-positive subspace.
    positions_unit : pint.Unit, optional
        The unit of the positions passed. This is used to appropriately convert
        ``batch_positions`` to ASE units. If ``None``, no conversion is performed,
        which assumes that the input positions are in the same units used by
        ASE.
    energy_unit : pint.Unit, optional
        The unit used for the returned energies (and as a consequence forces).
        This is used to appropriately convert ASE energies into the desired
        units. If ``None``, no conversion is performed, which means that energies
        and forces will use ASE units.
    parallelization_strategy : tfep.utils.parallel.ParallelizationStrategy, optional
        The parallelization strategy used to distribute batches of energy and
        gradient calculations. By default, these are executed serially.

    Returns
    -------
    potentials : torch.Tensor
        ``potentials[i]`` is the potential energy of configuration
        ``batch_positions[i]``.

    See Also
    --------
    :class:`.ASEPotential`
        ``Module`` API for computing potential energies with ASE.

    """

    @staticmethod
    def forward(
            ctx,
            batch_positions,
            atoms,
            batch_cell=None,
            positions_unit=None,
            energy_unit=None,
            parallelization_strategy=None,
    ):
        """Compute the potential energy of the molecule with ASE."""
        # Handle mutable default arguments.
        if parallelization_strategy is None:
            parallelization_strategy = SerialStrategy()

        # Convert tensor to numpy array with shape (batch_size, n_atoms, 3) and in ASE units (angstrom).
        batch_positions_arr_ang = flattened_to_atom(batch_positions.detach().cpu().numpy())
        if positions_unit is not None:
            batch_positions_arr_ang = _to_ase_units(batch_positions_arr_ang, positions_unit)

        # We need to convert also the cells in Angstroms.
        if batch_cell is not None:
            batch_cell_arr_ang = batch_cell.detach().cpu().numpy()
            if positions_unit is not None:
                if batch_cell.shape[1:] == (6,):
                    # The last 3 entires of each batch cell are angles, not lengths.
                    batch_cell_arr_ang[:, :3] = _to_ase_units(batch_cell_arr_ang[:, :3], positions_unit)
                else:
                    # All entries of batch_cells are lengths.
                    batch_cell_arr_ang = _to_ase_units(batch_cell_arr_ang, positions_unit)

        # We use functools.partial to encode the arguments that are common to all tasks.
        if batch_cell is None:
            task = functools.partial(
                _get_potential_energy_task, atoms, batch_cell)
            distributed_args = zip(batch_positions_arr_ang)
        else:
            task = functools.partial(
                _get_potential_energy_task, atoms)
            distributed_args = zip(batch_cell_arr_ang, batch_positions_arr_ang)

        # Run all batches with the provided parallelization strategy.
        batch_results = parallelization_strategy.run(task, distributed_args)

        # Unpack the results. From [(atoms, energy), ...] to ([atoms, ...], [energy, ...]).
        batch_atoms, energies = list(zip(*batch_results))

        # Convert energies to unitless tensors.
        if energy_unit is None:
            energies = torch.tensor(energies)
        else:
            energies *= ASEPotential.default_energy_unit(energy_unit._REGISTRY)
            energies = energies_array_to_tensor(energies, energy_unit)
        energies = energies.to(batch_positions)

        # Save the Atoms objects with the results of the calculation (including
        # forces) for backward propagation.
        ctx.batch_atoms = batch_atoms
        ctx.energy_unit = energy_unit
        ctx.positions_unit = positions_unit
        ctx.parallelization_strategy = parallelization_strategy

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
            # The Atoms object should already have precomputed forces stored.
            distributed_args = zip(ctx.batch_atoms)
            forces = ctx.parallelization_strategy.run(_get_forces_task, distributed_args)
            forces = np.stack(forces)

            # Convert to unitless tensors.
            if (ctx.energy_unit is None) and (ctx.positions_unit is None):
                forces = torch.from_numpy(atom_to_flattened(forces))
            else:
                ureg = ctx.energy_unit._REGISTRY
                default_positions_unit = ASEPotential.default_positions_unit(ureg)
                default_energy_unit = ASEPotential.default_energy_unit(ureg)
                forces *= default_energy_unit / default_positions_unit
                forces = forces_array_to_tensor(forces, ctx.positions_unit, ctx.energy_unit)
            forces = forces.to(grad_output)

            # Accumulate gradient
            grad_input[0] = -forces * grad_output[:, None]

        return tuple(grad_input)


def ase_potential_energy(
        batch_positions,
        atoms,
        batch_cell=None,
        positions_unit=None,
        energy_unit=None,
        parallelization_strategy=None,
):
    """PyTorch-differentiable potential energy using ASE.

    PyTorch ``Function``s do not accept keyword arguments. This function wraps
    :func:`.ASEPotentialEnergyFunc.apply` to enable standard functional notation.
    See the documentation on the original function for the input parameters.

    See Also
    --------
    :class:`.ASEPotentialEnergyFunc`
        More details on input parameters and implementation details.

    """
    # apply() does not accept keyword arguments.
    return ASEPotentialEnergyFunc.apply(
        batch_positions,
        atoms,
        batch_cell,
        positions_unit,
        energy_unit,
        parallelization_strategy,
    )


# =============================================================================
# RUNNING UTILITY FUNCTIONS
# =============================================================================

def _to_ase_units(x, positions_unit):
    """Convert x from positions_unit to angstroms."""
    default_positions_unit = ASEPotential.default_positions_unit(positions_unit._REGISTRY)
    return (x * positions_unit).to(default_positions_unit).magnitude


def _get_potential_energy_task(
    atoms,
    cell,
    positions,
):
    """Compute potential energy for a single configuration.

    This function is used as task function for a ParallelStrategy.

    Both positions and cell are expected to be numpy arrays and in units of
    Angstrom. The returned energies are in units of eV.

    """
    # Return a copy of the atoms instance which contain all the results (beside return value).
    atoms = copy.deepcopy(atoms)

    # Set positions and cell.
    atoms.set_positions(positions)
    if cell is not None:
        atoms.set_cell(cell)

    # Compute energies.
    energy = atoms.get_potential_energy()

    return atoms, energy


def _get_forces_task(atoms):
    """Compute forces for a single configuration.

    This function is used as task function for a ParallelStrategy.

    This assumes that atoms already has the positions/cell set.

    The returned forces are in units of eV/Angstrom.

    """
    return atoms.get_forces()
