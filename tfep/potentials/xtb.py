#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Modules and functions to compute semiempirical QM energies and gradients with tblite.

The function/classes in this module wrap a tblite ``Calculator``s and makes it
compatible with PyTorch.

See Also
--------
tblite: https://tblite.readthedocs.io/en/latest/index.html

"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

# DO NOT IMPORT tblite HERE! tblite is an optional dependency of tfep.
import functools
import logging
from typing import Optional

import numpy as np
import pint
import torch

from tfep.potentials.base import PotentialBase
from tfep.utils.misc import (
    atom_to_flattened, flattened_to_atom,
    energies_array_to_tensor, forces_array_to_tensor
)
from tfep.utils.parallel import ParallelizationStrategy, SerialStrategy


# =============================================================================
# GLOBAL VARIABLES
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# TORCH MODULE API
# =============================================================================

class TBLitePotential(PotentialBase):
    """Potential energy and gradients with tblite.

    This ``Module`` wraps :class:``.TBLitePotentialEnergyFunc`` to provide a
    differentiable potential energy function for training.

    .. warning::
        Currently double-backpropagation is not supported, which means force
        matching cannot be performed during training.

    """

    #: The default energy unit.
    DEFAULT_ENERGY_UNIT : str = 'hartree'

    #: The default positions unit.
    DEFAULT_POSITIONS_UNIT : str = 'bohr'

    def __init__(
            self,
            method: str,
            numbers: np.ndarray,
            positions_unit: Optional[pint.Unit] = None,
            energy_unit: Optional[pint.Unit] = None,
            precompute_gradient: bool = False,
            parallelization_strategy: Optional[ParallelizationStrategy] = None,
            verbosity: int = 0,
            return_nan_on_failure: bool = False,
    ):
        """Constructor.

        Parameters
        ----------
        method : str
            The method. Example ``'GFN2-xTB'``.
        numbers: numpy.ndarray
            Atomic numbers. Examples ``[8, 1, 1]`` for water.
        positions_unit : pint.Unit, optional
            The unit of the positions passed to the class methods. Since input
            ``Tensor``s do not have units attached, this is used to appropriately
            convert ``batch_positions`` to tblite units. If ``None``, no conversion
            is performed, which assumes that the input positions are in the same
            units used by tblite (e.g., bohr).
        energy_unit : pint.Unit, optional
            The unit used for the returned energies (and as a consequence gradients).
            Since ``Tensor``s do not have units attached, this is used to
            appropriately convert tblite energies into the desired units. If ``None``
            is performed, which means that energies and gradients will be returned
            in tblite units (e.g., eV).
        precompute_gradient : bool, optional
            If ``True``, the gradient is computed in the forward pass and saved
            to be consumed during the backward pass. This speeds up the training,
            but should be deactivated if gradients are not needed. Setting this
            to ``False`` (default) will cause an exception if a backward pass is
            attempted.
        parallelization_strategy : tfep.utils.parallel.ParallelizationStrategy, optional
            The parallelization strategy used to distribute batches of energy and
            gradient calculations. By default, these are executed serially.
        verbosity : Literal[0, 1, 2], optional
            Verbosity to pass to the tblite ``Calculator`` (0 turns off printing,
            2 is maximum verbosity).
        return_nan_on_failure : bool, optional
            If ``True`` and the single-point calculation fails (e.g., because the
            the SCF does not converge), the returned energy (gradient) is set to
            NaN (zero).

        See Also
        --------
        :class:`.TBLitePotentialEnergyFunc`
            More details on input parameters and implementation details.

        """
        super().__init__(positions_unit=positions_unit, energy_unit=energy_unit)

        # Handle mutable default arguments.
        if parallelization_strategy is None:
            parallelization_strategy = SerialStrategy()

        #: The method for tblite.
        self.method = method

        #: The atomic numbers.
        self.numbers = numbers

        #: Whether to compute the gradients in the forward pass to speed up the backward pass.
        self.precompute_gradient = precompute_gradient

        #: The strategy used to parallelize the single-point calculations.
        self.parallelization_strategy = parallelization_strategy

    def forward(self, batch_positions: torch.Tensor) -> torch.Tensor:
        """Compute a differential potential energy for a batch of configurations.

        Parameters
        ----------
        batch_positions : torch.Tensor
            Shape ``(batch_size, 3*n_atoms)``. The atoms positions in units of
            ``self.positions_unit``.

        Returns
        -------
        potential_energy : torch.Tensor
            ``potential_energy[i]`` is the potential energy of configuration
            ``batch_positions[i]`` in units of ``self.energy_unit``.

        """
        return tblite_potential_energy(
            batch_positions,
            method=self.method,
            numbers=self.numbers,
            positions_unit=self.positions_unit,
            energy_unit=self.energy_unit,
            precompute_gradient=self.precompute_gradient,
            parallelization_strategy=self.parallelization_strategy,
        )


# =============================================================================
# TORCH FUNCTIONAL API
# =============================================================================

class TBLitePotentialEnergyFunc(torch.autograd.Function):
    """PyTorch-differentiable potential energy using tblite.

    This wraps a tblite ``Calculator`` to perform batchwise energy and gradients
    calculation used for the forward pass and backpropagation.

    .. warning::
        Currently double-backpropagation is not supported, which means force
        matching cannot be performed during training.

    By default, the perform the batch of energy/gradient calculations serially.
    This scheme is, however, embarassingly parallel. Thus, the module supports
    batch parallelization schemes through :class:``tfep.utils.parallel.ParallelizationStrategy``s.

    Parameters
    ----------
    ctx : torch.autograd.function._ContextMethodMixin
        A context to save information for the gradient.
    batch_positions : torch.Tensor
        Shape ``(batch_size, 3*n_atoms)``. The atoms positions in units of
        ``positions_unit`` (or tblite units if ``positions_unit`` is not provided).
    method : str
        The method. Example ``'GFN2-xTB'``.
    numbers: numpy.ndarray
        Atomic numbers. Examples ``[8, 1, 1]`` for water.
    positions_unit : pint.Unit, optional
        The unit of the positions passed. This is used to appropriately convert
        ``batch_positions`` to the units used by tblite. If ``None``, no conversion
        is performed, which assumes that the input positions are in Bohr.
    energy_unit : pint.Unit, optional
        The unit used for the returned energies (and as a consequence gradients).
        This is used to appropriately convert tblite energies into the desired
        units. If ``None``, no conversion is performed, which means that energies
        and gradients will be in hartrees and hartrees/bohr respectively.
    precompute_gradient : bool, optional
        If ``True``, the gradient is computed in the forward pass and saved to
        be consumed during the backward pass. This speeds up the training, but
        should be deactivated if gradients are not needed. Setting this to
        ``False`` (default) will cause an exception if a backward pass is
        attempted.
    parallelization_strategy : tfep.utils.parallel.ParallelizationStrategy, optional
        The parallelization strategy used to distribute batches of energy and
        gradient calculations. By default, these are executed serially.
    verbosity : Literal[0, 1, 2], optional
        Verbosity to pass to the tblite ``Calculator`` (0 turns off printing,
        2 is maximum verbosity).
    return_nan_on_failure : bool, optional
        If ``True`` and the single-point calculation fails (e.g., because the
        the SCF does not converge), the returned energy (gradient) is set to
        NaN (zero).

    Returns
    -------
    potentials : torch.Tensor
        ``potentials[i]`` is the potential energy of configuration
        ``batch_positions[i]``.

    See Also
    --------
    :class:`.TBLitePotential`
        ``Module`` API for computing potential energies with tblite.

    """

    @staticmethod
    def forward(
            ctx,
            batch_positions: torch.Tensor,
            method: str,
            numbers: np.ndarray,
            positions_unit: Optional[pint.Unit] = None,
            energy_unit: Optional[pint.Unit] = None,
            precompute_gradient: bool = False,
            parallelization_strategy: Optional[ParallelizationStrategy] = None,
            verbosity: int = 0,
            return_nan_on_failure: bool = False,
    ) -> torch.Tensor:
        """Compute the potential energy of the molecule with tblite."""
        # Handle mutable default arguments.
        if parallelization_strategy is None:
            parallelization_strategy = SerialStrategy()

        # Convert tensor to numpy array with shape (batch_size, n_atoms, 3) and in tblite units (angstrom).
        batch_positions_arr_bohr = flattened_to_atom(batch_positions.detach().cpu().numpy())
        if positions_unit is not None:
            batch_positions_arr_bohr = _to_tblite_units(batch_positions_arr_bohr, positions_unit)

        # We use functools.partial to encode the arguments that are common to all tasks.
        task = functools.partial(_run_single_point, method, numbers, precompute_gradient, verbosity, return_nan_on_failure)
        distributed_args = zip(batch_positions_arr_bohr)

        # Run all batches with the provided parallelization strategy.
        batch_results = parallelization_strategy.run(task, distributed_args)

        # Unpack the results. From [(energy, gradients), ...] to ([energy, ...], [gradients, ...]).
        if precompute_gradient:
            energies, gradients = list(zip(*batch_results))
            # From Tuple[ndarray] to ndarray.
            gradients = np.array(gradients)

        # Convert energies to unitless tensors.
        if energy_unit is None:
            energies = torch.from_numpy(np.array(energies))
        else:
            energies *= TBLitePotential.default_energy_unit(energy_unit._REGISTRY)
            energies = energies_array_to_tensor(energies, energy_unit)
        energies = energies.to(batch_positions)

        if logger.isEnabledFor(logging.INFO):
            nan_energies = torch.isnan(energies)
            if nan_energies.any():
                logger.info(f'Found {torch.sum(nan_energies)} NaN {method} energies.')

        # Save the gradients for backward propagation. We do not support backward
        # passes with precompute_gradient=False.
        if precompute_gradient:
            ctx.gradients = gradients
            ctx.energy_unit = energy_unit
            ctx.positions_unit = positions_unit
        else:
            ctx.gradients = None

        return energies

    @staticmethod
    def backward(ctx, grad_output):
        """Compute the gradient of the potential energy."""
        # We still need to return a None gradient for each
        # input of forward() beside batch_positions.
        n_input_args = 9
        grad_input = [None for _ in range(n_input_args)]

        # Compute gradient w.r.t. batch_positions.
        if ctx.needs_input_grad[0]:
            # Check that gradients are available.
            if ctx.gradients is None:
                raise ValueError('precompute_gradient must be set to True for backward pass.')

            # Convert to unitless tensors.
            if (ctx.energy_unit is None) and (ctx.positions_unit is None):
                gradients = torch.from_numpy(atom_to_flattened(ctx.gradients))
            else:
                ureg = ctx.energy_unit._REGISTRY
                default_positions_unit = TBLitePotential.default_positions_unit(ureg)
                default_energy_unit = TBLitePotential.default_energy_unit(ureg)
                gradients = ctx.gradients * default_energy_unit / default_positions_unit
                gradients = forces_array_to_tensor(gradients, ctx.positions_unit, ctx.energy_unit)
            gradients = gradients.to(grad_output)

            # Accumulate gradient
            grad_input[0] = gradients * grad_output[:, None]

        return tuple(grad_input)


def tblite_potential_energy(
        batch_positions: torch.Tensor,
        method: str,
        numbers: np.ndarray,
        positions_unit: Optional[pint.Unit] = None,
        energy_unit: Optional[pint.Unit] = None,
        precompute_gradient: bool = False,
        parallelization_strategy: Optional[ParallelizationStrategy] = None,
        verbosity: int = 0,
        return_nan_on_failure: bool = False,
):
    """PyTorch-differentiable potential energy using tblite.

    PyTorch ``Function``s do not accept keyword arguments. This function wraps
    :func:`.TBLitePotentialEnergyFunc.apply` to enable standard functional notation.
    See the documentation on the original function for the input parameters.

    See Also
    --------
    :class:`.TBLitePotentialEnergyFunc`
        More details on input parameters and implementation details.

    """
    # apply() does not accept keyword arguments.
    return TBLitePotentialEnergyFunc.apply(
        batch_positions,
        method,
        numbers,
        positions_unit,
        energy_unit,
        precompute_gradient,
        parallelization_strategy,
    )


# =============================================================================
# RUNNING UTILITY FUNCTIONS
# =============================================================================

def _to_tblite_units(x, positions_unit):
    """Convert x from positions_unit to angstroms."""
    default_positions_unit = TBLitePotential.default_positions_unit(positions_unit._REGISTRY)
    return (x * positions_unit).to(default_positions_unit).magnitude


def _run_single_point(
        method: str,
        numbers: np.ndarray,
        return_gradients: bool,
        verbosity: int,
        return_nan_on_failure: bool,
        positions: np.ndarray,
):
    """Compute potential energy for a single configuration.

    This function is used as task function for a ParallelStrategy.

    Both positions are expected to be numpy arrays and in units of
    bohr. The returned energies are in units of hartree.

    """
    from tblite.interface import Calculator

    calc = Calculator(method, numbers, positions)
    calc.set('verbosity', verbosity)

    try:
        res = calc.singlepoint()  # energy printed is only the electronic part
    except RuntimeError as e:
        if return_nan_on_failure:
            if return_gradients:
                return [np.nan, np.zeros_like(positions)]
            return np.nan
        raise

    energy = res.get('energy')
    if return_gradients:
        return [energy, res.get('gradient')]
    return energy
