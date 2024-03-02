#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test objects and function in the module ``tfep.potentials.tblite``.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

# tblite is an optional dependency of tfep.
try:
    import tblite
except ImportError:
    TBLITE_INSTALLED = False
else:
    TBLITE_INSTALLED = True
    from tblite.interface import Calculator

import contextlib

import numpy as np
import pint
import pytest
import torch

from tfep.potentials.tblite import TBLitePotential, tblite_potential_energy
from tfep.utils.misc import atom_to_flattened, flattened_to_atom
from tfep.utils.parallel import SerialStrategy, ProcessPoolStrategy


# =============================================================================
# GLOBAL VARIABLES
# =============================================================================

# Common unit registry for all tests.
_UREG = pint.UnitRegistry()

# UNIT CONVERSION CONSTANTS
_ANGSTROM_TO_BOHR = 1.8897259886
_HARTREE_TO_KCALMOL = 627.5096080305927


# =============================================================================
# TEST MODULE CONFIGURATION
# =============================================================================

_old_default_dtype = None

def setup_module(module):
    global _old_default_dtype
    _old_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.double)


def teardown_module(module):
    torch.set_default_dtype(_old_default_dtype)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_two_waters(batch_size):
    """Crete two water molecules and batch positions.

    The return positions are in Angstrom.
    """
    # Atomic numbers
    numbers = np.array([8, 1, 1, 8, 1, 1])

    # Basic positions to be perturbed.
    positions = torch.tensor([
        [-0.956, -0.121, 0],
        [-1.308, 0.770, 0],
        [0.000, 0.000, 0],
        [3.903, 0.000, 0],
        [4.215, -0.497, -0.759],
        [4.215, -0.497, 0.759]
    ])  # in Angstrom.

    # Create small random perturbations around the initial geometry.
    # Using a RandomState with fixed seed makes it deterministic.
    random_state = np.random.RandomState(6378)
    batch_positions = torch.empty(batch_size, positions.shape[0], 3)
    for batch_idx in range(batch_size):
        perburbation = random_state.uniform(-0.2, 0.2, size=positions.shape)
        batch_positions[batch_idx] = positions + perburbation

    return numbers, batch_positions.reshape(batch_size, -1)


@contextlib.contextmanager
def parallelization_strategy(strategy_name):
    """Context manager safely creating/destroying the parallelization strategy."""
    if strategy_name is None:
        yield None  # Default parallelization strategy.
    elif strategy_name == 'serial':
        yield SerialStrategy()
    else:
        # Keep the pool of processes open until the contextmanager has left.
        mp_context = torch.multiprocessing.get_context('forkserver')
        with mp_context.Pool(2) as p:
            yield ProcessPoolStrategy(p)


def reference_energy_gradients(numbers, batch_positions):
    """Compute the energy and gradients of the molecule at batch_positions.

    Expects batch_positions in units of Angstrom and returns energies(gradients)
    in units of kcal/mol(*A).
    """
    batch_positions = flattened_to_atom(batch_positions.detach()).numpy()
    batch_size, n_atoms, _ = batch_positions.shape

    energies = np.empty(shape=(batch_size,))
    gradients = np.empty(shape=batch_positions.shape)
    for i, positions in enumerate(batch_positions):
        calc = Calculator('GFN2-xTB', numbers, positions * _ANGSTROM_TO_BOHR)
        res = calc.singlepoint()
        energies[i] = res.get('energy')  # in hartree.
        gradients[i] = res.get('gradient')  # in hartree/bohr.

    # Convert units.
    energies = torch.tensor(energies) * _HARTREE_TO_KCALMOL
    gradients = torch.tensor(gradients) * _HARTREE_TO_KCALMOL * _ANGSTROM_TO_BOHR

    return energies, gradients


# =============================================================================
# TESTS
# =============================================================================

@pytest.mark.skipif(not TBLITE_INSTALLED, reason='requires tblite to be installed')
@pytest.mark.parametrize('batch_size', [1, 2])
@pytest.mark.parametrize('strategy', [None, 'serial', 'pool'])
def test_potential_tblite_energy_gradient(batch_size, strategy):
    """Test the calculation of energies/gradients with TBLitePotential.

    This tests that:
    - Energies/gradients are handled correctly.
    - Test Serial and ProcessPool parallelization strategies.

    """
    numbers, batch_positions = create_two_waters(batch_size)
    batch_positions.requires_grad = True

    # Compute reference energies and gradients.
    ref_energies, ref_gradients = reference_energy_gradients(numbers, batch_positions)

    with parallelization_strategy(strategy) as ps:
        potential = TBLitePotential(
            method='GFN2-xTB',
            numbers=numbers,
            positions_unit=_UREG.angstrom,
            energy_unit=_UREG.kcal/_UREG.mole,
            precompute_gradient=True,
            parallelization_strategy=ps,
        )

        # Compute energy and compare to reference.
        energies = potential(batch_positions)
        assert torch.allclose(energies, ref_energies)

        # Compute gradients (negative gradient).
        energies.sum().backward()
        assert torch.allclose(batch_positions.grad, atom_to_flattened(ref_gradients))


@pytest.mark.skipif(not TBLITE_INSTALLED, reason='requires tblite to be installed')
def test_tblite_potential_energy_gradcheck():
    """Test that tblite_potential_energy implements the correct gradient."""
    batch_size = 2
    numbers, batch_positions = create_two_waters(batch_size)
    batch_positions.requires_grad = True

    # Run gradcheck.
    torch.autograd.gradcheck(
        func=tblite_potential_energy,
        inputs=[
            batch_positions * _ANGSTROM_TO_BOHR,
            'GFN2-xTB',
            numbers,
            None,  # positions_unit
            None,  # energy_unit
            True,  # precompute_gradient
            None,  # parallelization_strategy
            0,     # verbosity
            False, # return_nan_on_failure
        ],
    )
