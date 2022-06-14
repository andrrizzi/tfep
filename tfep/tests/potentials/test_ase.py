#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test objects and function in the module ``tfep.potentials.ase``.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

# Psi4 is an optional dependency of tfep.
try:
    import ase
except ImportError:
    ASE_INSTALLED = False
else:
    ASE_INSTALLED = True

import contextlib

import numpy as np
import pint
import pytest
import torch

from tfep.potentials.ase import PotentialASE, potential_energy_ase
from tfep.utils.misc import atom_to_flattened, flattened_to_atom
from tfep.utils.parallel import SerialStrategy, ProcessPoolStrategy


# =============================================================================
# GLOBAL VARIABLES
# =============================================================================

# Common unit registry for all tests.
_UREG = pint.UnitRegistry()


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

def create_two_waters(batch_size, calculator):
    """Crete an ase.Atoms object with two water molecules and batch positions.

    Returns an ase Atoms object and the batch positions (as tensor in flattened
    format).
    """
    atoms = ase.Atoms('OH2OH2', calculator=calculator)

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

    return atoms, batch_positions.reshape(batch_size, -1)


@contextlib.contextmanager
def parallelization_strategy(strategy_name):
    """Context manager safely creating/destroying the parallelization strategy."""
    if strategy_name == 'serial':
        yield SerialStrategy()
    else:
        # Keep the pool of processes open until the contextmanager has left.
        with torch.multiprocessing.Pool(2) as p:
            yield ProcessPoolStrategy(p)


def reference_energy_forces(atoms, batch_positions):
    """Compute the energy and force of atoms at batch_positions.

    Expects batch_positions in units of Angstrom and returns energies(forces)
    in units of kcal/mol(*A).
    """
    from ase.units import kcal, mol

    batch_positions = flattened_to_atom(batch_positions.detach()).numpy()
    batch_size, n_atoms, _ = batch_positions.shape

    energies = np.empty(shape=(batch_size,))
    forces = np.empty(shape=batch_positions.shape)
    for i, positions in enumerate(batch_positions):
        atoms.set_positions(positions)
        energies[i] = atoms.get_potential_energy()  # in eV.
        forces[i] = atoms.get_forces()  # in eV/A.

    # Convert units.
    ev_to_kcalmol = mol / kcal
    energies = torch.tensor(energies) * ev_to_kcalmol
    forces = torch.tensor(forces) * ev_to_kcalmol

    return energies, forces

# =============================================================================
# TESTS
# =============================================================================

# TODO: TEST ALSO batch_cell with pbc=True?

@pytest.mark.skipif(not ASE_INSTALLED, reason='requires ASE to be installed')
@pytest.mark.parametrize('batch_size', [1, 2])
@pytest.mark.parametrize('strategy', ['serial', 'pool'])
def test_potential_ase_energy_force(batch_size, strategy):
    """Test the calculation of energies/forces with PotentialASE.

    This tests that:
    - Energies/forces from ASE calculators are handled correctly.
    - Test Serial and ProcessPool parallelization strategies.

    """
    from ase.calculators.tip3p import TIP3P

    calculator = TIP3P()
    atoms, batch_positions = create_two_waters(batch_size, calculator)
    batch_positions.requires_grad = True

    # Compute reference energies and forces.
    ref_energies, ref_forces = reference_energy_forces(atoms, batch_positions)

    with parallelization_strategy(strategy) as ps:
        potential = PotentialASE(
            calculator=calculator,
            symbols=atoms.symbols,
            position_unit=_UREG.angstrom,
            energy_unit=_UREG.kcal/_UREG.mole,
            parallelization_strategy=ps,
        )

        # Compute energy and compare to reference.
        energies = potential(batch_positions)
        assert torch.allclose(energies, ref_energies)

        # Compute forces (negative gradient).
        energies.sum().backward()
        forces = -batch_positions.grad
        assert torch.allclose(forces, atom_to_flattened(ref_forces))


@pytest.mark.skipif(not ASE_INSTALLED, reason='requires ASE to be installed')
def test_potential_energy_ase_gradcheck():
    """Test that potential_energy_ase implements the correct gradient."""
    from ase.calculators.tip3p import TIP3P

    batch_size = 2
    calculator = TIP3P()
    atoms, batch_positions = create_two_waters(batch_size, calculator)
    batch_positions.requires_grad = True

    # Run gradcheck. We keep precompute_gradient = False because gradcheck
    # performs a bunch of forward without backward.
    torch.autograd.gradcheck(
        func=potential_energy_ase,
        inputs=[
            batch_positions,
            atoms,
            None,  # batch_cell
            None,  # positions_unit
            None,  # energy_unit
            None,  # parallelization_strategy
        ],
    )