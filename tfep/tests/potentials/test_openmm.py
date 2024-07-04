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

import contextlib

# OpenMM is an optional dependency of tfep.
try:
    import openmm
except ImportError:
    OPENMM_INSTALLED = False
else:
    OPENMM_INSTALLED = True
    import openmm.app
    import openmm.unit

from MDAnalysis.lib.mdamath import triclinic_vectors
import numpy as np
import pint
import pytest
import torch

import tfep.potentials.openmm
from tfep.potentials.openmm import OpenMMPotential, openmm_potential_energy
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
    # Clear the OpenMM Context cache.
    tfep.potentials.openmm.global_context_cache = {}
    import gc
    gc.collect()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

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


def create_two_waters(batch_size, seed=42):
    """Crete a System object with two water molecules and batch positions.

    Returns a Context object, the batch positions (as tensor in flattened
    format) in Angstrom and the batch unit cell as a 6-element vector where
    the cell lengths are in Angstroms and angles in degrees.
    """
    # Create two tip3p water molecules.
    n_waters = 2
    model = 'tip3p'

    # Force field.
    force_field = openmm.app.ForceField(model + '.xml')

    # Modeller.
    modeller = openmm.app.Modeller(
        openmm.app.Topology(),
        openmm.unit.Quantity((), openmm.unit.angstroms)
    )
    modeller.addSolvent(
        force_field,
        model=model,
        numAdded=n_waters,
    )

    # OpenMM System.
    system = force_field.createSystem(
        modeller.getTopology(),
        nonbondedCutoff=openmm.NonbondedForce.PME,
        constraints=None,
        rigidWater=False,
        removeCMMotion=False,
    )

    # Basic positions to be perturbed.
    positions = torch.tensor([
        [-0.956, -0.121, 0],
        [-1.308, 0.770, 0],
        [0.000, 0.000, 0],
        [3.903, 0.000, 0],
        [4.215, -0.497, -0.759],
        [4.215, -0.497, 0.759]
    ])  # in angstrom.

    # Create small random perturbations around the initial geometry.
    # Using a RandomState with fixed seed makes it deterministic.
    random_state = np.random.RandomState(seed)
    batch_positions = torch.empty(batch_size, positions.shape[0], 3)
    for batch_idx in range(batch_size):
        perburbation = random_state.uniform(-0.2, 0.2, size=positions.shape)
        batch_positions[batch_idx] = positions + perburbation

    # Constant unit cell.
    batch_cells = torch.empty(batch_size, 6)
    batch_cells[:, :3] = 10.  # angstrom
    batch_cells[:, 3:] = 90.  # degrees

    return system, batch_positions.reshape(batch_size, -1), batch_cells


def two_atoms_bonded_in_vacuum(batch_size=2, seed=42):
    """Two atoms with a single harmonic bonds in vacuum.

    Returns a Context object and the batch positions (as tensor in flattened
    format) in nanometers.
    """
    # Harmonic bond parameters.
    r0=0.155*openmm.unit.nanometers
    K=290.1*openmm.unit.kilocalories_per_mole/openmm.unit.angstrom**2

    # Create system with two particles.
    system = openmm.System()
    system.addParticle(39.948*openmm.unit.amu)
    system.addParticle(39.948*openmm.unit.amu)

    # Add bond.
    force = openmm.HarmonicBondForce()
    force.addBond(0, 1, r0, K)
    system.addForce(force)

    # Equilibrium position.
    eq_positions = torch.zeros(2, 3)
    eq_positions[1, 0] = r0._value

    # Perturb equilibrium positions using a RandomState with fixed seed makes it deterministic.
    random_state = np.random.RandomState(seed)
    batch_positions = torch.empty(batch_size, *eq_positions.shape)
    for batch_idx in range(batch_size):
        perburbation = random_state.uniform(-0.02, 0.02, size=eq_positions.shape)  # nanometers
        batch_positions[batch_idx] = eq_positions + perburbation

    return system, batch_positions.reshape(batch_size, -1)


def reference_energy_forces(system, batch_positions, batch_cell):
    """Compute the energy and force of atoms at batch_positions.

    Expects batch_positions/cell in units of Angstrom and returns energies(forces)
    in units of kcal/mol(*A).
    """
    context = openmm.Context(
        system,
        openmm.VerletIntegrator(0.001),
        openmm.Platform.getPlatformByName('CPU'),
    )
    batch_positions = flattened_to_atom(batch_positions.detach()).numpy()
    batch_size, n_atoms, _ = batch_positions.shape

    energies = []
    forces = []
    for i, (positions, cell) in enumerate(zip(batch_positions, batch_cell)):
        # OpenMM internal units are nanometers.
        context.setPeriodicBoxVectors(*(triclinic_vectors(cell) / 10))
        context.setPositions(positions / 10)
        state = context.getState(getEnergy=True, getForces=True)
        energies.append(state.getPotentialEnergy())  # in kJ/mol.
        forces.append(state.getForces(asNumpy=True))  # in kJ/mol/nm.

    # Convert units.
    kcal_mole = openmm.unit.kilocalories / openmm.unit.mole
    kcal_mole_A = kcal_mole / openmm.unit.angstroms
    energies = torch.tensor([e.value_in_unit(kcal_mole) for e in energies])
    forces = torch.tensor([f.value_in_unit(kcal_mole_A) for f in forces])

    return energies, forces

# =============================================================================
# TESTS
# =============================================================================

@pytest.mark.skipif(not OPENMM_INSTALLED, reason='requires OPENMM to be installed')
@pytest.mark.parametrize('batch_size', [1, 2])
@pytest.mark.parametrize('platform', [None, 'CPU'])
@pytest.mark.parametrize('strategy', [None, 'serial', 'pool'])
def test_openmm_potential_energy_force(batch_size, platform, strategy):
    """Test the calculation of energies/forces with OpenMMPotential."""
    system, batch_positions, batch_cell = create_two_waters(batch_size)
    batch_positions.requires_grad = True

    # Compute reference energies and forces.
    ref_energies, ref_forces = reference_energy_forces(system, batch_positions, batch_cell)

    with parallelization_strategy(strategy) as ps:
        # Compute through OpenMMPotential.
        potential = OpenMMPotential(
            system=system,
            platform=platform,
            positions_unit=_UREG.angstrom,
            energy_unit=_UREG.kcal/_UREG.mole,
            system_name='two_waters',
            precompute_gradient=True,
            parallelization_strategy=ps,
        )

        # Compute energy and compare to reference.
        energies = potential(batch_positions, batch_cell)
    assert torch.allclose(energies, ref_energies)

    # Compute forces (negative gradient).
    energies.sum().backward()
    forces = -batch_positions.grad
    assert torch.allclose(forces, atom_to_flattened(ref_forces))


@pytest.mark.skipif(not OPENMM_INSTALLED, reason='requires OpenMM to be installed')
def test_openmm_context_cache():
    """The OpenMM caching of Context objects works correctly."""
    batch_size = 2

    # Clear cache to start the test.
    tfep.potentials.openmm.global_context_cache = {}

    # Initialize two different potentials.
    system1, batch_positions1, batch_cell1 = create_two_waters(batch_size)
    potential1 = OpenMMPotential(
        system=system1,
        platform='CPU',
        positions_unit=_UREG.angstrom,
        system_name='system1',
    )

    system2, batch_positions2 = two_atoms_bonded_in_vacuum(batch_size=batch_size)
    potential2 = OpenMMPotential(
        system=system2,
        platform='CPU',
        positions_unit=_UREG.angstrom,
        system_name='system2',
    )

    # Compute the energy of two different systems and cache the Contexts.
    energies1 = potential1(batch_positions1, batch_cell1)
    assert len(tfep.potentials.openmm.global_context_cache) == 1
    assert 'system1' in tfep.potentials.openmm.global_context_cache
    energies2 = potential2(batch_positions2)
    assert len(tfep.potentials.openmm.global_context_cache) == 2
    assert 'system2' in tfep.potentials.openmm.global_context_cache
    assert not torch.any(torch.isclose(energies1, energies2))

    # Re-compute them and check that the system retrieves the correct Context.
    assert torch.allclose(energies1, potential1(batch_positions1, batch_cell1))
    assert torch.allclose(energies2, potential2(batch_positions2))
    assert len(tfep.potentials.openmm.global_context_cache) == 2
    assert set(tfep.potentials.openmm.global_context_cache.keys()) == {'system1', 'system2'}

    # The same is true when caching is not activated.
    potential1 = OpenMMPotential(system=system1, platform='CPU', positions_unit=_UREG.angstrom)
    potential2 = OpenMMPotential(system=system2, platform='CPU', positions_unit=_UREG.angstrom)
    assert torch.allclose(energies1, potential1(batch_positions1, batch_cell1))
    assert torch.allclose(energies2, potential2(batch_positions2))
    assert len(tfep.potentials.openmm.global_context_cache) == 2
    assert set(tfep.potentials.openmm.global_context_cache.keys()) == {'system1', 'system2'}

    # Clear cache for other tests.
    tfep.potentials.openmm.global_context_cache = {}


@pytest.mark.skipif(not OPENMM_INSTALLED, reason='requires OpenMM to be installed')
def test_openmm_potential_energy_gradcheck():
    """Test that openmm_potential_energy implements the correct gradient."""
    system, batch_positions = two_atoms_bonded_in_vacuum(batch_size=2)
    batch_positions.requires_grad = True

    # Run gradcheck.
    torch.autograd.gradcheck(
        func=openmm_potential_energy,
        inputs=[
            batch_positions,
            system,
            openmm.Platform.getPlatformByName('CPU'),  # platform
            None,  # batch cell
            None,  # positions_unit
            None,  # energy_unit
            None,  # system_name
            True,  # precompute_gradient
            None,  # parallelization_strategy
        ],    )
