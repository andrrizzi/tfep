#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test objects and function in the module ``tfep.potentials.psi4``.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

# Psi4 is an optional dependency of tfep.
try:
    import psi4
except ImportError:
    PSI4_INSTALLED = False
else:
    PSI4_INSTALLED = True

import contextlib
import os
import tempfile

import numpy as np
import pint
import pytest
import torch

from tfep.utils.parallel import SerialStrategy, ProcessPoolStrategy
from tfep.potentials.psi4 import (
    create_psi4_molecule, configure_psi4, _run_psi4,
    potential_energy_psi4
)


# =============================================================================
# GLOBAL VARIABLES
# =============================================================================

# Common unit registry for all tests.
_UREG = pint.UnitRegistry()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_water_molecule(batch_size=None, **kwargs):
    """Crete a water Molecule object with batch positions.

    Returns a Psi4 molecule object and the batch positions. If batch_size is
    None, the returned batch_positions is also None.
    """
    # Water molecule with basic positions.
    positions = np.array([
        [-0.2950, -0.2180, 0.1540],
        [-0.0170, 0.6750, 0.4080],
        [0.3120, -0.4570, -0.5630],
    ], dtype=np.double) * _UREG.angstrom

    molecule = create_psi4_molecule(positions=positions, elem=['O', 'H', 'H'], **kwargs)

    if batch_size is not None:
        # Create small random perturbations around the initial geometry.
        # Using a RandomState with fixed seed makes it deterministic.
        random_state = np.random.RandomState(0)
        batch_positions = np.empty(shape=(batch_size, positions.shape[0], 3), dtype=np.double)
        batch_positions *= _UREG.angstrom
        for batch_idx in range(batch_size):
            perburbation = random_state.uniform(-0.3, 0.3, size=positions.shape)
            batch_positions[batch_idx] = positions + perburbation*_UREG.angstrom
    else:
        batch_positions = None

    return molecule, batch_positions


def pool_process_initializer(psi4_config):
    """Initialize a subprocess for pool parallelization."""
    molecule, _ = create_water_molecule()

    # Use a different scratch dir for each process.
    if 'psi4_scratch_dir_path' in psi4_config:
        subdir = os.path.join(psi4_config['psi4_scratch_dir_path'], str(os.getpid()))
        os.makedirs(subdir, exist_ok=True)
        psi4_config['psi4_scratch_dir_path'] = subdir
    configure_psi4(active_molecule=molecule, **psi4_config)


@contextlib.contextmanager
def parallelization_strategy(strategy_name, psi4_config):
    """Context manager safely creating/destroying the parallelization strategy."""
    if strategy_name == 'serial':
        yield SerialStrategy()
    else:
        # Keep the pool of processes open until the contextmanager is left.
        with torch.multiprocessing.Pool(2, pool_process_initializer, initargs=[psi4_config]) as p:
            yield ProcessPoolStrategy(p)


# =============================================================================
# TESTS
# =============================================================================

@pytest.mark.skipif(not PSI4_INSTALLED, reason='requires a Python installation of Psi4')
@pytest.mark.parametrize('batch_size', [None, 1, 2])
def test_run_psi4_energy(batch_size):
    """Test that the calculation of energies with _run_psi4.

    This tests that:
    - Both single and batch configurations are handled correctly and the shape
      of the returned quantity is correct.
    - Molecule can be None if it is set active beforehand.
    - The return_wfn argument works.
    - The write_orbitals argument leaves the orbitals at the given path.
    - Test Serial and ProcessPool parallelization strategies.

    """
    import psi4

    molecule, batch_positions = create_water_molecule(batch_size=batch_size)

    # Determine the expected potentials.
    if batch_size is None:
        expected_potentials = -76.05605256451271 * _UREG.hartree
    else:
        expected_potentials = np.array([-75.96441684781715, -76.03989190203629]) * _UREG.hartree
        expected_potentials = expected_potentials[:batch_size]

    # Common kwargs to all _run_psi4 calls.
    kwargs = {'name': 'scf', 'return_energy': True}

    with tempfile.TemporaryDirectory() as tmp_dir_path:
        # Set global options.
        psi4_config = dict(
            global_options=dict(basis='cc-pvtz', reference='RHF'),
            psi4_output_file_path='quiet',
            psi4_scratch_dir_path=tmp_dir_path,
        )
        configure_psi4(**psi4_config)

        # Determine path to cached wavefunctions.
        if batch_size is None:
            cached_wfn_file_path = os.path.join(tmp_dir_path, 'wfn')
        else:
            cached_wfn_file_path = [os.path.join(tmp_dir_path, 'wfn'+str(i)) for i in range(batch_size)]

        # First run passing explicitly the molecule and save the optimized wavefunctions.
        potentials = _run_psi4(molecule=molecule, write_orbitals=cached_wfn_file_path,
                               batch_positions=batch_positions, **kwargs)
        assert np.allclose(potentials.magnitude, expected_potentials.magnitude)
        if batch_size is None:
            assert os.path.isfile(cached_wfn_file_path + '.npy')
        else:
            assert all([os.path.isfile(p + '.npy') for p in cached_wfn_file_path])

        # At this point we should be able to run without the molecule since it has
        # been already activated. Try different parallelization strategies.
        for strategy in ['serial', 'pool']:
            with parallelization_strategy(strategy, psi4_config) as ps:
                potentials = _run_psi4(batch_positions=batch_positions,
                                       parallelization_strategy=ps, **kwargs)
            assert np.allclose(potentials.magnitude, expected_potentials.magnitude)

        # And re-use previously optimized wavefunctions. For this we don't use
        # the parallelization strategy as Wavefunctions cannot be pickled.
        potentials, wfn = _run_psi4(batch_positions=batch_positions, return_wfn=True,
                                    restart_file=cached_wfn_file_path, **kwargs)
        assert np.allclose(potentials.magnitude, expected_potentials.magnitude)
        if batch_size is None:
            assert isinstance(wfn, psi4.core.Wavefunction)
        else:
            assert len(wfn) == batch_size
            assert isinstance(wfn[0], psi4.core.Wavefunction)


@pytest.mark.skipif(not PSI4_INSTALLED, reason='requires a Python installation of Psi4')
@pytest.mark.parametrize('batch_size,name', [
    (None, 'scf'),
    (1, 'scf'),
    (2, 'scf'),
    (None, 'mp2'),
])
def test_run_psi4_force(batch_size, name):
    """Test that the calculation of forces with _run_psi4.

    This tests that:
    - Both single and batch configurations are handled correctly and the shape
      of the returned quantity is correct.
    - The same force is computed if the input is a molecule or wavefunction or
      restart files.
    - The energies returned with ``return_energy`` are identical to those returned
      without computing forces.
    - There are no 0 components when fix_com and fix_orientation are set to True
      which should be default behavior of create_water_molecule.
    - The write_orbitals argument leaves the orbitals at the given path.
    - The return_wfn argument works.
    - Test Serial and ProcessPool parallelization strategies.

    """
    molecule, batch_positions = create_water_molecule(batch_size=batch_size)

    # Common kwargs to all _run_psi4 calls.
    kwargs = dict(
        batch_positions=batch_positions,
        name=name,
        return_energy=True,
        return_force=True
    )

    # All energies and forces to compare at the end.
    all_energies = []
    all_forces = []

    with tempfile.TemporaryDirectory() as tmp_dir_path:
        # Set global options.
        psi4_config = dict(
            global_options=dict(basis='cc-pvtz', reference='RHF'),
            psi4_output_file_path='quiet',
            psi4_scratch_dir_path=tmp_dir_path,
        )
        configure_psi4(**psi4_config)

        # The first call should create permanent restart files for the guess wavefunction.
        if batch_size is None:
            cached_wfn_file_path = os.path.join(tmp_dir_path, 'wfn')
        else:
            cached_wfn_file_path = [os.path.join(tmp_dir_path, 'wfn'+str(i)) for i in range(batch_size)]

        energies, forces, wavefunctions = _run_psi4(
            molecule=molecule,
            write_orbitals=cached_wfn_file_path,
            return_wfn=True,
            **kwargs
        )
        all_energies.append(energies)
        all_forces.append(forces)

        # Obtain reference ref functions.
        if name != 'scf':
            if batch_size is None:
                wavefunctions = wavefunctions.reference_wavefunction()
            else:
                wavefunctions = [w.reference_wavefunction() for w in wavefunctions]

        # Check that forces computed from reference wavefunction are identical.
        # Wavefunctions cannot be pickled so we don't test process pool.
        energies, forces = _run_psi4(
            molecule=wavefunctions,
            **kwargs
        )
        all_energies.append(energies)
        all_forces.append(forces)

        # Check that forces computed from the restart files are identical.
        # Try different parallelization strategies.
        for strategy in ['serial', 'pool']:
            with parallelization_strategy(strategy, psi4_config) as ps:
                energies, forces = _run_psi4(
                    restart_file=cached_wfn_file_path,
                    parallelization_strategy=ps, **kwargs
                )
                all_energies.append(energies)
                all_forces.append(forces)

        # Check that the energies returned by the gradient
        # are identical to those returned by the potential.
        kwargs.pop('return_force')
        energies_potential = _run_psi4(molecule=molecule, **kwargs)


    # The shape should be identical to the positions.
    if batch_size is None:
        assert forces.shape == molecule.geometry().to_array().shape
    else:
        assert forces.shape == batch_positions.shape

    # Check equivalence of all calls above.
    for f, e in zip(all_forces, all_energies):
        assert np.allclose(forces.magnitude, f.magnitude)
        assert np.allclose(energies.magnitude, e.magnitude)
    assert np.allclose(energies.magnitude, energies_potential.magnitude)

    # By default, Psi4 orient the molecule so that the gradient along
    # the z-axis is 0 for 3 atoms so here we check that we remove this.
    assert np.all(forces.magnitude != 0)


@pytest.mark.skipif(not PSI4_INSTALLED, reason='requires a Python installation of Psi4')
@pytest.mark.parametrize('name', ['scf', 'mp2'])
def test_potential_energy_psi4_gradcheck(name):
    """Test that potential_energy_psi4 implements the correct gradient."""
    batch_size = 2
    molecule, batch_positions = create_water_molecule(batch_size)

    # Convert to tensor.
    batch_positions = np.reshape(batch_positions.to('angstrom').magnitude,
                                 (batch_size, molecule.natom()*3))
    batch_positions = torch.tensor(batch_positions, requires_grad=True, dtype=torch.double)

    # Use restart files to speedup gradcheck.
    with tempfile.TemporaryDirectory() as tmp_dir_path:
        # Set global options.
        configure_psi4(
            global_options=dict(basis='cc-pvtz', reference='RHF'),
            psi4_output_file_path='quiet',
            psi4_scratch_dir_path=tmp_dir_path,
            active_molecule=molecule
        )

        # Run a first SCF calculation to generate the wavefunction restart files.
        cached_wfn_file_path = [os.path.join(tmp_dir_path, 'wfn'+str(i)) for i in range(batch_size)]
        potential_energy_psi4(
            batch_positions,
            name,
            positions_unit=_UREG.angstrom,
            energy_unit=_UREG.kJ / _UREG.mol,
            write_orbitals=cached_wfn_file_path,
            precompute_gradient=False,
        )

        # Run gradcheck. We keep precompute_gradient = False because gradcheck
        # performs a bunch of forward without backward.
        torch.autograd.gradcheck(
            func=potential_energy_psi4,
            inputs=[
                batch_positions,
                name,
                None,  # molecule
                _UREG.angstrom,
                _UREG.kJ / _UREG.mol,
                None,  # write_orbitals
                cached_wfn_file_path,
                False,  # precompute_gradient
                None,  # parallelization strategy
            ],
            atol=0.5,
        )
