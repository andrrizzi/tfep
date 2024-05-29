#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test objects and function in the module ``tfep.potentials.mimic``.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import contextlib
import os
import shutil
import tempfile

import MDAnalysis
import numpy as np
import pint
import pytest
import torch

from tfep.potentials.gromacs import GmxGrompp, GmxMdrun, GROMACSPotential
from tfep.utils.cli import Launcher, SRunLauncher
from tfep.utils.misc import flattened_to_atom
from tfep.utils.parallel import ProcessPoolStrategy

from .. import DATA_DIR_PATH


# =============================================================================
# GLOBAL VARIABLES
# =============================================================================

GROMPP_GMX = 'gmx'
MDRUN_GMX = 'gmx_mpi'

MIMIC_INPUT_DIR_PATH = os.path.realpath(os.path.join(DATA_DIR_PATH, 'mimic'))
TPR_FILE_PATH = os.path.join(MIMIC_INPUT_DIR_PATH, 'gromacs-only.tpr')

_UREG = pint.UnitRegistry()


# =============================================================================
# TEST MODULE CONFIGURATION
# =============================================================================

_old_default_dtype = None

def setup_module(module):
    # Executable.
    set_executables()

    # Double precision.
    global _old_default_dtype
    _old_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.double)

    # Create a temporary tpr file to run all GROMACS tests.
    grompp = GmxGrompp(
        mdp_input_file_path=os.path.join(MIMIC_INPUT_DIR_PATH, 'gromacs-only.mdp'),
        structure_input_file_path=os.path.join(MIMIC_INPUT_DIR_PATH, 'equilibrated.gro'),
        top_input_file_path=os.path.join(MIMIC_INPUT_DIR_PATH, 'acetone.top'),
        tpr_output_file_path=TPR_FILE_PATH,
    )
    # Run grompp in a temporary directory so that all log files gets deleted at the end.
    with tempfile.TemporaryDirectory() as tmp_dir_path:
        Launcher().run(grompp, cwd=os.path.realpath(tmp_dir_path))


def teardown_module(module):
    torch.set_default_dtype(_old_default_dtype)
    os.remove(TPR_FILE_PATH)


@pytest.fixture(scope='module')
def configurations():
    """Read the configurations from the equilibrated.gro and mimic.pdb files."""
    batch_size, n_atoms = 2, 1528

    batch_positions = np.empty((batch_size, n_atoms*3))
    batch_cell = np.empty((2, 6))
    with MDAnalysis.coordinates.GRO.GROReader(os.path.join(MIMIC_INPUT_DIR_PATH, 'equilibrated.gro')) as reader:
        batch_positions[0] = reader.ts.positions.flatten()
        batch_cell[0] = reader.ts.dimensions
    with MDAnalysis.coordinates.PDB.PDBReader(os.path.join(MIMIC_INPUT_DIR_PATH, 'mimic.pdb')) as reader:
        batch_positions[1] = reader.ts.positions.flatten()
        batch_cell[1] = reader.ts.dimensions

    # Expected energies are hardcoded here.
    expected_energies = np.array([-21275.45703125, -21355.24609375]) * _UREG.kJ / _UREG.mole

    # Expected forces are saved in files.
    expected_forces = np.empty((batch_size, n_atoms, 3))
    for i, file_path in enumerate([
        os.path.join(MIMIC_INPUT_DIR_PATH, 'equilibrated-forces-gromacs-only.trr'),
        os.path.join(MIMIC_INPUT_DIR_PATH, 'mimic-forces-gromacs-only.trr'),
    ]):
        with MDAnalysis.coordinates.TRR.TRRReader(file_path, convert_units=False) as reader:
            expected_forces[i] = reader.ts.forces
    expected_forces *= _UREG.kJ / _UREG.mole / _UREG.nanometer

    return torch.from_numpy(batch_positions), torch.from_numpy(batch_cell), expected_energies, expected_forces


# =============================================================================
# UTILS
# =============================================================================

def set_executables():
    GmxGrompp.EXECUTABLE_PATH = GROMPP_GMX
    GmxMdrun.EXECUTABLE_PATH = MDRUN_GMX


@contextlib.contextmanager
def get_parallelization_strategy(parallel, n_processes):
    """Used to avoid code branching to handle Pool and Serial parallelization strategies."""
    if parallel:
        mp_context = torch.multiprocessing.get_context('forkserver')
        with mp_context.Pool(n_processes, initializer=set_executables) as pool:
            yield ProcessPoolStrategy(pool)
    else:
        yield None


@contextlib.contextmanager
def get_working_dir_path(set_working_dir_path, batch_size):
    """Used to avoid code branching to handle a default or a user-defined working directory."""
    if set_working_dir_path:
        with tempfile.TemporaryDirectory() as tmp_dir_path:
            tmp_dir_path = os.path.realpath(tmp_dir_path)
            working_dir_paths = [os.path.join(tmp_dir_path, f'conf{i}') for i in range(batch_size)]
            for dir_path in working_dir_paths:
                os.makedirs(dir_path, exist_ok=True)
            yield working_dir_paths
    else:
        yield None


# =============================================================================
# TESTS
# =============================================================================

@pytest.mark.skipif(shutil.which(GmxMdrun.EXECUTABLE_PATH) is None, reason='requires GROMACS to be installed')
@pytest.mark.parametrize('batch_size', [1, 2])
@pytest.mark.parametrize('set_working_dir_path', [False, True])
@pytest.mark.parametrize('parallel', [False, True])
@pytest.mark.parametrize('launcher_cls', [None, SRunLauncher])
def test_gromacs_energies_and_forces(configurations, batch_size, set_working_dir_path, parallel, launcher_cls):
    """GROMACSPotential returns the expected energies and forces."""
    batch_positions, batch_cell, expected_energies, expected_forces = configurations
    batch_positions = batch_positions[:batch_size].clone().detach().requires_grad_(True)
    batch_cell = batch_cell[:batch_size].clone().detach()

    # Convert units.
    energy_unit = _UREG.kcal / _UREG.mole
    positions_unit = _UREG.angstrom
    expected_energies = expected_energies[:batch_size].to(energy_unit).magnitude
    expected_forces = expected_forces[:batch_size].to(energy_unit/positions_unit).magnitude

    # Configure launcher.
    if launcher_cls is None:
        launcher = None
    elif launcher_cls is SRunLauncher:
        if os.getenv('SLURM_CPUS_PER_TASK') is None:
            pytest.skip('Requires the tests to be launched in a slurm environment.')
        launcher = launcher_cls(
            n_nodes=1,
            n_tasks=int(os.getenv('SLURM_NTASKS_PER_NODE'))//batch_size,
            n_cpus_per_task=int(os.getenv('SLURM_CPUS_PER_TASK'))
        )
    else:
        launcher = launcher_cls()

    # Get the potential energy.
    with get_parallelization_strategy(parallel, n_processes=batch_size) as parallelization_strategy:
        with get_working_dir_path(set_working_dir_path, batch_size) as working_dir_path:
            potential = GROMACSPotential(
                tpr_file_path=TPR_FILE_PATH,
                launcher=launcher,
                positions_unit=positions_unit,
                energy_unit=energy_unit,
                precompute_gradient=True,
                working_dir_path=working_dir_path,
                parallelization_strategy=parallelization_strategy,
            )
            # Run tested function.
            energies = potential(batch_positions, batch_cell)

            # Compute forces.
            energies.sum().backward()
            forces = -flattened_to_atom(batch_positions.grad)

    # Check against the expected energies.
    assert np.allclose(energies.detach().numpy(), expected_energies)
    assert np.allclose(forces.detach().numpy(), expected_forces, atol=1e-3)


@pytest.mark.skipif(shutil.which(GmxMdrun.EXECUTABLE_PATH) is None, reason='requires GROMACS to be installed')
def test_error_precompute_gradient(configurations):
    """An error is raised if backpropagation is attempted with precompute_gradient=False."""
    potential = GROMACSPotential(
        tpr_file_path=TPR_FILE_PATH,
        positions_unit=_UREG.angstrom,
        precompute_gradient=False,
    )
    batch_positions, batch_cell, _, _ = configurations
    batch_positions.requires_grad = True
    energies = potential(batch_positions[:1], batch_cell[:1])
    with pytest.raises(ValueError, match='precompute_gradient'):
        energies.sum().backward()
