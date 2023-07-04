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

import os
import shutil
import tempfile

import MDAnalysis
import numpy as np
import pint
import pytest
import torch

from tfep.potentials.mimic import (Cpmd, GmxGrompp, GmxMdrun, potential_energy_mimic,
                                   _run_mimic, _prepare_cpmd_command, _prepare_mdrun_command)
from tfep.utils.cli import SRunLauncher
from tfep.utils.parallel import ProcessPoolStrategy
from tfep.utils.misc import temporary_cd


# =============================================================================
# GLOBAL VARIABLES
# =============================================================================

SCRIPT_DIR_PATH = os.path.dirname(__file__)
MIMIC_INPUT_DIR_PATH = os.path.realpath(os.path.join(SCRIPT_DIR_PATH, '..', 'data', 'mimic'))

_UREG = pint.UnitRegistry()

# Executables.
CPMD_EXECUTABLE = 'cpmd.x'
GMX_EXECUTABLE = 'gmx_mpi_d'

# We check only the forces only for a few atoms.
EXPECTED_N_ATOMS = 1528
EXPECTED_ENERGIES = np.array([-44.4971622405, -44.5383632532]) * _UREG.hartree
EXPECTED_FORCES = np.array([
    # Force associated to equilibrated.gro configuration.
    [[0.00188575119206, -0.01581822206690, -0.00066802830670],
     [0.01346810738814, -0.00882991538296, -0.01872516125423],
     [-0.00849423507670, -0.00422106523353, -0.00078139730338],
     [0.00480826564192, 0.03029561725466, -0.00276897300748],
     [-0.01083117728982, 0.01789210732776, 0.00433791232407],
     [-0.00866605198428, -0.01176377825443, 0.01263633683792],
     [-0.00339017283143, -0.00245648936633, 0.00032359276322],
     [0.00613424545661, 0.00196892403541, 0.00048428221610],
     [0.00780775997503, 0.00207059990897, -0.00072361095022],
     [-0.00783096584361, -0.00295623987521, 0.00968047481745]],

    # Force associated to em.pdb configuration.
    [[0.00199580230092, -0.00016264967100, -0.00384635054664],
     [0.02837011114346, -0.02787071044456, 0.00159443102251],
     [-0.00612024747724, 0.03066245453101, 0.02771361040134],
     [-0.02157298024699, 0.00093731396821, 0.00994237691917],
     [-0.00114012439776, 0.00025583533913, 0.00155145419821],
     [0.00173034920115, 0.00279475172988, -0.01070319336887],
     [-0.00570859949922, 0.00259206993431, -0.00230802177114],
     [-0.00445232115762, 0.00461372291579, -0.01505945889682],
     [-0.00088666968531, 0.00127563889547, 0.00728713790857],
     [0.00569476133723, -0.02219780937585, -0.01165483165429]],
]) * _UREG.hartree / _UREG.bohr


# =============================================================================
# TEST UTILITIES
# =============================================================================

class DummySRunLauncher:
    """A Fake SRunLauncher to test the launcher outside the SLURM environment."""

    def __init__(self, write=True):
        self.multiprog = True
        self.write = write

    def run(self, *commands, **kwargs):
        """Create fake output ENERGIES and FTRAJECTORY files."""
        if not self.write:
            return

        # We check the name of the working directory to figure out which force/energy to write.
        cwd = kwargs.get('cwd', None)
        if cwd is None or cwd[-1] == '0':
            config_idx = 0
        else:
            config_idx = 1

        # Write the energy and force of this configuration.
        with temporary_cd(kwargs.get('cwd', None)):
            with open('ENERGIES', 'w') as f:
                f.write('  1  0.0  5.0  ' + str(EXPECTED_ENERGIES[config_idx].magnitude) + '\n')

            # Re-order the forces to correspond to the GROMACS order, not CPMD order.
            force = EXPECTED_FORCES[config_idx].magnitude
            n_atoms = force.shape[0]
            force = np.concatenate([force[3:], force[:3]])
            force_str = [[str(x) for x in atom_pos] for atom_pos in force]
            with open('FTRAJECTORY', 'w') as f:
                for atom_idx in range(EXPECTED_N_ATOMS):
                    line = '1  0.0  0.0  0.0  0.0  0.0  0.0  ' + '  '.join(force_str[atom_idx%n_atoms])
                    f.write(line + '\n')


# =============================================================================
# FIXTURES
# =============================================================================

def mimic_srun_commands(parallel_strategy=False):
    """Return the SRunLauncher and CPMD and GROMACS commands used to run the tests.

    These commands are set to run on the compiled version of the software on
    the JURECA DC cluster. You will probably have to set the path to the executables
    to run it somewhere else.

    If parallel_strategy is True, the SRunLauncher is configured to run only on
    half the number of CPUs available to the job so that two srun commands can
    run in parallel.

    """
    batch_size = 2

    # Check if we are testing within a SLURM environment or not.
    use_dummy_launcher = os.getenv('SLURM_JOB_NUM_NODES') is None

    # Load batch positions and box vectors.
    batch_positions = np.empty((batch_size, EXPECTED_N_ATOMS, 3), dtype=float)
    batch_cell = np.empty((batch_size, 3), dtype=float)
    for i, file_name in enumerate(['equilibrated.gro', 'mimic.pdb']):
        universe = MDAnalysis.Universe(os.path.join(MIMIC_INPUT_DIR_PATH, file_name))
        batch_positions[i] = universe.atoms.positions
        batch_cell[i] = universe.dimensions[:3]
    batch_positions = batch_positions * _UREG.angstrom
    batch_cell = batch_cell * _UREG.angstrom

    # Read SLURM job configuration.
    if use_dummy_launcher:
        n_cpus_per_task = 1
        srun_launcher = DummySRunLauncher()
    else:
        n_cpus_per_task = int(os.getenv('SLURM_CPUS_PER_TASK'))
        n_tasks = int(os.getenv('SLURM_NTASKS'))
        n_nodes = int(os.getenv('SLURM_JOB_NUM_NODES'))
        assert n_nodes > 1

        # GROMACS always runs on one node and CPMD on all the others.
        n_tasks_per_node = n_tasks // n_nodes
        if parallel_strategy:
            n_nodes = n_nodes // batch_size
            n_tasks_per_node = n_tasks_per_node // batch_size
        n_tasks_cpmd = n_tasks_per_node * (n_nodes-1)
        srun_launcher = SRunLauncher(
            # n_nodes=n_nodes, n_tasks=[n_tasks_cpmd, n_tasks_per_node], multiprog=True
            n_nodes=[n_nodes-1, 1], n_tasks=[n_tasks_cpmd, n_tasks_per_node], multiprog=False
        )

    if isinstance(srun_launcher, DummySRunLauncher):
        grompp_launcher = DummySRunLauncher(write=False)
    else:
        grompp_launcher = None

    # Initialize commands pointing to the correct scripts.
    cpmd = Cpmd(
        os.path.join(MIMIC_INPUT_DIR_PATH, 'cpmd.inp'),
        '/p/project/cias-5/ippoliti/PROGRAMS/ARCHIVE/CPMD/PP/',
        executable_path=CPMD_EXECUTABLE,
    )
    mdrun = GmxMdrun(
        executable_path=GMX_EXECUTABLE,
        tpr_input_file_path=os.path.join(MIMIC_INPUT_DIR_PATH, 'gromacs.tpr'),
        default_file_name='gromacs',
        n_omp_threads_per_mpi_rank=n_cpus_per_task,
    )
    grompp = GmxGrompp(
        executable_path=GMX_EXECUTABLE,
        mdp_input_file_path=os.path.join(MIMIC_INPUT_DIR_PATH, 'gromacs.mdp'),
        structure_input_file_path=os.path.join(MIMIC_INPUT_DIR_PATH, 'equilibrated.gro'),
        top_input_file_path=os.path.join(MIMIC_INPUT_DIR_PATH, 'acetone.top'),
        index_input_file_path=os.path.join(MIMIC_INPUT_DIR_PATH, 'index.ndx'),
        n_max_warnings=1,
    )

    return cpmd, mdrun, grompp, srun_launcher, grompp_launcher, batch_positions, batch_cell


# =============================================================================
# TESTS
# =============================================================================

@pytest.mark.parametrize('update_positions', [False, True])
def test_prepare_cpmd_command(update_positions):
    """Test the function _prepare_cpmd_command().

    This tests that:
    - If no change is required, the input script and command object are not
      copied.
    - If needed, the &MIMIC.PATHS cpmd option is updated to point to the given
      working directory.
    - The positions in the script are updated correctly.

    """
    cpmd_cmd, _, _, _, _, _, _ = mimic_srun_commands()

    # Check if we need new positions.
    n_atoms = 10
    if update_positions:
        # The input/output positions will have the GROMACS/CPMD atom order respectively.
        new_positions = np.array([[float(i)]*3 for i in range(n_atoms)]) * _UREG.angstrom
        expected_positions = new_positions[[3, 4, 5, 6, 7, 8, 9, 0, 1, 2]]
        new_box_vectors = np.array([3.0, 3.0, 3.0]) * _UREG.nanometer
    else:
        new_positions = None
        new_box_vectors = None

    with tempfile.TemporaryDirectory() as tmp_dir_path:
        tmp_dir_path = os.path.realpath(tmp_dir_path)

        # Run the tested function
        new_cpmd_cmd, _ = _prepare_cpmd_command(cpmd_cmd, tmp_dir_path, new_positions, new_box_vectors)

        # Test that the commands were updated.
        assert new_cpmd_cmd != cpmd_cmd
        assert new_cpmd_cmd.args[0] == 'cpmd.inp'

        # Read the original and modified files.
        with open(cpmd_cmd.args[0], 'r') as f:
            input_lines = f.readlines()
        with open(os.path.join(tmp_dir_path, 'cpmd.inp'), 'r') as f:
            copied_lines = f.readlines()
        assert len(input_lines) == len(copied_lines)

        # Check the positions if needed.
        if update_positions:
            # The line numbers should be the same as the cpmd file.
            line_indices = [84, 83, 82, 77, 76, 75, 74, 73, 72, 67]
            input_pos_lines = list(reversed([input_lines.pop(i) for i in line_indices]))
            copied_pos_lines = list(reversed([copied_lines.pop(i) for i in line_indices]))
            for l1, l2 in zip(input_pos_lines, copied_pos_lines):
                assert l1 != l2

            # Check that the written positions are actually what we expect
            written_positions = [[float(x) for x in l.split()] for l in copied_pos_lines]
            written_positions = np.array(written_positions)
            assert np.allclose(expected_positions.to('bohr').magnitude, written_positions)

            # Check also the box vector.
            input_line = input_lines.pop(5)
            copied_line = copied_lines.pop(5)
            written_box_vectors = np.array([float(x) for x in copied_line.split()])
            assert input_line != copied_line
            assert np.allclose(new_box_vectors.to('bohr').magnitude, written_box_vectors)

        # Make sure the content of the cpmd input file is identical but for &MIMIC.PATHS.
        # Check that the &MIMIC.PATHS option was updated.
        input_line = input_lines.pop(3)
        copied_line = copied_lines.pop(3)
        assert copied_line == os.path.realpath(tmp_dir_path) + '\n'
        assert input_line != copied_line

        # All other lines should be identical.
        for l1, l2 in zip(input_lines, copied_lines):
            assert l1 == l2

        # Now, if we re-prepare the new command without asking for updated
        # positions, it should not change it.
        new_new_cpmd_cmd, _ = _prepare_cpmd_command(new_cpmd_cmd, tmp_dir_path)
        assert new_new_cpmd_cmd == new_cpmd_cmd


@pytest.mark.skipif(shutil.which(GMX_EXECUTABLE) is None, reason='requires GROMACS to be installed')
@pytest.mark.parametrize('template_structure_file_name', ['equilibrated.gro', 'mimic.pdb'])
def test_prepare_mdrun_command(template_structure_file_name):
    """Test the function _prepare_mdrun_command().

    This tests that a new .tpr is generated with the correct updated positions.
    It tries both using as template structure file files in pdb and gro format.

    """
    import subprocess

    _, mdrun_cmd, grompp_cmd, _, _, batch_positions, batch_cell = mimic_srun_commands()

    with tempfile.TemporaryDirectory() as tmp_dir_path:
        # Fix template.
        template_structure_file_path = os.path.join(MIMIC_INPUT_DIR_PATH, template_structure_file_name)
        grompp_cmd.structure_input_file_path = template_structure_file_path

        # Run the tested function
        new_mdrun_cmd = _prepare_mdrun_command(
            mdrun_cmd, grompp_cmd, working_dir_path=tmp_dir_path,
            positions=batch_positions[0], box_vectors=batch_cell[0],
        )

        # Check that the new mdrun was copied to not modify the original cmd.
        assert new_mdrun_cmd != mdrun_cmd

        # The mdrun cmd points to the correct tpr.
        with temporary_cd(tmp_dir_path):
            new_trp_file_path = os.path.realpath(new_mdrun_cmd.tpr_input_file_path)
        tmp_trp_file_path = os.path.realpath(os.path.join(tmp_dir_path, 'gromacs.tpr'))
        assert new_trp_file_path == tmp_trp_file_path

        # Now convert the TPR file to something readable and check that
        # the positions are correct. The only way to do this seems to
        # use gmx dump (which prints everything) and then parse the stdout
        # for the coordinates.
        result = subprocess.run([GMX_EXECUTABLE, 'dump', '-s', tmp_trp_file_path],
                                capture_output=True, text=True)

        # Initialize values to read.
        out_box_vectors = np.empty((3, 3), dtype=float)
        out_positions = np.empty((EXPECTED_N_ATOMS, 3), dtype=float)
        out_arrays = [out_box_vectors, out_positions]

        # Read the box vector and positions.
        start_idx = 0
        for i, (start_str, n_lines) in enumerate([
            ('box (3x3):', 3),
            ('x (' + str(EXPECTED_N_ATOMS) + 'x3):', EXPECTED_N_ATOMS)
        ]):
            start_idx = result.stdout.find(start_str, start_idx) + len(start_str) + 2
            for line_idx, line in enumerate(result.stdout[start_idx:].splitlines()):
                if line_idx >= n_lines:
                    break
                # The last character of each float is a comma or a bracket. E.g.,
                #     box[    0]={ 2.48732e+00,  0.00000e+00,  0.00000e+00}
                line = line.split(sep='{')[1].split()
                out_arrays[i][line_idx] = [float(line[col_idx][:-1]) for col_idx in range(3)]

        # Only the diagonal elements of box vectors should be non zero.
        assert np.count_nonzero(out_box_vectors - np.diag(np.diagonal(out_box_vectors))) == 0

        # The positions and box vectors should be identical.
        assert np.allclose(out_positions, batch_positions[0].to('nm').magnitude)
        assert np.allclose(np.diagonal(out_box_vectors), batch_cell[0].to('nm').magnitude)


@pytest.mark.skipif(shutil.which(CPMD_EXECUTABLE) is None, reason='requires MiMiC to be installed')
@pytest.mark.parametrize('config', [
    (None, False),
    (1, False),
    (2, False),
    (2, True)
])
def test_run_mimic(config):
    """Test the function _run_mimic().

    This tests that the function return the expected energy/forces.

    """
    batch_size, parallel = config
    (cpmd_cmd, mdrun_cmd, grompp_cmd, launcher, grompp_launcher,
     batch_positions, batch_cell) = mimic_srun_commands(parallel)

    with tempfile.TemporaryDirectory() as tmp_dir_path:

        # Configure batch positions, box vectors, and working directories.
        if batch_size is None:
            batch_positions, batch_cell = None, None
            working_dir_path = os.path.join(tmp_dir_path, 'conf0')
        elif batch_size == 1:
            batch_positions = batch_positions[0]
            batch_cell = batch_cell[0]
            working_dir_path = os.path.join(tmp_dir_path, 'conf0')
        else:
            working_dir_path = [os.path.join(tmp_dir_path, 'conf'+str(i)) for i in range(batch_size)]

        # Create working directories.
        if isinstance(working_dir_path, list):
            for dir_path in working_dir_path:
                os.makedirs(dir_path, exist_ok=True)
        else:
            os.makedirs(working_dir_path, exist_ok=True)

        pool = None
        try:
            # Check wheter we need to run using a parallel pool.
            if parallel:
                pool = torch.multiprocessing.get_context('forkserver').Pool(batch_size)
                parallelization_strategy = ProcessPoolStrategy(pool)
            else:
                parallelization_strategy = None

            # Run tested function.
            energies, forces = _run_mimic(
                cpmd_cmd, mdrun_cmd, grompp_cmd,
                batch_positions=batch_positions, batch_cell=batch_cell,
                launcher=launcher, grompp_launcher=grompp_launcher,
                return_energy=True, return_force=True,
                working_dir_path=working_dir_path, cleanup_working_dir=True,
                parallelization_strategy=parallelization_strategy,
                on_unconverged='success',
            )
        finally:
            if pool is not None:
                pool.close()

        # Check that the energy and forces (for a few atoms) are correct.
        if batch_size is None or batch_size == 1:
            energies = [energies]
            forces = [forces]

        for conf_idx, (energy, force) in enumerate(zip(energies, forces)):
            expected_energy = EXPECTED_ENERGIES[conf_idx]
            expected_force = EXPECTED_FORCES[conf_idx]
            assert np.isclose(energy, expected_energy)
            assert np.allclose(force[:len(expected_force)], expected_force, atol=1e-5)

            # Check that the number of atoms is correct.
            assert force.shape == (EXPECTED_N_ATOMS, 3)


def _potential_energy_mimic_wrapper(n_mapped_atoms, batch_positions):
    """A wrapper that fixes a number of atoms positions and create the correct LATEST file for restart.

    This can be used to compute the gradient of only a subset of the degrees
    of freedom which greatly speeds up the calculation of the numerical gradient
    in autograd.gradcheck().

    """
    def _potential_energy_mimic_wrapped(*args):
        args = list(args)

        # Add the fixed positions.
        args[0] = torch.cat((args[0], batch_positions[:, n_mapped_atoms*3:]), dim=1)

        # Create LATEST file pointing to the reference RESTART function.
        working_dir_path = args[10]
        for dir_path in working_dir_path:
            if 'conf0' in dir_path:
                conf_idx = '0'
            else:
                conf_idx = '1'
            with open(os.path.join(dir_path, 'LATEST'), 'w') as f:
                f.write('../restart' + conf_idx + '/RESTART.1\n           1')

        return potential_energy_mimic(*args)
    return _potential_energy_mimic_wrapped


@pytest.mark.skipif(os.getenv('SLURM_JOB_NUM_NODES') is None, reason='requires SLURM execution')
def test_potential_energy_mimic_gradcheck():
    """Test that potential_energy_mimic implements the correct gradient."""
    n_mapped_atoms = 2

    (cpmd_cmd, mdrun_cmd, grompp_cmd, launcher, grompp_launcher,
     batch_positions, batch_cell) = mimic_srun_commands(parallel_strategy=True)
    batch_size, n_atoms, _ = batch_positions.shape
    positions_unit = _UREG.nanometer
    energy_unit = _UREG.kJ / _UREG.mol

    # Run using a parallel parallelization strategy.
    with torch.multiprocessing.Pool(batch_size) as p:
        parallelization_strategy = ProcessPoolStrategy(p)

        with tempfile.TemporaryDirectory() as tmp_dir_path:

            # Run a first calculation to create a restart file to speedup gradcheck.
            restart_dir_paths = [os.path.join(tmp_dir_path, 'restart'+str(i)) for i in range(batch_size)]
            for dir_path in restart_dir_paths:
                os.makedirs(dir_path, exist_ok=True)

            _run_mimic(
                cpmd_cmd, mdrun_cmd, grompp_cmd,
                batch_positions=batch_positions, batch_cell=batch_cell,
                launcher=launcher, grompp_launcher=grompp_launcher,
                return_energy=False, return_force=False,
                working_dir_path=restart_dir_paths, cleanup_working_dir=False,
                parallelization_strategy=parallelization_strategy,
                on_unconverged='success',
            )

            # Create a template CPMD input file with the RESTART directive.
            with open(cpmd_cmd.args[0], 'r') as f:
                cpmd_input_lines = f.readlines()

            cpmd_input_file_path = os.path.join(tmp_dir_path, 'cpmd.inp')
            cpmd_input_lines.insert(30, 'RESTART WAVEFUNCTION VELOCITIES LATEST\n')
            with open(cpmd_input_file_path, 'w') as f:
                f.writelines(cpmd_input_lines)

            cpmd_cmd.args = [cpmd_input_file_path] + list(cpmd_cmd.args[1:])

            # Create temporary directories where to save the simulation files.
            working_dir_path = [os.path.join(tmp_dir_path, 'conf'+str(i)) for i in range(batch_size)]
            for dir_path in working_dir_path:
                os.makedirs(dir_path, exist_ok=True)

            # Convert batch_positions to tensor and nanometer.
            batch_positions = np.reshape(batch_positions.to(positions_unit).magnitude,
                                         (batch_size, n_atoms*3))
            batch_positions = torch.tensor(batch_positions, requires_grad=False, dtype=torch.double)
            batch_cell = torch.tensor(batch_cell.to(positions_unit).magnitude,
                                             requires_grad=False, dtype=torch.double)

            # Run gradcheck only on a subset of the DOFs.
            func = _potential_energy_mimic_wrapper(n_mapped_atoms, batch_positions)
            batch_positions = batch_positions[:, :n_mapped_atoms*3].clone().detach().requires_grad_(True)

            # Run gradcheck on a subset of the DOFs.
            torch.autograd.gradcheck(
                func=func,
                inputs=[batch_positions,
                        batch_cell,
                        cpmd_cmd,
                        mdrun_cmd,
                        grompp_cmd,
                        launcher,
                        grompp_launcher,
                        positions_unit,
                        energy_unit,
                        True,  # precompute_gradient
                        working_dir_path,
                        True,  # cleanup_working_dir
                        parallelization_strategy,
                        None,  # launcher_kwargs
                        None,  # grompp_launcher_kwargs
                        1,  # n_attempts
                        'success',  # on_unconverged
                        'raise',  # on_local_error
                        ],
                atol=0.5
            )
