#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test objects and function in the module ``tfep.utils.cli.launcher``.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import os
import subprocess
import tempfile

import pytest

from tfep.utils.cli.tool import CLITool
from tfep.utils.cli.launcher import Launcher, SRunLauncher

from test_tool import check_command


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope='module')
def launcher():
    """A standard ``Launcher`` object."""
    return Launcher()


# =============================================================================
# TEST UTILITIES
# =============================================================================

class Echo(CLITool):
    EXECUTABLE_PATH = 'echo'


def _check_echo_stdout(launcher, *commands):
    results = launcher.run(*commands, capture_output=True, text=True)

    if len(commands) == 1:
        results = [results]

    for command, result in zip(commands, results):
        try:  # CliTool.
            echoed_string = command.args[0]
        except:  # subprocess list.
            echoed_string = command[1]
        assert result.stdout == echoed_string + '\n'


# =============================================================================
# TESTS
# =============================================================================

def test_launcher_run_command(launcher):
    """Test the correct execution of ``Launcher.run``.

    The test checks that:
    - Both commands are launched and returned.
    - Both CLITool and subprocess-style commands are executed.

    """
    # Test subprocess format.
    subprocess_cmd = ['echo', 'this is subprocess']
    _check_echo_stdout(launcher, subprocess_cmd)

    # Test CLITool compatibility.
    clitool_cmd = Echo('this is CLITool')
    _check_echo_stdout(launcher, clitool_cmd)

    # Test parallel execution
    _check_echo_stdout(launcher, subprocess_cmd, clitool_cmd)


def test_launcher_run_stdout(launcher):
    """``Launcher.run`` handles stdout correctly.

    Test that it is possible to redirect the stdout independently for multiple
    processes.

    """
    echoed_string = 'print this'
    commands = [['echo', echoed_string], Echo(echoed_string)]

    # First test running a single command.
    for command in commands:
        with tempfile.NamedTemporaryFile('w+') as f:
            launcher.run(command, stdout=f)

            # Check content.
            f.seek(0)
            assert f.read() == echoed_string + '\n'

    # Now test with parallel executions.
    with tempfile.NamedTemporaryFile('w+') as f1, tempfile.NamedTemporaryFile('w+') as f2:
        launcher.run(*commands, stdout=[f1, f2])
        for f in [f1, f2]:
            f.seek(0)
            assert f.read() == echoed_string + '\n'


def test_launcher_run_working_dir(launcher):
    """``Launcher.run`` runs in the correct working directory.

    The test checks that:
    - Parallel commands are executed in process-specific working directories.
    - After the execution the working directory is restored.

    """
    cwd1 = os.getcwd()
    commands = [['pwd'], ['pwd']]

    # Create a temporary directory where to execute the test.
    with tempfile.TemporaryDirectory() as tmp_dir1, tempfile.TemporaryDirectory() as tmp_dir2:
        # Check that the execution runs in a different working directories.
        results = launcher.run(*commands, cwd=[tmp_dir1, tmp_dir2], capture_output=True, text=True)
        for result, tmp_dir in zip(results, [tmp_dir1, tmp_dir2]):
            assert os.path.realpath(result.stdout) == os.path.realpath(tmp_dir) + '\n'

    # Make sure the working dir has been restored.
    cwd2 = os.getcwd()
    assert cwd1 == cwd2


def test_launcher_run_check_error(launcher):
    """``Launcher.run`` raises an exception when the command returns a non-zero exit code."""
    with pytest.raises(subprocess.CalledProcessError, match='grep'):
        launcher.run(['pwd'], ['grep'], check=True)


def test_launcher_run_timeout_error(launcher):
    """``Launcher.run`` raises an exception when a command runs too long."""
    with pytest.raises(subprocess.TimeoutExpired, match='sleep'):
        launcher.run(['pwd'], ['sleep', '2'], timeout=0.05)


@pytest.mark.parametrize('srun_kwargs,expected_kwargs', [
    ({'n_nodes': 1, 'n_tasks': 3}, {'--nodes': '1', '--ntasks': '3'}),
    ({'n_nodes': 1, 'n_tasks': 3, 'n_tasks_per_node': 2, 'n_cpus_per_task': 5},
        {'--nodes': '1', '--ntasks': '3', '--ntasks-per-node': '2', '--cpus-per-task': '5'}),
    ({'n_nodes': [1, 2], 'n_tasks_per_node': 3},
        [{'--nodes': '1', '--ntasks-per-node': '3'}, {'--nodes': '2', '--ntasks-per-node': '3'}]),
    ({'n_nodes': 1, 'n_tasks': [3, 2], 'multiprog': True},
        {'--nodes': '1', '--ntasks': '5', '--multi-prog': 'srun-job.conf'}),
    ({'n_nodes': 1, 'n_tasks': 2, 'n_tasks_per_node': 3, 'multiprog': True},
        {'--nodes': '1', '--ntasks': '4', '--multi-prog': 'srun-job.conf'}),
])
def test_srun_launcher_create_command(srun_kwargs, expected_kwargs):
    """Test the ``SRunLauncher._create_srun_commands`` method.

    The test check that the method prepends the correct srun command.

    """
    commands = [['pwd'], Echo('ciao')]

    # If no option is a list, we can also test single commands.
    test_single_cmd = all(not isinstance(val, list) for val in srun_kwargs.values())

    # Check if the parallel test must be done with multiprog.
    multiprog = srun_kwargs.get('multiprog', False)
    test_single_cmd = test_single_cmd and not multiprog

    # Create launcher.
    srun_launcher = SRunLauncher(**srun_kwargs)

    # Without multiprog, we also test single commands.
    if test_single_cmd:
        for cmd in commands:
            list_cmd = cmd if not isinstance(cmd, CLITool) else cmd.to_subprocess()
            srun_cmd = srun_launcher._create_srun_commands([cmd])[0]
            check_command(srun_cmd, executable='srun', args=list_cmd,
                          kwargs=expected_kwargs)

    # Create the commands.
    srun_cmds = srun_launcher._create_srun_commands(commands)

    # Test the creation of multiple commands.
    if multiprog:
        assert len(srun_cmds) == 1

        expected_kwargs.update({'--multi-prog': 'srun-job.conf'})
        check_command(srun_cmds[0], executable='srun', kwargs=expected_kwargs)
    else:
        assert len(srun_cmds) == len(commands)

        # If the two srun commands must have different options, expected_kwargs
        # is already a list. Otherwise, we duplicate the expected_kwargs.
        if test_single_cmd:
            expected_kwargs = [expected_kwargs, expected_kwargs]

        # The two commands are run with two srun executions.
        for cmd_idx, (cmd, srun_cmd) in enumerate(zip(commands, srun_cmds)):
            list_cmd = cmd if not isinstance(cmd, CLITool) else cmd.to_subprocess()
            check_command(srun_cmd, executable='srun', args=list_cmd,
                          kwargs=expected_kwargs[cmd_idx])


@pytest.mark.parametrize('srun_kwargs,multi_command,match_str', [
    ({'n_nodes': [1, 2], 'n_tasks_per_node': 3}, False, '1 commands but only 2 n_nodes'),
    ({'n_nodes': [1, 2], 'n_tasks': 3, 'multiprog': True}, True, 'must be integers'),
])
def test_srun_launcher_incompatible_config(srun_kwargs, multi_command, match_str):
    """``SRunLauncher`` raises an error if its configuration is incompatible with the commands."""
    if multi_command:
        commands = [['pwd'], ['pwd']]
    else:
        commands = [['pwd']]

    srun_launcher = SRunLauncher(**srun_kwargs)
    with pytest.raises(ValueError, match=match_str):
        srun_launcher.run(*commands)


@pytest.mark.parametrize('n_tasks,expected_ranks', [
    ([2, 3], ['0-1', '2-4']),
    (3, ['0-2', '3-5'])
])
def test_srun_launcher_create_multiprog_config(n_tasks, expected_ranks):
    """Test the ``SRunLauncher`` multiprog feature.

    The test checks that ``SRunLauncher`` creates the correct config file for
    the srun --multi-prog command.

    """
    commands = [['pwd'], ['echo', 'ciao']]

    srun_launcher = SRunLauncher(n_tasks=n_tasks, multiprog=True)

    with tempfile.NamedTemporaryFile('w+') as f:
        # Create the file.
        srun_launcher.multiprog_config_file_path = f.name
        srun_launcher._create_multiprog_config_file(commands)

        # Read it back and parse it.
        f.seek(0)
        lines = f.readlines()

    for cmd_idx, cmd in enumerate(commands):
        line = lines[cmd_idx].split()
        list_cmd = cmd if not isinstance(cmd, CLITool) else cmd.to_subprocess()
        assert line[0] == expected_ranks[cmd_idx]
        assert line[1:] == list_cmd
