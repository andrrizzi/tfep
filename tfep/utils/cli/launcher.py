#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Utility classes to launch and execute command line executables.

The :class:`~tfep.utils.cli.Launcher` masks the way a command is run to allow
easy executions in different settings such as on through mpirun or Slurm's srun.
These classes work similarly to the standard library function ``subprocess.run``,
but they can handle running multiple commands in parallel.

"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import contextlib
import subprocess

from tfep.utils.cli import CLITool, KeyValueOption
from tfep.utils.misc import temporary_cd


# =============================================================================
# STANDARD EXECUTION LAUNCHER
# =============================================================================

class Launcher:
    """Runs an executable as a standard command line subprocess.

    The class can run a command that is specified either as a list of strings
    (in the same format used by the standard module ``subprocess``) or as a
    :class:`tfep.utils.cli.CLITool`. See the documentation of :class:`~tfep.utils.cli.CLITool`
    for details on how to create a CLI wrapper compatible with ``Launcher``.

    .. note::
        When commands are run in a different working directory, relative
        file/directory paths passed as command arguments are interpreted
        relative to the specified working directory.

        To avoid this behavior, you can either specify absolute paths or use the
        CLI options :class:`tfep.utils.cli.AbsolutePathOption` when wrapping your
        command with :class:`~tfep.utils.cli.CLITool`.

    See Also
    --------
    tfep.utils.cli.CLITool : CLI wrapper utility.

    Examples
    --------

    >>> launcher = Launcher()
    >>> result = launcher.run(['echo', 'print this'], capture_output=True, text=True)
    >>> print(result.stdout.strip())
    print this

    For more complicated cases, it may be convenient to use the CLI wrapping
    utilities in `tfep.utils.cli`

    >>> import tfep.utils.cli
    >>> class Echo(tfep.utils.cli.CLITool):
    ...     EXECUTABLE_PATH = 'echo'
    ...
    >>> echo_cmd = Echo('print this')
    >>> result = launcher.run(echo_cmd, capture_output=True, text=True)
    >>> print(result.stdout.strip())
    print this

    It is possible to launch multiple commands in parallel and obtain the results
    of all of them.

    >>> results = launcher.run(Echo('print1'), Echo('print2'), capture_output=True, text=True)
    >>> for res in results:
    ...     print(res.stdout.strip())
    ...
    print1
    print2

    """
    def run(self, *commands, capture_output=False, timeout=None, check=False,
            stdin=None, stdout=None, stderr=None, cwd=None, **kwargs):
        """Run one or more subprocesses in parallel.

        The method runs all the commands in parallel and waits for all of them
        to complete, collects their output (if ``capture_output`` is set) and
        return them.

        Currently, the method supports all keyword arguments supported by the
        ``subprocess.Popen``. Moreover, it handles running multiple processes in
        parallel, and, for some of the ``subprocess.Popen`` such as ``stdout``
        and ``cwd``, allow to specify process-specific arguments.

        Parameters
        ----------
        commands : List[str] or tfep.utils.cli.CLITool
            One or more commands to execute, either in the same list format used
            by ``subprocess.Popen`` or as a :class:`~tfep.utils.cli.CLITool`.
        capture_output : bool, optional
            If ``True``, stdout and stderr will be captured and returned as an
            attribute of the ``subprocess.CompletedProcess`` objects. If ``True``,
            this overwrites the values of the ``stdout`` and ``stderr`` arguments.
        timeout : float, optional
            The timeout (in seconds) is passed to ``Popen.communicate``. If it
            expires for any of the processes, ``subprocess.TimeoutExpired`` error
            is raised.
        check : bool, optional
            If ``True`` and the exit code of any of the subprocesses was non-zero,
            a ``subprocess.CalledProcessError`` error is raised.
        stdin : optional
            This can take any value accepted by ``subprocess.Popen``. If multiple
            commands are run, this can be a list specifying one stdin per process.
        stdout : optional
            This can take any value accepted by ``subprocess.Popen``. If multiple
            commands are run, this can be a list specifying one stdout per process.
        stderr : optional
            This can take any value accepted by ``subprocess.Popen``. If multiple
            commands are run, this can be a list specifying one stderr per process.
        cwd : str or List[str], optional
            This can take any value accepted by ``subprocess.Popen``. If multiple
            commands are run, this can be a list specifying one current working
            directory per process.
        **kwargs
            Other keyword arguments to pass to ``subprocess.Popen``.

        Returns
        -------
        result : subprocess.CompletedProcess or List[subprocess.CompletedProcess]
            The object encapsulating the results of the project. If multiple
            processes are run in parallel, this is a ``list`` of results, one
            for each process.

        Raises
        ------
        subprocess.CalledProcessError
            If any of the run processes returned a non-zero status and ``check``
            is ``True``.
        subprocess.TimeoutExpired``
            If ``timeout`` was set and the timeout expired.

        See Also
        --------
        subprocess.run : Standard library function to run commands.
        subprocess.Popen : Basic interface to run subprocesses.

        """
        stdin, stdout, stderr, cwd = _ensure_lists(
            len(commands), [stdin, stdout, stderr, cwd])

        # Set the values of stdout/stderr if capture_output has been passed.
        if capture_output:
            for cmd_idx in range(len(stdout)):
                stdout[cmd_idx] = subprocess.PIPE
                stderr[cmd_idx] = subprocess.PIPE

        # Run all processes in parallel.
        with contextlib.ExitStack() as stack:
            processes = []

            # Create all Popen objects. This starts the execution of all subprocesses.
            for cmd_idx, cmd in enumerate(commands):
                # Convert the command to subprocess format.
                if isinstance(cmd, CLITool):
                    cmd = cmd.to_subprocess()

                # Create Popen.
                processes.append(stack.enter_context(subprocess.Popen(
                    cmd, stdin=stdin[cmd_idx], stdout=stdout[cmd_idx],
                    stderr=stderr[cmd_idx], cwd=cwd[cmd_idx], **kwargs
                )))

            # Now wait for the end and collect all outputs.
            results = []
            for process in processes:
                try:
                    stdout, stderr = process.communicate(timeout=timeout)
                except subprocess.TimeoutExpired as exception:
                    self._on_timeout_expired(process, exception, cwd)
                except:
                    process.kill()
                    process.wait()
                    raise
                retcode = process.poll()
                if check and retcode:
                    raise subprocess.CalledProcessError(
                        retcode, process.args, output=stdout, stderr=stderr)
                results.append(subprocess.CompletedProcess(
                    process.args, retcode, stdout, stderr))

        if len(commands) == 1:
            return results[0]
        return results

    def _handle_process(self, process, timeout, cwd):
        """Handle the process and returns stdout, stderr and return code."""
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exception:
            self._on_timeout_expired(process, exception)
        except:
            process.kill()
            process.wait()
            raise
        retcode = process.poll()
        return stdout, stderr, retcode

    def _on_timeout_expired(self, process, exception):
        """Terminate the process and raises a TimeoutExpired error."""
        process.kill()
        stdout, stderr = process.communicate()
        raise subprocess.TimeoutExpired(
            process.args, exception.timeout, output=stdout, stderr=stderr)


# =============================================================================
# LAUNCH WITH SLURM SRUN
# =============================================================================

class SRunTool(CLITool):
    """SLURM srun command line utility."""
    EXECUTABLE_PATH = 'srun'
    n_nodes = KeyValueOption('--nodes')
    n_tasks = KeyValueOption('--ntasks')
    n_tasks_per_node = KeyValueOption('--ntasks-per-node')
    n_cpus_per_task = KeyValueOption('--cpus-per-task')
    relative_node_idx = KeyValueOption('--relative')
    multiprog_config_file_path = KeyValueOption('--multi-prog')

    def to_subprocess(self):
        # For some reason, srun crashes if --multi-prog is not the last keyword
        # arguments so we overwrite CLITool's method to fix the order.
        cmd = super().to_subprocess()

        # Search the --multi-prog option.
        if self.multiprog_config_file_path is not None:
            multi_prog_idx = cmd.index('--multi-prog')
            if multi_prog_idx != len(cmd)-2:
                # Move --multi-prog at the end.
                cmd = cmd[:multi_prog_idx] + cmd[multi_prog_idx+2:] + cmd[multi_prog_idx:multi_prog_idx+2]

        return cmd


class SRunLauncher(Launcher):
    """Launch a command through SLURM's srun.

    The launcher simply prepends ``"srun"`` to each given command, setting the
    specified number of nodes, tasks per node, and cpus per task.

    The launcher also supports running multiple commands in parallel using the
    ``--multi-prog`` feature. The launcher assigns contiguous task ranks to each
    command.

    Parameters
    ----------
    n_nodes : int or List[int], optional
        The number of nodes to pass to ``srun``. If multiple commands are executed
        in parallel, it is possible to specify the number of nodes for each command
        as a list.
    n_tasks : int or List[int], optional
        The number of tasks to pass to ``srun``. If multiple commands are executed
        in parallel, it is possible to specify the number of tasks for each
        command as a list.
    n_tasks_per_node : int or List[int], optional
        The number of tasks per node to pass to ``srun``. If multiple commands
        executed in parallel, it is possible to specify the number of tasks per
        node for each command as a list. Note that ``n_tasks`` takes precedence
        over this.
    n_cpus_per_task : int or List[int], optional
        The number of cpus per task to pass to ``srun``. If multiple commands
        are executed in parallel, it is possible to specify the number of cpus
        per task for each command as a list.
    relative_node_idx : int or List[int], optional
        Run a job step relative the ``relative_node_idx``-th node (starting from
        node 0) of the current allocation. If multiple commands are executed in
        parallel, it is possible to specify one relative node for each command
        as a list.
    multiprog : bool, optional
        If ``True`` multiple commands are run in parallel using the ``--multi-prog``
        argument. In this case, ``srun`` is invoked only once, and thus ``n_nodes``
        ``n_tasks_per_node``, etc. must be integers.
    multiprog_config_file_path : str, optional
        The file path (relative to the working directory) where the multiprog
        configuration file is created.

    Attributes
    ----------
    n_nodes : int or List[int] or None
        The number of nodes to pass to ``srun`` for each command.
    n_tasks : int or List[int] or None
        The number of tasks to pass to ``srun`` for each command.
    n_tasks_per_node : int or List[int] or None
        The number of tasks per node to pass to ``srun`` for each command.
    n_cpus_per_task : int or List[int] or None
        The number of cpus per task to pass to ``srun`` for each command.
    multiprog : bool
        Whether the ``--multi-prog`` feature should be used to run multiple
        commands.
    multiprog_config_file_path : str
        The file path (relative to the working directory) where the multiprog
        configuration file is created.

    See Also
    --------
    Launcher : Standard launcher class.

    Examples
    --------

    If the number of nodes/tasks/cpus are given as an integer, all ``srun``
    parallel executions will have the same number of nodes/tasks/cpus.

    >>> launcher = SRunLauncher(n_nodes=2, n_tasks_per_node=4, n_cpus_per_task=4)

    Multiple commands can be run in parallel either by calling ``srun`` twice
    by calling it once with the ``--multi-prog`` argument, which is design to
    support multiple-program multiple-data (MPMD) MPI programs. In the first
    case, it is possible to specify the configuration for each ``srun``.

    For example, this modifies the launcher to run two commands in parallel using
    the same number of cpus per task but different number of nodes and tasks per
    node.

    >>> launcher.n_nodes = [1, 4]
    >>> launcher.n_tasks_per_node = [8, 4]

    Instead, when ``--multi-prog`` is used, ``srun`` is invoked only once. Thus
    no option can be a list, except for ``n_tasks``, which must be provided as
    a list and is used to determine the task ranks assigned to each program.

    The following example configures the launcher to run three programs on 4
    nodes, and 8 tasks. It assigns 3 tasks to the second process and 2 tasks to
    the others.

    >>> launcher = SRunLauncher(n_nodes=4, n_tasks=[2, 3, 2], multiprog=True)

    """

    def __init__(self, n_nodes=None, n_tasks=None, n_tasks_per_node=None, n_cpus_per_task=None,
                 relative_node_idx=None, multiprog=False, multiprog_config_file_path='srun-job.conf'):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_tasks = n_tasks
        self.n_tasks_per_node = n_tasks_per_node
        self.n_cpus_per_task = n_cpus_per_task
        self.relative_node_idx = relative_node_idx
        self.multiprog = multiprog
        self.multiprog_config_file_path = multiprog_config_file_path

    def run(self, *commands, **kwargs):
        """Run one or more commands with srun.

        The method accepts all keyword arguments supported by
        :func:`tfep.utils.cli.Launcher.run`.

        Parameters
        ----------
        *commands
            One or more commands to execute, either in the same list format used
            by ``subprocess.Popen`` or as a :class:`~tfep.utils.cli.CLITool`.
        **kwargs
            Other keyword arguments to pass to :class:`.Launcher.run`.

        Returns
        -------
        result : subprocess.CompletedProcess or List[subprocess.CompletedProcess]
            The object encapsulating the results of the project. If multiple
            processes are run in parallel, this is a ``list`` of results, one
            for each process.

        See Also
        --------
        tfep.utils.cli.Launcher.run : The parent class method.

        """
        n_commands = len(commands)

        # Check that the configuration is consistent with the given commands.
        # With multiprog, all options must be integers except for n_task, which
        # specify the task ranks for each command, and n_tasks_per_node, which
        # is ignored.
        run_with_multiprog = n_commands > 1 and self.multiprog
        if run_with_multiprog:
            for attr_name in ['n_nodes', 'n_cpus_per_task', 'relative_node_idx']:
                if isinstance(getattr(self, attr_name), list):
                    raise ValueError(f'With multiprog execution, "{attr_name}" must be an integer.')

        # List options (one value for each command) must have the right length.
        for attr_name in ['n_nodes', 'n_tasks', 'n_tasks_per_node', 'n_cpus_per_task', 'relative_node_idx']:
            attr_val = getattr(self, attr_name)
            if isinstance(attr_val, list) and len(attr_val) != n_commands:
                raise ValueError(f'Passed {n_commands} commands but '
                                 f'{len(attr_val)} {attr_name}: {attr_val}')

        # Prepend srun to all commands.
        srun_commands = self._create_srun_commands(commands)

        # Create a srun configuration file if necessary. The path must be
        # relative to the working directory, which can be changed with the
        # cwd keyword argument.
        if len(commands) > 1 and self.multiprog:
            with temporary_cd(kwargs.get('cwd', None)):
                self._create_multiprog_config_file(commands)

        return super().run(*srun_commands, **kwargs)

    def _create_srun_commands(self, commands):
        """Return the commands in list format with 'srun [options]' prepended."""
        # Convert commands to list format.
        commands = [cmd if not isinstance(cmd, CLITool) else cmd.to_subprocess() for cmd in commands]

        # Check if we need to run with multiprog.
        if len(commands) > 1 and self.multiprog:
            return self._create_srun_multiprog_command(commands)
        return self._create_srun_standard_commands(commands)

    def _create_srun_standard_commands(self, commands):
        """Return multiple srun commands in list format.

        ``commands`` must already be a list of commands in list format (not CLITool).
        """
        # Convert arguments to list format.
        n_nodes, n_tasks, n_tasks_per_node, n_cpus_per_task, relative_node_idx = _ensure_lists(
            len(commands),
            [self.n_nodes, self.n_tasks, self.n_tasks_per_node, self.n_cpus_per_task, self.relative_node_idx]
        )

        # Prepend srun to all commands.
        srun_commands = []
        for cmd_idx, cmd in enumerate(commands):
            # Create srun execution
            srun = SRunTool(
                n_nodes=n_nodes[cmd_idx],
                n_tasks=n_tasks[cmd_idx],
                n_tasks_per_node=n_tasks_per_node[cmd_idx],
                n_cpus_per_task=n_cpus_per_task[cmd_idx],
                relative_node_idx=relative_node_idx[cmd_idx],
            )

            # Prepend the srun command.
            srun_commands.append(srun.to_subprocess() + cmd)

        return srun_commands

    def _create_srun_multiprog_command(self, commands):
        """Return the srun command in list format for the 'srun --multi-prog' case."""
        # Make sure n_tasks is a list.
        n_tasks = _ensure_lists(len(commands), [self.n_tasks])[0]

        # We run a single job with a total number of tasks given by
        # the sum of the number of tasks assigned to each command.
        # We also ignore n_tasks_per_node since it's overwritten by n_tasks.
        srun = SRunTool(
            n_nodes=self.n_nodes,
            n_tasks=sum(n_tasks),
            n_cpus_per_task=self.n_cpus_per_task,
            relative_node_idx=self.relative_node_idx,
            multiprog_config_file_path=self.multiprog_config_file_path,
        )
        return [srun.to_subprocess()]

    def _create_multiprog_config_file(self, commands):
        """Create the configuration file to be passed to the --multi-prog option."""
        # Convert commands to list format.
        commands = [cmd if not isinstance(cmd, CLITool) else cmd.to_subprocess() for cmd in commands]

        # Make sure n_tasks is a list.
        n_tasks = _ensure_lists(len(commands), [self.n_tasks])[0]

        # Determine the task ranks to assign to each command.
        task_ranks = []
        current_task_rank = 0
        for n_tasks in n_tasks:
            ranks = str(current_task_rank) + '-' + str(current_task_rank+n_tasks-1)
            task_ranks.append(ranks)
            current_task_rank += n_tasks

        # Create the file.
        with open(self.multiprog_config_file_path, 'w') as f:
            for cmd_idx, cmd in enumerate(commands):
                line = ' '.join([task_ranks[cmd_idx], *cmd])
                f.write(line + '\n')


# =============================================================================
# UTILS
# =============================================================================

def _ensure_lists(length, args):
    """Make sure each argument is a list of the same length as ``commands``.

    This works only if the value of each argument cannot be of list type.

    """
    for var_idx, val in enumerate(args):
        if not isinstance(val, list):
            args[var_idx] = [val] * length
    return args
