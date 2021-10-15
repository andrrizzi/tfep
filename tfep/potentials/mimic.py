#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Modules and functions to compute QM/MM energies and gradients with MiMiC, GROMACS,
and CPMD.

The code interfaces with the molecular dynamics software through the command line.

"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import copy
import functools
import os
import shutil

import numpy as np
import pint
import torch

from tfep.utils.cli import Launcher, CLITool, KeyValueOption
from tfep.utils.geometry import flattened_to_atom
from tfep.utils.misc import temporary_cd, energies_array_to_tensor, forces_array_to_tensor
from tfep.utils.parallel import SerialStrategy

# TODO: CAPISCI COS'È KINETIC ENERGY IN WAVEFUNCTION OPTIMIZATION!
# TODO: THE PV CONTRIBUTION IS NOT COMPUTED! THE RETURNED ENERGY IS NOT THE REDUCED POTENTIAL.


# =============================================================================
# GROMACS/CPMD COMMANDS UTILITIES
# =============================================================================

class Cpmd(CLITool):
    """The CPMD command.

    The CPMD command takes two positional arguments: the path to the input script
    and the path to the directory including the definitions of the pseudopoentials.
    If given as relative paths, these must be relative to the working directory
    at the time of the execution.

    Parameters
    ----------
    input_file_path : str
        The CPMD input file path.
    pseudopotential_dir_path : str, optional
        The path to the directory including the pseudopotential definitions.
    executable_path : str, optional
        The executable path associated to the instance of the command. If this
        is not specified, the ``EXECUTABLE_PATH`` class variable is used instead.

    Examples
    --------

    >>> cmd = Cpmd('input.in', 'path/to/pseudopotentials/')
    >>> cmd.to_subprocess()
    ['cpmd', 'input.in', 'path/to/pseudopotentials/']

    If the executable is called differently, simply specify the executable path
    as a keyword argument.

    >>> cmd = Cpmd('input.in', 'path/to/pseudopotentials/', executable_path='cpmd.x')
    >>> cmd.to_subprocess()
    ['cpmd.x', 'input.in', 'path/to/pseudopotentials/']

    """
    EXECUTABLE_PATH = 'cpmd'


class GmxGrompp(CLITool):
    """The grompp subprogram of gmx from the GROMACS suite.

    The executable path variable only specifies the path to the gmx executable.
    The class takes care of adding the "grompp" subprogram after "gmx" so it
    does not have to be passed when the command is instantiated.

    All file paths must be either absolute or relative to the working directory
    used for executing the command.

    Parameters
    ----------
    executable_path : str, optional
        The executable path associated to the instance of the command. If this
        is not specified, the ``EXECUTABLE_PATH`` class variable is used instead.
    mdp_input_file_path : str, optional
        Path to input .mdp file. If a relative path is given, this must be relative
        to the working directory when the command is executed, not when ``GmxGrompp``
        is initialized.
    structure_input_file_path : str, optional
        The file including the structure and (if ``trajectory_input_file_path``
        is not provided) the starting coordinates of the calculation (e.g., in
        .gro or .pdb format).
    top_input_file_path : str, optional
        The path to the input .top topology file.
    trajectory_input_file_path : str, optional
        If given, the last frame is used to determine the starting coordinates
        (e.g., in .trr or .cpt) file.
    index_input_file_path : str, optional
        The path to the input .ndx file with the atom group indices.
    tpr_output_file_path : str, optional
        The path to the output .tpr file.
    n_max_warnings : int, optional
        The maximum number of warnings after which an error is raised.

    Examples
    --------

    >>> cmd = GmxGrompp(mdp_input_file_path='mysimulation.mdp', n_max_warnings=2)
    >>> cmd.to_subprocess()
    ['gmx', 'grompp', '-f', 'mysimulation.mdp', '-maxwarn', '2']

    If the executable is called differently, simply specify the executable path
    as a keyword argument.

    >>> cmd = GmxGrompp(executable_path='gmx_mpi', mdp_input_file_path='mysimulation.mdp')
    >>> cmd.to_subprocess()
    ['gmx_mpi', 'grompp', '-f', 'mysimulation.mdp']

    """
    EXECUTABLE_PATH = 'gmx'
    SUBPROGRAM = 'grompp'
    mdp_input_file_path = KeyValueOption('-f')
    structure_input_file_path = KeyValueOption('-c')
    top_input_file_path = KeyValueOption('-p')
    trajectory_input_file_path = KeyValueOption('-t')
    index_input_file_path = KeyValueOption('-n')
    tpr_output_file_path = KeyValueOption('-o')
    n_max_warnings = KeyValueOption('-maxwarn')


class GmxMdrun(CLITool):
    """The mdrun subprogram of gmx from the GROMACS suite.

    The executable path variable only specifies the path to the gmx executable.
    The class takes care of adding the "mdrun" subprogram after "gmx" so it
    does not have to be passed when the command is instantiated.

    Parameters
    ----------
    executable_path : str, optional
        The executable path associated to the instance of the command. If this
        is not specified, the ``EXECUTABLE_PATH`` class variable is used instead.
    tpr_input_file_path : str, optional
        Path to input .tpr file. If a relative path is given, this must be relative
        to the working directory when the command is executed, not when ``GmxMdrun``
        is initialized.
    default_file_name : str, optional
        Default file name used for all other files.
    n_omp_threads_per_mpi_rank : str, optional
        Number of OpenMP threads per MPI rank.

    Examples
    --------

    >>> cmd = GmxMdrun(default_file_name='mysimulation', n_omp_threads_per_mpi_rank=4)
    >>> cmd.to_subprocess()
    ['gmx', 'mdrun', '-deffnm', 'mysimulation', '-ntomp', '4']

    If the executable is called differently, simply specify the executable path
    as a keyword argument.

    >>> cmd = GmxMdrun(executable_path='gmx_mpi', default_file_name='mysimulation')
    >>> cmd.to_subprocess()
    ['gmx_mpi', 'mdrun', '-deffnm', 'mysimulation']

    """
    EXECUTABLE_PATH = 'gmx'
    SUBPROGRAM = 'mdrun'
    tpr_input_file_path = KeyValueOption('-s')
    default_file_name = KeyValueOption('-deffnm')
    n_omp_threads_per_mpi_rank = KeyValueOption('-ntomp')


# =============================================================================
# TORCH MODULE API
# =============================================================================

class PotentialMiMiC(torch.nn.Module):
    """Potential energy and forces with MiMiC.

    This ``Module`` wraps :class:``.PotentialEnergyMiMiCFunc`` to provide a
    differentiable potential energy function for training. It also provides an
    API to compute energies and forces with MiMiC from batches of coordinates in
    ``numpy`` arrays in standard format (i.e., shape ``(n_atoms, 3)``) rather
    than flattened ``torch.Tensor``s (i.e., shape ``(n_atoms*3,)``).

    Parameters
    ----------
    cpmd_cmd : tfep.potentials.mimic.Cpmd
        The CPMD command to be run for MiMiC's execution that encapsulates the
        path to the CPMD input script and options.

        The ``&MIMIC.PATHS`` option and atomic coordinates can be placeholders
        as they are automatically set by this function according to the
        ``working_dir_path`` and ``batch_positions`` arguments. All other options
        must be set correctly for the function to run successfully.
    mdrun_cmd : tfep.potentials.mimic.GmxMdrun
        The GMX mdrun command to be run for MiMiC's execution that encapsulates
        the path to the GROMACS input script and running options.

        The ``mdrun_cmd.tpr_input_file_path`` can be left unset since a new
        ``.tpr`` file with the correct positions is automatically generated with
        ``gromp_cmd``.
    grompp_cmd : tfep.potentials.mimic.GmxGrompp, optional
        This command is used to generate the the ``.tpr`` file with the correct
        coordinates. To do so, the batch positions are first stored in a ``.trr``
        file which is then passed to grompp. Thus, the ``GmxGrompp.tpr_output_file_path``
        and ``GmxGrompp.trajectory_input_file_path`` options can be ``None``.
    launcher : tfep.utils.cli.Launcher, optional
        The ``Launcher`` to use to run the ``cpmd_cmd`` and ``mdrun_cmd``. If
        not passed, a new :class:`tfep.utils.cli.Launcher` is created.
    grompp_launcher : tfep.utils.cli.Launcher, optional
        The ``Launcher`` to use to run the ``grompp_cmd`` command. If not passed,
        a new :class:`tfep.utils.cli.Launcher` is created.
    positions_unit : pint.Unit, optional
        The unit of the positions passed. This is used to appropriately convert
        ``batch_positions`` to the units used by MiMiC. If ``None``, no conversion
        is performed, which assumes that the input positions are in Bohr.
    energy_unit : pint.Unit, optional
        The unit used for the returned energies (and as a consequence forces).
        This is used to appropriately convert MiMiC energies into the desired
        units. If ``None``, no conversion is performed, which means that energies
        and forces will be in hartrees and hartrees/bohr respectively.
    precompute_gradient : bool, optional
        If ``False``, the ``FTRAJECTORY`` file is not read after executing MiMiC.
        This might save a small amount of time if backpropagation is not needed.
    working_dir_path : str or List[str], optional
        The working directory to be used to run MiMiC and grompp. This must exist.
        If a list, ``batch_positions[i]`` is evaluated in the directory
        ``working_dir_path[i]``.
    cleanup_working_dir : bool, optional
        If ``True`` and ``working_dir_path`` is passed, all the files inside the
        working directory are removed after executing MiMiC. The directory(s)
        itself is not deleted.
    parallelization_strategy : tfep.utils.parallel.ParallelizationStrategy, optional
        The parallelization strategy used to distribute batches of energy and
        gradient calculations. By default, these are executed serially.
    launcher_kwargs : Dict, optional
        Other kwargs for ``launcher`` (with the exception of ``cwd`` which is
        automatically determined based on ``working_dir_path``).
    grompp_launcher_kwargs : Dict, optional
        Other kwargs for ``grompp_launcher``.

    See Also
    --------
    :class:`.PotentialEnergyMiMiCFunc`
        More details on input parameters and implementation details.

    """
    def __init__(
            self,
            cpmd_cmd,
            mdrun_cmd,
            grompp_cmd,
            launcher=None,
            grompp_launcher=None,
            positions_unit=None,
            energy_unit=None,
            precompute_gradient=True,
            working_dir_path=None,
            cleanup_working_dir=False,
            parallelization_strategy=None,
            launcher_kwargs=None,
            grompp_launcher_kwargs=None
    ):
        super().__init__()

        self.cpmd_cmd = cpmd_cmd
        self.mdrun_cmd = mdrun_cmd
        self.grompp_cmd = grompp_cmd
        self.launcher = launcher
        self.grompp_launcher = grompp_launcher
        self.positions_unit = positions_unit
        self.energy_unit = energy_unit
        self.precompute_gradient = precompute_gradient
        self.working_dir_path = working_dir_path
        self.cleanup_working_dir = cleanup_working_dir
        self.parallelization_strategy = parallelization_strategy
        self.launcher_kwargs = launcher_kwargs
        self.grompp_launcher_kwargs = grompp_launcher_kwargs

    def forward(self, batch_positions, batch_box_vectors):
        """Compute a differential potential energy for a batch of configurations.

        Parameters
        ----------
        batch_positions : torch.Tensor
            A tensor of positions in flattened format (i.e., with shape
            ``(batch_size, 3*n_atoms)``) in units of ``self.positions_units``
            (or MiMiC units is ``positions_units`` was not provided).

            Note that the order of the atoms is assumed to be that of the GROMACS
            input files, not the one used internally by CPMD (which always puts the
            QM atoms first). The atom map used to match them is based on the CPMD
            input script option ``&MIMIC.OVERLAPS``.
        batch_box_vectors : torch.Tensor, optional
            An tensor of box vectors with shape ``(batch_size, 3)`` defining the
            orthorhombic box side lengths (the only one currently supported in MiMiC)
            in units of ``self.positions_units`` (or MiMiC units is ``positions_units``
            was not provided).

        Returns
        -------
        potential_energy : torch.Tensor
            ``potential_energy[i]`` is the potential energy of configuration
            ``batch_positions[i]`` and ``batch_box_vectors[i]`` in units of
            ``self.energy_unit`` (or MiMiC units if ``energy_unit`` is not
            provided).

        """
        return potential_energy_mimic(
            batch_positions=batch_positions,
            batch_box_vectors=batch_box_vectors,
            cpmd_cmd=self.cpmd_cmd,
            mdrun_cmd=self.mdrun_cmd,
            grompp_cmd=self.grompp_cmd,
            launcher=self.launcher,
            grompp_launcher=self.grompp_launcher,
            positions_unit=self.positions_unit,
            energy_unit=self.energy_unit,
            precompute_gradient=self.precompute_gradient,
            working_dir_path=self.working_dir_path,
            cleanup_working_dir=self.cleanup_working_dir,
            parallelization_strategy=self.parallelization_strategy,
            launcher_kwargs=self.launcher_kwargs,
            grompp_launcher_kwargs=self.grompp_launcher_kwargs
        )

    def energy(self, batch_positions, batch_box_vectors):
        """Compute a the potential energy of a batch of configurations.

        Parameters
        ----------
        batch_positions : pint.Quantity, optional
            An array of positions with units and shape: ``(batch_size, n_atoms, 3)``
            or ``(n_atoms, 3)``. If no units are attached to the array, it is
            assumed the positions are is in ``self.position_unit`` units (or MiMiC
            units if ``position_unit`` was not provided).

            Note that the order of the atoms is assumed to be that of the GROMACS
            input files, not the one used internally by CPMD (which always puts the
            QM atoms first). The atom map used to match them is based on the CPMD
            input script option ``&MIMIC.OVERLAPS``.
        batch_box_vectors : pint.Quantity, optional
            An array of box vectors with units and shape: ``(batch_size, 3)`` or
            ``(3,)`` defining the orthorhombic box side lengths (the only one currently
            supported in MiMiC). If no units are attached to the array, it is
            assumed the positions are is in ``self.position_unit`` units (or
            MiMiC units if ``position_unit`` was not provided).

        Returns
        -------
        potential_energies : pint.Quantity
            ``potential_energies[i]`` is the potential energy of configuration
            ``batch_positions[i]`` and ``batch_box_vectors[i]``.

        """
        # Add units.
        batch_positions = self._ensure_positions_has_units(batch_positions)
        batch_box_vectors = self._ensure_positions_has_units(batch_box_vectors)
        return _run_mimic(
            self.cpmd_cmd,
            self.mdrun_cmd,
            grompp_cmd=self.grompp_cmd,
            batch_positions=batch_positions,
            batch_box_vectors=batch_box_vectors,
            launcher=self.launcher,
            grompp_launcher=self.grompp_launcher,
            return_energy=True,
            return_force=False,
            unit_registry=None,
            working_dir_path=self.working_dir_path,
            cleanup_working_dir=self.cleanup_working_dir,
            parallelization_strategy=self.parallelization_strategy,
            launcher_kwargs=self.launcher_kwargs,
            grompp_launcher_kwargs=self.grompp_launcher_kwargs,
        )

    def force(self, batch_positions, batch_box_vectors):
        """Compute the force for a batch of configurations.

        Parameters
        ----------
        batch_positions : pint.Quantity, optional
            An array of positions with units and shape: ``(batch_size, n_atoms, 3)``
            or ``(n_atoms, 3)``. If no units are attached to the array, it is
            assumed the positions are is in ``self.position_unit`` units (or MiMiC
            units if ``position_unit`` was not provided).

            Note that the order of the atoms is assumed to be that of the GROMACS
            input files, not the one used internally by CPMD (which always puts the
            QM atoms first). The atom map used to match them is based on the CPMD
            input script option ``&MIMIC.OVERLAPS``.
        batch_box_vectors : pint.Quantity, optional
            An array of box vectors with units and shape: ``(batch_size, 3)`` or
            ``(3,)`` defining the orthorhombic box side lengths (the only one currently
            supported in MiMiC). If no units are attached to the array, it is
            assumed the positions are is in ``self.position_unit`` units (or
            MiMiC units if ``position_unit`` was not provided).

        Returns
        -------
        forces : pint.Quantity
            ``forces[i]`` is the force of configuration ``batch_positions[i]``
            and ``batch_box_vectors[i]``.

        """
        # Add units.
        batch_positions = self._ensure_positions_has_units(batch_positions)
        batch_box_vectors = self._ensure_positions_has_units(batch_box_vectors)
        return _run_mimic(
            self.cpmd_cmd,
            self.mdrun_cmd,
            grompp_cmd=self.grompp_cmd,
            batch_positions=batch_positions,
            batch_box_vectors=batch_box_vectors,
            launcher=self.launcher,
            grompp_launcher=self.grompp_launcher,
            return_energy=False,
            return_force=True,
            unit_registry=None,
            working_dir_path=self.working_dir_path,
            cleanup_working_dir=self.cleanup_working_dir,
            parallelization_strategy=self.parallelization_strategy,
            launcher_kwargs=self.launcher_kwargs,
            grompp_launcher_kwargs=self.grompp_launcher_kwargs,
        )

    def _ensure_positions_has_units(self, batch_positions):
        """Add units to an array of positions."""
        try:
            batch_positions.units
        except AttributeError:
            if self.position_unit is not None:
                return batch_positions * self.position_unit
            else:
                # Find a unit registry.
                if self.energy_unit is None:
                    ureg = pint.UnitRegistry()
                else:
                    ureg = self.energy_unit._REGISTRY
                return batch_positions * ureg.bohr
        return batch_positions


# =============================================================================
# TORCH FUNCTIONAL API
# =============================================================================

class PotentialEnergyMiMiCFunc(torch.autograd.Function):
    """PyTorch-differentiable QM/MM potential energy function wrapped around MiMiC.

    The function calls MiMiC through the command line interface using user-prepared
    CPMD and GROMACS input files and reads the resulting energies and forces.
    The function can use the input files as templates and evaluate energies and
    forces for batches of configurations. To do this, the input files must be
    setup to generate the ``ENERGIES`` and ``FTRAJECTORY`` trajectory files,
    which can be  generated when both ``&CPMD.MOLECULAR DYNAMICS`` and
    ``&CPMD.TRAJECTORY FORCES`` options are used in the CPMD input script. The
    returned energy and force will be read from the first entry in the respective
    trajectory file.

    .. note::
        With this mechanism, the returned energy and force will be associated to
        the configuration AFTER the first integration step, not to the configuration
        passed in ``batch_positions``. However, this is necessary as currently
        there is no way in MiMiC to save the forces of all the atoms (including
        the MM atoms) with a single wavefunction optimization calculation.

        To maximize the efficiency and reduce the error, we suggest setting
        ``&CPMD.TRAJECTORY FORCES`` and ``&CPMD.MAXSTEP`` to 1 and set a very
        small ``&CPMD.TIMESTEP`` (e.g., 0.000001 a.u.).

    .. note::
        The order of the atoms typically differs in GROMACS and CPMD. The input
        and output positions and forces of this function always use the GROMACS
        atom order.

    If backpropagation is not necessary, ``&CPMD.TRAJECTORY FORCES`` can be left
    out but the option ``precompute_gradients`` must be set to ``False`` or the
    function will expect an output ``FTRAJECTORY`` file.

    Because GROMACS does not support resuming from a coordinate file (it requires
    a full checkpoint file), to perform the calculation starting from an arbitrary
    batch data point with different coordinates than those in the tpr file, we
    need to regenerate a temporary tpr file with the correct coordinates with
    GROMPP.

    Only orthorhombic boxes are currently supported by MiMiC. thus the box vectors
    are simply specified by the lengths of the box sides.

    The function supports running each MiMiC execution in a separate working
    directory to safely support batch parallelization schemes through
    :class:``tfep.utils.parallel.ParallelizationStrategy`` objects.

    Parameters
    ----------
    ctx : torch.autograd.function._ContextMethodMixin
        A context to save information for the gradient.
    batch_positions : torch.Tensor
        A tensor of positions in flattened format (i.e., with shape
        ``(batch_size, 3*n_atoms)``).

        Note that the order of the atoms is assumed to be that of the GROMACS
        input files, not the one used internally by CPMD (which always puts the
        QM atoms first). The atom map used to match them is based on the CPMD
        input script option ``&MIMIC.OVERLAPS``.
    batch_box_vectors : pint.Quantity, optional
        An tensor of box vectors with shape ``(batch_size, 3)`` defining the
        orthorhombic box side lengths (the only one currently supported in MiMiC).
    cpmd_cmd : tfep.potentials.mimic.Cpmd
        The CPMD command to be run for MiMiC's execution that encapsulates the
        path to the CPMD input script and options.

        The ``&MIMIC.PATHS`` option and atomic coordinates can be placeholders
        as they are automatically set by this function according to the
        ``working_dir_path`` and ``batch_positions`` arguments. All other options
        must be set correctly for the function to run successfully.
    mdrun_cmd : tfep.potentials.mimic.GmxMdrun
        The GMX mdrun command to be run for MiMiC's execution that encapsulates
        the path to the GROMACS input script and running options.

        The ``mdrun_cmd.tpr_input_file_path`` can be left unset since a new
        ``.tpr`` file with the correct positions is automatically generated with
        ``gromp_cmd``.
    grompp_cmd : tfep.potentials.mimic.GmxGrompp, optional
        This command is used to generate the the ``.tpr`` file with the correct
        coordinates. To do so, the batch positions are first stored in a ``.trr``
        file which is then passed to grompp. Thus, the ``GmxGrompp.tpr_output_file_path``
        and ``GmxGrompp.trajectory_input_file_path`` options can be ``None``.
    launcher : tfep.utils.cli.Launcher, optional
        The ``Launcher`` to use to run the ``cpmd_cmd`` and ``mdrun_cmd``. If
        not passed, a new :class:`tfep.utils.cli.Launcher` is created.
    grompp_launcher : tfep.utils.cli.Launcher, optional
        The ``Launcher`` to use to run the ``grompp_cmd`` command. If not passed,
        a new :class:`tfep.utils.cli.Launcher` is created.
    positions_unit : pint.Unit, optional
        The unit of the positions passed. This is used to appropriately convert
        ``batch_positions`` to the units used by MiMiC. If ``None``, no conversion
        is performed, which assumes that the input positions are in Bohr.
    energy_unit : pint.Unit, optional
        The unit used for the returned energies (and as a consequence forces).
        This is used to appropriately convert MiMiC energies into the desired
        units. If ``None``, no conversion is performed, which means that energies
        and forces will be in hartrees and hartrees/bohr respectively.
    precompute_gradient : bool, optional
        If ``False``, the ``FTRAJECTORY`` file is not read after executing MiMiC.
        This might save a small amount of time if backpropagation is not needed.
    working_dir_path : str or List[str], optional
        The working directory to be used to run MiMiC and grompp. This must exist.
        If a list, ``batch_positions[i]`` is evaluated in the directory
        ``working_dir_path[i]``.
    cleanup_working_dir : bool, optional
        If ``True`` and ``working_dir_path`` is passed, all the files inside the
        working directory are removed after executing MiMiC. The directory(s)
        itself is not deleted.
    parallelization_strategy : tfep.utils.parallel.ParallelizationStrategy, optional
        The parallelization strategy used to distribute batches of energy and
        gradient calculations. By default, these are executed serially.
    launcher_kwargs : Dict, optional
        Other kwargs for ``launcher`` (with the exception of ``cwd`` which is
        automatically determined based on ``working_dir_path``).
    grompp_launcher_kwargs : Dict, optional
        Other kwargs for ``grompp_launcher``.

    Returns
    -------
    potentials : torch.Tensor
        ``potentials[i]`` is the potential energy of configuration
        ``batch_positions[i]``.

    See Also
    --------
    :class:`.PotentialMiMiC`
        ``Module`` API for computing potential energies with MiMiC.

    """

    @staticmethod
    def forward(
            ctx,
            batch_positions,
            batch_box_vectors,
            cpmd_cmd,
            mdrun_cmd,
            grompp_cmd,
            launcher=None,
            grompp_launcher=None,
            positions_unit=None,
            energy_unit=None,
            precompute_gradient=True,
            working_dir_path=None,
            cleanup_working_dir=False,
            parallelization_strategy=None,
            launcher_kwargs=None,
            grompp_launcher_kwargs=None
    ):
        """Compute the potential energy of the molecule with MiMiC."""
        # Check for unit registry.
        if positions_unit is not None:
            unit_registry = positions_unit._REGISTRY
        elif energy_unit is not None:
            unit_registry = energy_unit._REGISTRY
        else:
            unit_registry = pint.UnitRegistry()

        # Convert flattened position tensor to numpy array of shape
        # (batch_size, n_atoms, 3) and attach units.
        if positions_unit is None:
            positions_unit = unit_registry.bohr

        batch_positions_arr = flattened_to_atom(batch_positions.detach().numpy())
        batch_positions_arr *= positions_unit

        if batch_box_vectors is None:
            batch_box_vectors_arr = None
        else:
            batch_box_vectors_arr = batch_box_vectors.detach().numpy()
            batch_box_vectors_arr *= positions_unit

        # Determine whether we need forces.
        if precompute_gradient:
            return_force = True
        else:
            return_force = False

        # Run MiMiC.
        result = _run_mimic(
            cpmd_cmd,
            mdrun_cmd,
            grompp_cmd,
            batch_positions=batch_positions_arr,
            batch_box_vectors=batch_box_vectors_arr,
            launcher=launcher,
            grompp_launcher=grompp_launcher,
            return_energy=True,
            return_force=return_force,
            unit_registry=unit_registry,
            working_dir_path=working_dir_path,
            cleanup_working_dir=cleanup_working_dir,
            parallelization_strategy=parallelization_strategy,
            launcher_kwargs=launcher_kwargs,
            grompp_launcher_kwargs=grompp_launcher_kwargs
        )

        if not precompute_gradient:
            energies = result
        else:
            energies, forces = result
            # Convert the force to a flattened tensor before storing it in ctx.
            # to compute the gradient during backpropagation.
            forces = forces_array_to_tensor(forces, positions_unit, energy_unit,
                                            dtype=batch_positions.dtype)
            ctx.save_for_backward(forces)

        # Convert to unitless Tensor.
        energies = energies_array_to_tensor(energies, energy_unit, batch_positions.dtype)
        return energies

    @staticmethod
    def backward(ctx, grad_output):
        """Compute the gradient of the potential energy."""
        # We still need to return a None gradient for each
        # input of forward() beside batch_positions.
        n_input_args = 15
        grad_input = [None for _ in range(n_input_args)]

        # Compute gradient w.r.t. batch_positions.
        if ctx.needs_input_grad[0]:
            # Retrieve pre-computed forces.
            if len(ctx.saved_tensors) == 1:
                forces, = ctx.saved_tensors
            else:
                raise ValueError('Cannot compute gradients if precompute_gradient '
                                 'option is set to False.')

            # Accumulate gradient, which has opposite sign of the force.
            grad_input[0] = -forces * grad_output[:, None]

        return tuple(grad_input)


def potential_energy_mimic(
        batch_positions,
        batch_box_vectors,
        cpmd_cmd,
        mdrun_cmd,
        grompp_cmd,
        launcher=None,
        grompp_launcher=None,
        positions_unit=None,
        energy_unit=None,
        precompute_gradient=True,
        working_dir_path=None,
        cleanup_working_dir=False,
        parallelization_strategy=None,
        launcher_kwargs=None,
        grompp_launcher_kwargs=None,
):
    """PyTorch-differentiable QM/MM potential energy using MiMIC.

    PyTorch ``Function``s do not accept keyword arguments. This function wraps
    :func:`.PotentialEnergyMiMiCFunc.apply` to enable standard functional notation.
    See the documentation on the original function for the input parameters.

    See Also
    --------
    :class:`.PotentialEnergyMiMiCFunc`
        More details on input parameters and implementation details.

    """
    # apply() does not accept keyword arguments.
    return PotentialEnergyMiMiCFunc.apply(
        batch_positions,
        batch_box_vectors,
        cpmd_cmd,
        mdrun_cmd,
        grompp_cmd,
        launcher,
        grompp_launcher,
        positions_unit,
        energy_unit,
        precompute_gradient,
        working_dir_path,
        cleanup_working_dir,
        parallelization_strategy,
        launcher_kwargs,
        grompp_launcher_kwargs,
    )


# =============================================================================
# MAIN FUNCTIONS WRAPPING MIMIC
# =============================================================================

def _run_mimic(
        cpmd_cmd,
        mdrun_cmd,
        grompp_cmd=None,
        batch_positions=None,
        batch_box_vectors=None,
        launcher=None,
        grompp_launcher=None,
        return_energy=False,
        return_force=False,
        unit_registry=None,
        working_dir_path=None,
        cleanup_working_dir=False,
        parallelization_strategy=None,
        launcher_kwargs=None,
        grompp_launcher_kwargs=None,
):
    """Run MiMiC.

    See also the docstring of ``PotentialEnergyMiMiCFunc``.

    Some notes to remember about the implementation.

    The input batch_positions are passed in GROMACS order (which is likely the most
    common user case), but the CPMD output files (e.g., FTRAJECTORY, GEOMETRY)
    use the CPMD order. Thus the function must map the atom positions back and
    forth using the atom map in &MIMIC.OVERLAPS.

    MiMiC centers the QM atoms in the box with a translation. This means that
    the positions passed do not correspond to those evaluated, but because only
    a translation is performed, the forces should be identical in both frames of
    reference.

    The &MIMIC.PATHS in the CPMD input script must point to the working directory
    at the time of the execution. If this is not the case, this function creates
    a copy of the script in the working directory (called 'cpmd.inp') with the
    correct &MIMIC.PATHS option to be used for executing the command.

    Parameters
    ----------
    cpmd_cmd : tfep.potentials.mimic.Cpmd
        The CPMD command to be run for MiMiC encapsulating the path to the input
        script and options.

        The ``&MIMIC.PATHS`` option and atomic coordinates can be placeholders
        as they are automatically set by this function according to the
        ``working_dir_path`` and ``batch_positions`` arguments. All other options
        must be set correctly for the execution.
    mdrun_cmd : tfep.potentials.mimic.GmxMdrun
        The GMX mdrun command to be run for MiMiC encapsulating the path to the input
        script and running options.

        When ``batch_positions`` is ``None``. The positions are taken from the
        ``.tpr`` file encapsulated in this command. Otherwise, this command is
        run using a new ``.tpr`` file generated with ``grompp_cmd``.
    grompp_cmd : tfep.potentials.mimic.GmxGrompp, optional
        If ``batch_positions`` is passed, this command is used to generate the
        the ``.tpr`` file with the correct starting coordinates. To do so, the
        batch positions are first stored in a ``.trr`` file which is then passed
        to grompp. Thus, the ``GmxGrompp.tpr_output_file_path`` and
        ``GmxGrompp.trajectory_input_file_path`` options can be ``None``.
    batch_positions : pint.Quantity, optional
        An array of positions with units and shape: ``(batch_size, n_atoms, 3)``
        or ``(n_atoms, 3)``. If ``None``, the coordinates in the input files are
        evaluated.

        Note that the order of the atoms is assumed to be that of the GROMACS
        input files, not the one used internally by CPMD (which always puts the
        QM atoms first). The atom map used to match them is based on the CPMD
        input script option ``&MIMIC.OVERLAPS``.
    batch_box_vectors : pint.Quantity, optional
        An array of box vectors with units and shape: ``(batch_size, 3)`` or
        ``(3,)`` defining the orthorhombic box side lengths (the only one currently
        supported in MiMiC). If ``None``, the box vectors in the input files are
        evaluated.
    launcher : tfep.utils.cli.Launcher, optional
        The ``Launcher`` to use to run the ``cpmd_cmd`` and ``mdrun_cmd``. If
        not passed, a new :class:`tfep.utils.cli.Launcher` is created.
    grompp_launcher : tfep.utils.cli.Launcher, optional
        The ``Launcher`` to use to run the ``grompp_cmd`` command. If not passed,
        a new :class:`tfep.utils.cli.Launcher` is created.
    return_energy : bool, optional
        If ``True``, the potential energies are returned.
    return_force : bool, optional
        If ``True``, the forces are returned.
    unit_registry : pint.UnitRegistry, optional
        The unit registry to use for the energy units. If ``None`` and
        ``batch_positions`` has units attached, the unit registry of the positions
        is used. Otherwise, the returned values use a new ``UnitRegistry`` is
        created.
    working_dir_path : str or List[str], optional
        The working directory to be used to run MiMiC (and eventually grompp).
        This must exist. If ``batch_positions`` is passed, this can be a list
        specifying one directory for each batch configuration.
    cleanup_working_dir : bool, optional
        If ``True`` and ``working_dir_path`` is passed, all the files inside the
        working directory are removed after executing MiMiC. The directory(s)
        itself is not deleted.
    parallelization_strategy : tfep.utils.parallel.ParallelizationStrategy, optional
        The parallelization strategy used to distribute batches of energy and
        gradient calculations. By default, these are executed serially.
    launcher_kwargs : Dict, optional
        Other kwargs for ``launcher`` (with the exception of ``cwd`` which is
        automatically determined based on ``working_dir_path``).
    grompp_launcher_kwargs : Dict, optional
        Other kwargs for ``grompp_launcher``.

    Returns
    -------
    energies : pint.Quantity, optional
        If ``batch_positions`` was not given or if it is of shape ``(n_atoms, 3)``,
        this is a single number. Otherwise, ``energies[i]`` is the potential of
        configuration ``batch_positions[i]``.

        This is returned only if ``return_energy`` is ``True``.
    forces : torch.Tensor or pint.Quantity, optional
        If ``batch_positions`` was not given or if it is of shape ``(n_atoms, 3)``,
        this is a numpy array of shape ``(n_atoms, 3)`` with the molecule's force.
        Otherwise, ``forces[i]`` is the force of configuration ``batch_positions[i]``.

        This is returned only if ``return_force`` is ``True``.

        As with the ``batch_positions``, the order of the atoms in each force
        follows that of the GROMACS input files, not the one used internally by
        CPMD (which always puts the QM atoms first). The atom map used to match
        them is based on the CPMD input script option ``&MIMIC.OVERLAPS``.

    """
    # Mutable default arguments.
    if parallelization_strategy is None:
        parallelization_strategy = SerialStrategy()

    # Obtain the unit registry.
    if batch_positions is not None:
        unit_registry = batch_positions._REGISTRY
    elif unit_registry is None:
        unit_registry = pint.UnitRegistry()

    # First dimension should be batch. Save the input shape before modifying.
    # We need this to know the result.
    is_batch = (batch_positions is not None) and (len(batch_positions.shape) >= 3)
    if not is_batch:
        batch_positions = [batch_positions]
        batch_box_vectors = [batch_box_vectors]
    elif batch_box_vectors is None:
        # batch_box_vectors still needs to be in batch format.
        batch_box_vectors = [batch_box_vectors] * batch_positions.shape[0]

    # Make sure working_dir_path is in batch format.
    n_configurations = len(batch_positions)
    if working_dir_path is None or isinstance(working_dir_path, str):
        working_dir_path = [working_dir_path] * n_configurations
    else:
        working_dir_path = [os.path.realpath(p) for p in working_dir_path]

    # Run the command.
    task = functools.partial(
        _run_mimic_task, cpmd_cmd, mdrun_cmd, grompp_cmd, launcher, grompp_launcher,
        return_energy, return_force, cleanup_working_dir, launcher_kwargs,
        grompp_launcher_kwargs
    )
    distributed_args = zip(batch_positions, batch_box_vectors, working_dir_path)
    returned_values = parallelization_strategy.run(task, distributed_args)

    # Convert from a list of shape (batch_size, 2) to (2, batch_size).
    returned_values = list(zip(*returned_values))

    # Convert to a single array or to not-batch format.
    if is_batch:
        returned_values = [np.array(res) for res in returned_values]
    else:
        returned_values = [res[0] for res in returned_values]

    # Add units.
    if return_energy:
        returned_values[0] = returned_values[0] * unit_registry.hartree
    if return_force:
        returned_values[-1] = returned_values[-1] * unit_registry.hartree / unit_registry.bohr

    return returned_values


def _run_mimic_task(
        cpmd_cmd,
        mdrun_cmd,
        grompp_cmd,
        launcher,
        grompp_launcher,
        return_energy,
        return_force,
        cleanup_working_dir,
        launcher_kwargs,
        grompp_launcher_kwargs,
        positions,
        box_vectors,
        working_dir_path,
):
    """This is the task passed to the ``ParallelizationStrategy`` to run MiMiC.

    The arguments are essentially the same as _run_mimic but for a single data
    point of a batch.

    It returns energy and force (depending on ``return_energy`` and ``return_force``
    respectively) as unitless arrays in units of hartree and hartree/bohr
    respectively.

    """
    # Mutable default arguments.
    if launcher_kwargs is None:
        launcher_kwargs = {}
    if grompp_launcher_kwargs is None:
        grompp_launcher_kwargs = {}

    # If no working directory was specified, this is executed in the current one.
    if working_dir_path is None:
        working_dir_path = os.getcwd()

    # Make sure working_dir_path is an absolute path
    working_dir_path = os.path.realpath(working_dir_path)

    # Prepare the cpmd command.
    cpmd_cmd, gromacs_to_cpmd_atom_map = _prepare_cpmd_command(
        cpmd_cmd, working_dir_path, positions, box_vectors)

    # Prepare the mdrun command.
    mdrun_cmd = _prepare_mdrun_command(
        mdrun_cmd, grompp_cmd, working_dir_path, positions, box_vectors,
        grompp_launcher, **grompp_launcher_kwargs
    )

    # Run MiMiC.
    if launcher is None:
        launcher = Launcher()
    launcher.run(cpmd_cmd, mdrun_cmd, cwd=working_dir_path, **launcher_kwargs)

    # Read the energy/force from the trajectory files.
    returned_values = []
    if return_energy:
        energy = _read_first_energy(working_dir_path)
        returned_values.append(energy)
    if return_force:
        force = _read_first_force(working_dir_path, gromacs_to_cpmd_atom_map)
        returned_values.append(force)

    # Clean up directory.
    if cleanup_working_dir:
        for file_name in os.listdir(working_dir_path):
            file_path = os.path.join(working_dir_path, file_name)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

    return returned_values


def _prepare_cpmd_command(cpmd_cmd, working_dir_path, positions=None, box_vectors=None):
    """Prepare the CPMD script and command.

    This function:
    - Makes sure the &MIMIC.PATHS option in the CPMD input script point to the
      working directory used for the execution.
    - Sets the positions and box vectors in the CPMD input script based to those
      in ``positions`` (if the positions are passed).
    - Returns an atom map of the QM atoms mapping the atom index in GROMACS to
      its corresponding atom index in CPMD.

    At the time of writing the keyword &SYSTEM.ANGSTROM is not supported in
    MiMiC so all positions are written in Bohr. Moreover, only orthorhombic boxes
    are supported in MiMiC, thus ``box_vectors`` simply contains the lengths of
    the box sides.

    When changes are needed, the modified version is created inside the working
    directory with the name 'cpmd.inp' to preserve the original file, and the
    returned ``Cpmd`` object is a copy of the ``cpmd_cmd`` argument pointing to
    the correct input file.

    The atom indices map in &MIMIC.OVERLAPS is used to resolve which positions
    in ``positions`` refer to which CPMD atom.

    Parameters
    ----------
    cpmd_cmd : CLITool
        The original CPMD command.
    working_dir_path : str
        The path to the working directory used for executing the mimic command.
    positions : pint.Quantity, optional
        An array of positions with units and shape ``(n_atoms, 3)``. If ``None``,
        the coordinates in the input file are evaluated.
    box_vectors : pint.Quantity, optional
        An array of box vectors with units and shape ``(3,)`` defining the
        orthorhombic box side lengths (the only one currently supported in
        MiMiC). If ``None``, the box in the input file is evaluated.

    Returns
    -------
    cpmd_cmd : tfep.potentials.mimic.Cpmd
        A CPMD commandthat points to the correct (original or copied) input file.
    gromacs_to_cpmd_atom_map : Dict[int, int]
        Associates a GROMACS atom index to a CPMD atom index (as given by
        ``&MIMIC.OVERLAPS``).

    """
    OUTPUT_CPMD_FILE_NAME = 'cpmd.inp'

    # Read the original CPMD script file. If a relative path, this is relative
    # to the execution working directory.
    with temporary_cd(working_dir_path):
        cpmd_input_file_path = os.path.realpath(cpmd_cmd.args[0])

    # Parse the file.
    (cpmd_file_lines, path_line_idx, box_vectors_line_idx, gromacs_to_cpmd_atom_map,
        cpmd_atom_to_line_idx) = _parse_cpmd_input(cpmd_input_file_path)

    # If the path is incorrect, update it.
    paths_value = cpmd_file_lines[path_line_idx].strip()
    update_path = working_dir_path != os.path.realpath(paths_value)
    if update_path:
        cpmd_file_lines[path_line_idx] = working_dir_path + '\n'

    # Update the box vectors and positions.
    if positions is not None:
        if box_vectors is not None:
            box_vectors_bohr = box_vectors.to('bohr').magnitude
            cpmd_file_lines[box_vectors_line_idx] = ' '.join([str(x) for x in box_vectors_bohr]) + '\n'

        # Cycle through all atoms and update their lines one by one.
        for gromacs_atom_idx, cpmd_atom_idx in gromacs_to_cpmd_atom_map.items():
            line_idx = cpmd_atom_to_line_idx[cpmd_atom_idx]
            atom_position = positions[gromacs_atom_idx].to('bohr').magnitude
            cpmd_file_lines[line_idx] = ' '.join([str(x) for x in atom_position]) + '\n'

    # Create a modified copy of the file and update the command to point to it.
    if update_path:
        # Write the file.
        with open(os.path.join(working_dir_path, OUTPUT_CPMD_FILE_NAME), 'w') as f:
            for line in cpmd_file_lines:
                f.write(line)

        # Update the command without modifying the original.
        cpmd_cmd = copy.deepcopy(cpmd_cmd)
        cpmd_cmd.args = [OUTPUT_CPMD_FILE_NAME] + list(cpmd_cmd.args)[1:]

    return cpmd_cmd, gromacs_to_cpmd_atom_map


def _prepare_mdrun_command(mdrun_cmd, grompp_cmd, working_dir_path,
                           positions=None, box_vectors=None, grompp_launcher=None, **kwargs):
    """Prepare the mdrun command for execution in the given working directory.

    If ``positions`` is not ``None``, the function uses the grompp command to
    create a new .tpr file with the correct positions and box vectors. Currently,
    it is impossible to update the tpr file with the new positions using the
    GROMACS tools without calling grompp again.

    In this case, the returned ``mdrun_cmd`` is a copy of the original that
    points to the new .tpr file.

    Only orthorhombic boxes are currently supported by MiMiC. thus ``box_vectors``
    simply contains the lengths of the box sides.

    Parameters
    ----------
    mdrun_cmd : GmxMdrun
        The mdrun command to prepare. This is modified.
    grompp_cmd : GmxGrompp
        The grompp command used to generate the new .tpr file. To do so, the
        batch positions are first stored in a ``.trr`` file which is then passed
        to grompp. Thus, the ``GmxGrompp.tpr_output_file_path`` and
        ``GmxGrompp.trajectory_input_file_path`` options can be ``None``.
    working_dir_path : str
        The path to the working directory used for executing the mimic (and
        eventually grompp) command.
    positions : pint.Quantity, optional
        An array of positions with units and shape ``(n_atoms, 3)``. If ``None``,
        the coordinates in the input file are evaluated.
    box_vectors : pint.Quantity, optional
        An array of box vectors with units and shape ``(3,)`` defining the
        orthorhombic box side lengths (the only one currently supported in
        MiMiC).  If ``None``, the box in the input file is evaluated.
    grompp_launcher : Launcher, optional
        The launcher to use for grompp. If ``None`` a standard ``Launcher`` is
        used.
    **kwargs
        Other keyword arguments to pass to ``grompp_launcher``.

    Returns
    -------
    mdrun_cmd : GmxMdrun
        The mdrun command pointing to the appropriate .tpr file.

    """
    if positions is None:
        return mdrun_cmd

    tpr_file_name = 'gromacs.tpr'
    in_gro_file_name = 'positions.trr'

    # Mutable default arguments.
    if grompp_launcher is None:
        grompp_launcher = Launcher()

    # Create a new .gro file in the working directory to be used as input for
    # grompp using as template the structure file supplied with grompp_cmd.
    import MDAnalysis
    template_universe = MDAnalysis.Universe(grompp_cmd.structure_input_file_path)
    template_universe.atoms.positions = positions.to('angstrom').magnitude

    # MDAnalysis needs the angles of the box vectors as well. MiMiC supports
    # only orthorombic boxes.
    if box_vectors is not None:
        dimensions = np.concatenate([box_vectors.to('angstrom').magnitude, [90.0, 90, 90]])
        template_universe.dimensions = dimensions

    in_gro_file_path = os.path.join(working_dir_path, in_gro_file_name)
    with MDAnalysis.Writer(in_gro_file_path, n_atoms=template_universe.trajectory.n_atoms) as w:
        w.write(template_universe)

    # Copy the commands to avoid modifying the original one.
    mdrun_cmd = copy.deepcopy(mdrun_cmd)
    grompp_cmd = copy.deepcopy(grompp_cmd)

    # Configure and run the command.
    grompp_cmd.trajectory_input_file_path = in_gro_file_name
    grompp_cmd.tpr_output_file_path = tpr_file_name

    grompp_launcher.run(grompp_cmd, cwd=working_dir_path, **kwargs)

    # Update the mdrun command. We can use the tpr_file_name since the execution
    # will happen using working_dir_path as working directory as well.
    mdrun_cmd.tpr_input_file_path = tpr_file_name
    return mdrun_cmd


# =============================================================================
# CPMD SCRIPT INPUT/OUTPUT PARSING UTILITIES
# =============================================================================

def _parse_cpmd_input(cpmd_input_file_path):
    """Parse the input file content and return some the relevant info.

    Returns
    -------
    cpmd_file_lines : List[str]
        The content of the file as read by file.readlines().
    paths_line_idx : int
        The line index where the path to the working directory should be.
    box_vectors_line_idx : int
        The line index where the box is specified in &MIMIC.BOX.
    gromacs_to_cpmd_atom_map : Dict[int, int]
        Associates a GROMACS atom index to a CPMD atom index (as given by
        ``&MIMIC.OVERLAPS``).
    cpmd_atom_to_line_idx : Dict[int, int]
        Associates a CPMD atom index to the line index where its coordinates
        are stored.

    """
    # Read the file.
    with open(cpmd_input_file_path, 'r') as f:
        cpmd_file_lines = f.readlines()

    # Parsing results organized by blocks.
    parsed = {}

    # Identify the line with the path to the working directory.
    line_idx = 0
    while line_idx < len(cpmd_file_lines):
        line = cpmd_file_lines[line_idx].strip()

        # If the line matches a block name, the entire block is parsed.
        try:
            line_idx = _parse_cpmd_block_dispatch[line](cpmd_file_lines, line_idx+1, parsed)
        except KeyError:
            # Unknown block or empty line.
            line_idx += 1

    return (cpmd_file_lines, parsed['paths_line_idx'],
            parsed['box_vectors_line_idx'],
            parsed['gromacs_to_cpmd_atom_map'],
            parsed['cpmd_atom_to_line_idx'])


def _parse_cpmd_mimic_block(lines, line_idx, parsed):
    """Update parsed with info in the &MIMIC block.

    It adds to ``parsed`` the following keys.
    - paths_line_idx: the line index where the path to the working dir is.
    - box_vectors_line_idx: the line index where the box is defined in
      &MIMIC.BOX.
    - gromacs_to_cpmd_atom_map: a Dict[Int, Int] that associate a GROMACS
      atom index to a CPMD atom index (as given by &MIMIC.OVERLAPS).

    Parameters
    ----------
    lines : List[str]
        The lines of the entire file as read by file.readlines().
    line_idx : int
        The index of the first line after '&MIMIC'.
    parsed : Dict
        A memo dictionary where the parsed info is saved.

    Returns
    -------
    line_idx : int
        The line index right after the '&END' directive of the '&MIMIC' block.

    """
    parsed['paths_line_idx'] = None
    parsed['box_vectors_line_idx'] = None
    parsed['gromacs_to_cpmd_atom_map'] = {}

    while line_idx < len(lines):
        line = lines[line_idx].strip()

        if line.startswith('PATHS'):
            # First line after PATHS option is the number of layers. The second
            # line after PATHS is the actual path to the working directory.
            parsed['paths_line_idx'] = line_idx + 2
            line_idx += 3
        elif line.startswith('BOX'):
            parsed['box_vectors_line_idx'] = line_idx + 1
            line_idx += 2
        elif line.startswith('OVERLAPS'):
            # First line is the number of atoms.
            n_atoms = int(lines[line_idx+1])

            # Parse all OVERLAPS lines.
            line_idx += 2
            for i in range(n_atoms):
                line = lines[line_idx+i].split()

                gromacs_idx, cpmd_idx = int(line[1])-1, int(line[3])-1
                if line[0] == 1:
                    gromacs_idx, cpmd_idx = cpmd_idx, gromacs_idx

                parsed['gromacs_to_cpmd_atom_map'][gromacs_idx] = cpmd_idx

            # Update first line.
            line_idx += n_atoms
        elif line.startswith('&END'):
            break
        else:
            line_idx += 1

    return line_idx + 1


def _parse_cpmd_atoms_block(lines, line_idx, parsed):
    """Add to ``parsed`` ``cpmd_atom_to_line_idx``.

    ``cpmd_atom_to_line_idx`` is a Dict[int, int] that associates a CPMD atom
    index to the line number where its coordinates are stored.

    Parameters
    ----------
    lines : List[str]
        The lines of the entire file as read by file.readlines().
    line_idx : int
        The index of the first line after '&ATOMS'.
    parsed : Dict
        A memo dictionary where the parsed info is saved.

    Returns
    -------
    line_idx : int
        The line index right after the '&END' directive of the '&MIMIC' block.

    """
    parsed['cpmd_atom_to_line_idx'] = {}

    current_atom_idx = 0
    while line_idx < len(lines):
        line = lines[line_idx].strip()

        if line.startswith('*'):
            # New atom type. First line is nonlocality, second is number of atoms.
            n_atoms = int(lines[line_idx+2])

            # Add the atoms to the map.
            line_idx += 3
            for element_atom_idx in range(n_atoms):
                parsed['cpmd_atom_to_line_idx'][current_atom_idx] = line_idx+element_atom_idx
                current_atom_idx += 1

            line_idx += n_atoms
        elif line.startswith('&END'):
            break
        else:
            # Empty line.
            line_idx += 1

    return line_idx + 1


# Parsing dispatch function.
_parse_cpmd_block_dispatch = {
    '&MIMIC': _parse_cpmd_mimic_block,
    '&ATOMS': _parse_cpmd_atoms_block,
}


def _read_first_energy(cpmd_dir_path):
    """Read the first energy from the ENERGIES trajectory file.

    The energy is returned as a unitless float in the same units used by CPMD.
    """
    energies_traj_file_path = os.path.join(cpmd_dir_path, 'ENERGIES')
    with open(energies_traj_file_path, 'r') as f:
        for line in f:
            line = line.split()
            if int(line[0]) == 1:
                return float(line[3])


def _read_first_force(cpmd_dir_path, gromacs_to_cpmd_atom_map):
    """Read the first force from the FTRAJECTORY trajectory file.

    Parameters
    ----------
    cpmd_dir_path : str
        The directory where the FTRAJECTORY file is stored.
    gromacs_to_cpmd_atom_map : Dict[int, int], optional
        Associates a GROMACS atom index to a CPMD atom index. This must be passed
        with ``positions``.

    Returns
    -------
    force : np.ndarray
        The force as unitless numpy array of floats in the same units used by
        CPMD.
    """
    force = []

    # Read the forces in the FTRAJECTORY file, which use the CPMD atom order.
    force_traj_file_path = os.path.join(cpmd_dir_path, 'FTRAJECTORY')
    with open(force_traj_file_path, 'r') as f:
        for line in f:
            line = line.split()
            if line[0] == '1':
                # Columns 2-7 are positions and velocities.
                force.append([float(x) for x in line[7:]])

    # Convert from CPMD to GROMACS atom order.
    n_atoms = len(force)
    force = [force[gromacs_to_cpmd_atom_map.get(i, i)] for i in range(n_atoms)]

    # Convert to array.
    return np.array(force)
