#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Modules and functions to compute energies and gradients with GROMACS.

The code interfaces with the molecular dynamics software through the command line.

"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import functools
import os
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Literal, Optional, Union

import MDAnalysis.auxiliary.EDR
import MDAnalysis.coordinates.TRR
import MDAnalysis.lib.mdamath
import numpy as np
import pint
import torch

from tfep.potentials.base import PotentialBase
from tfep.utils.cli import Launcher, CLITool, FlagOption, KeyValueOption
from tfep.utils.misc import (
    flattened_to_atom, energies_array_to_tensor, forces_array_to_tensor)
from tfep.utils.parallel import ParallelizationStrategy, SerialStrategy


# =============================================================================
# GROMACS COMMANDS UTILITIES
# =============================================================================

class GmxGrompp(CLITool):
    """The grompp subprogram of gmx from the GROMACS suite.

    The executable path variable only specifies the path to the gmx executable.
    The class takes care of adding the "grompp" subprogram after "gmx" so it
    does not have to be passed when the command is instantiated.

    Relative file paths must be relative to the working directory used for
    executing the command (i.e., they are not converted to absolute paths before
    changing the working directory). This is to enable executing multiple
    instances of the command in parallel in different working directories.

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

    Relative file paths must be relative to the working directory used for
    executing the command (i.e., they are not converted to absolute paths before
    changing the working directory). This is to enable executing multiple
    instances of the command in parallel in different working directories.

    Parameters
    ----------
    executable_path : str, optional
        The executable path associated to the instance of the command. If this
        is not specified, the ``EXECUTABLE_PATH`` class variable is used instead.
    tpr_file_path : str, optional
        Path to input .tpr file. If a relative path is given, this must be relative
        to the working directory when the command is executed, not when ``GmxMdrun``
        is initialized.
    rerun_traj_file_path : str, optional
        Tells mdrun to re-compute properties (e.g., energies, forces) for the
        configurations in this trajectory file.
    traj_file_path : str, optional
        The name of the output trajectory file created by mdrun.
    edr_file_path : str, optional
        The name of the output edr file created by mdrun to log energies.
    default_file_name : str, optional
        Default file name used for all other files.
    n_ranks_pme : int, optional
        Number of separate ranks used for PME.
    n_thread_mpi_ranks : int, optional
        Number of thread-MPI ranks.
    n_omp_threads_per_mpi_rank : int, optional
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
    tpr_file_path = KeyValueOption('-s')
    rerun_traj_file_path = KeyValueOption('-rerun')
    traj_file_path = KeyValueOption('-o')
    edr_file_path = KeyValueOption('-e')
    default_file_name = KeyValueOption('-deffnm')
    n_ranks_pme = KeyValueOption('-npme')
    n_thread_mpi_ranks = KeyValueOption('-ntmpi')
    n_omp_threads_per_mpi_rank = KeyValueOption('-ntomp')


class GmxTraj(CLITool):
    """The traj subprogram of gmx from the GROMACS suite.

    The executable path variable only specifies the path to the gmx executable.
    The class takes care of adding the "traj" subprogram after "gmx" so it
    does not have to be passed when the command is instantiated.

    Relative file paths must be relative to the working directory used for
    executing the command (i.e., they are not converted to absolute paths before
    changing the working directory). This is to enable executing multiple
    instances of the command in parallel in different working directories.

    Parameters
    ----------
    executable_path : str, optional
        The executable path associated to the instance of the command. If this
        is not specified, the ``EXECUTABLE_PATH`` class variable is used instead.
    traj_file_path : str
        Path to the input .trr file.
    tpr_file_path : str
        Path to input .tpr file.
    force_xvg_file_path : str
        Path to output .xvg file holding the forces.
    full_precision : bool, optional
        Whether to save the output in full precision or always single.

    """
    EXECUTABLE_PATH = 'gmx'
    SUBPROGRAM = 'traj'
    traj_file_path = KeyValueOption('-f')
    tpr_file_path = KeyValueOption('-s')
    force_xvg_file_path = KeyValueOption('-of')
    full_precision = FlagOption('-fp', prepend_to_false='no')


# =============================================================================
# TORCH MODULE API
# =============================================================================

class GROMACSPotential(PotentialBase):
    """Potential energy and forces with GROMACS.

    This ``Module`` wraps :class:``.GROMACSPotentialEnergyFunc`` to provide a
    differentiable potential energy function for training.

    """

    #: The default energy unit.
    DEFAULT_ENERGY_UNIT : str = 'kJ/mol'

    #: The default positions unit.
    DEFAULT_POSITIONS_UNIT : str = 'nanometer'

    def __init__(
            self,
            tpr_file_path: str,
            launcher: Optional[Launcher] = None,
            positions_unit: Optional[pint.Unit] = None,
            energy_unit: Optional[pint.Unit] = None,
            precompute_gradient: bool = True,
            working_dir_path: Optional[Union[str, List[str]]] = None,
            cleanup_working_dir: bool = False,
            parallelization_strategy: Optional[ParallelizationStrategy] = None,
            launcher_kwargs: Optional[Dict[str, Any]] = None,
            mdrun_kwargs: Optional[Dict[str, Any]] = None,
            on_mdrun_error: Literal['raise', 'nan'] = 'raise',
    ):
        """Constructor.

        Parameters
        ----------
        tpr_file_path : str
            The path to the ``.tpr`` file holding the information on topology
            and the simulation parameters. The coordinates in this file are
            not important as they will be overwritten by the positions passed
            in the forward pass.
        launcher : tfep.utils.cli.Launcher, optional
            The ``Launcher`` to use to run the ``mdrun`` command used to compute
            energies and forces. If not passed, a new :class:`tfep.utils.cli.Launcher`
            is created.
        positions_unit : pint.Unit, optional
            The unit of the positions passed. This is used to appropriately convert
            ``batch_positions`` to GROMACS' units. If ``None``, no conversion is
            performed, which assumes that the input positions are in nm.
        energy_unit : pint.Unit, optional
            The unit used for the returned energies (and as a consequence forces).
            This is used to appropriately convert GROMACS energies into the desired
            units. If ``None``, no conversion is performed, which means that
            energies and forces will be in kJ/mol and kJ/mol/nm respectively.
        precompute_gradient : bool, optional
            If ``False``, the forces are not read after executing GROMACS. This
            might save a small amount of time if backpropagation is not needed.
        working_dir_path : str or List[str], optional
            The working directory to be used to run the GROMACS' commands. This must
            exist. If a list, ``batch_positions[i]`` is evaluated in the directory
            ``working_dir_path[i]``.
        cleanup_working_dir : bool, optional
            If ``True`` and ``working_dir_path`` is passed, all the files inside
            the working directory are removed after executing GROMACS. The directory(s)
            itself is not deleted.
        parallelization_strategy : tfep.utils.parallel.ParallelizationStrategy, optional
            The parallelization strategy used to distribute batches of energy and
            gradient calculations. By default, these are executed serially.
        launcher_kwargs : Dict, optional
            Other kwargs for ``launcher`` (with the exception of ``cwd`` which
            is automatically determined based on ``working_dir_path``).
        mdrun_kwargs : Dict, optional
            Other kwargs for ``GmxMdrun``.
        on_mdrun_error : Literal['raise', 'nan'], optional
            Whether to raise an exception or return NaN potential when the single-
            point energy calculation with mdrun fails. In the latter case, the
            returned forces are set to zero.

        See Also
        --------
        :class:`.GROMACSPotentialEnergyFunc`
            More details on input parameters and implementation details.

        """
        super().__init__(positions_unit=positions_unit, energy_unit=energy_unit)

        self.tpr_file_path = tpr_file_path
        self.launcher = launcher
        self.precompute_gradient = precompute_gradient
        self.working_dir_path = working_dir_path
        self.cleanup_working_dir = cleanup_working_dir
        self.parallelization_strategy = parallelization_strategy
        self.launcher_kwargs = launcher_kwargs
        self.mdrun_kwargs = mdrun_kwargs
        self.on_mdrun_error = on_mdrun_error

    def forward(self, batch_positions: torch.Tensor, batch_cell: torch.Tensor) -> torch.Tensor:
        """Compute a differential potential energy for a batch of configurations.

        Parameters
        ----------
        batch_positions : torch.Tensor
            A tensor of positions in flattened format (i.e., with shape
            ``(batch_size, 3*n_atoms)``) in units of ``self.positions_unit``.
        batch_cell : torch.Tensor
            Shape ``(batch_size, 6)``. Unitcell dimensions. For each data point,
            the first 3 elements represent the vector lengths in units of
            ``self.positions_unit`` and the last 3 their respective angles (in
            degrees).

        Returns
        -------
        potential_energy : torch.Tensor
            ``potential_energy[i]`` is the potential energy of configuration
            ``batch_positions[i]`` and ``batch_cell[i]`` in units of
            ``self.energy_unit`` (or GROMACS units if ``energy_unit`` is not
            provided).

        """
        return gromacs_potential_energy(
            batch_positions=batch_positions,
            batch_cell=batch_cell,
            tpr_file_path=self.tpr_file_path,
            launcher=self.launcher,
            positions_unit=self._positions_unit,
            energy_unit=self._energy_unit,
            precompute_gradient=self.precompute_gradient,
            working_dir_path=self.working_dir_path,
            cleanup_working_dir=self.cleanup_working_dir,
            parallelization_strategy=self.parallelization_strategy,
            launcher_kwargs=self.launcher_kwargs,
            mdrun_kwargs=self.mdrun_kwargs,
            on_mdrun_error=self.on_mdrun_error,
        )


# =============================================================================
# TORCH FUNCTIONAL API
# =============================================================================

class GROMACSPotentialEnergyFunc(torch.autograd.Function):
    """PyTorch-differentiable QM/MM potential energy function wrapped around GROMACS.

    The function calls GROMACS through the command line interface using user-prepared
    GROMACS input files and reads the resulting energies and forces.

    The function supports running each GROMACS execution in a separate working
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
        input files, not the one used internally by CPMD.
    batch_cell : torch.Tensor, optional
        Shape ``(batch_size, 6)``. Unitcell dimensions. For each data point,
        the first 3 elements represent the vector lengths in units of
        ``self.positions_unit`` and the last 3 their respective angles (in
        degrees).
    tpr_file_path : str
        The path to the ``.tpr`` file holding the information on topology and
        the simulation parameters. The coordinates in this file are not important
        as they will be overwritten by the positions passed in the forward pass.
    launcher : tfep.utils.cli.Launcher, optional
        The ``Launcher`` to use to run the ``cpmd_cmd`` and ``mdrun_cmd``. If
        not passed, a new :class:`tfep.utils.cli.Launcher` is created.
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
    mdrun_kwargs : Dict, optional
        Other kwargs for ``GmxMdrun``.
    on_mdrun_error : Literal['raise', 'nan'], optional
        Whether to raise an exception or return NaN potential when the single-
        point energy calculation with mdrun fails. In the latter case, the returned
        forces are set to zero.

    Returns
    -------
    potentials : torch.Tensor
        ``potentials[i]`` is the potential energy of configuration
        ``batch_positions[i]``.

    See Also
    --------
    :class:`.MiMiCPotential`
        ``Module`` API for computing potential energies with MiMiC.

    """

    @staticmethod
    def forward(
            ctx: torch.autograd.function._ContextMethodMixin,
            batch_positions: torch.Tensor,
            batch_cell: torch.Tensor,
            tpr_file_path: str,
            launcher: Optional[Launcher] = None,
            positions_unit: Optional[pint.Unit] = None,
            energy_unit: Optional[pint.Unit] = None,
            precompute_gradient: bool = True,
            working_dir_path: Optional[Union[str, List[str]]]=None,
            cleanup_working_dir: bool = False,
            parallelization_strategy: Optional[ParallelizationStrategy] = None,
            launcher_kwargs: Optional[Dict[str, Any]] = None,
            mdrun_kwargs: Optional[Dict[str, Any]] = None,
            on_mdrun_error: Literal['raise', 'nan'] = 'raise',
    ):
        """Compute the potential energy of the molecule with GROMACS."""
        # Mutable default arguments.
        if parallelization_strategy is None:
            parallelization_strategy = SerialStrategy()

        # Convert flattened positions tensor to numpy array of shape (batch_size, n_atoms, 3).
        batch_positions_nm = flattened_to_atom(batch_positions.detach().cpu().numpy())

        # GROMACS supports triclinic boxes in vector format.
        if batch_cell is None:
            batch_box_vectors_nm = None
        else:
            batch_box_vectors_nm = batch_cell.detach().cpu().numpy()
            batch_box_vectors_nm = np.apply_along_axis(MDAnalysis.lib.mdamath.triclinic_vectors,
                                                       axis=1, arr=batch_box_vectors_nm)

        # Both positions and box vectors must be in nanometers.
        if positions_unit is not None:
            batch_positions_nm = _to_gromacs_units(batch_positions_nm, positions_unit)
            batch_box_vectors_nm = _to_gromacs_units(batch_box_vectors_nm, positions_unit)

        # Make sure working_dir_path and launcher are in batch format.
        batch_size = batch_positions_nm.shape[0]
        if working_dir_path is None or isinstance(working_dir_path, str):
            working_dir_path = [working_dir_path] * batch_size
        else:
            working_dir_path = [os.path.realpath(p) for p in working_dir_path]
        try:
            iter(launcher)
        except TypeError:
            launcher = [launcher] * batch_size

        # Run the command.
        task = functools.partial(_run_gromacs_task, tpr_file_path, precompute_gradient,
                                 cleanup_working_dir, launcher_kwargs, mdrun_kwargs, on_mdrun_error)
        distributed_args = zip(batch_positions_nm, batch_box_vectors_nm, launcher, working_dir_path)
        result = parallelization_strategy.run(task, distributed_args)

        # Set units for unit conversion.
        if positions_unit is not None:
            units = positions_unit._REGISTRY
        elif energy_unit is not None:
            units = energy_unit._REGISTRY
        else:
            units = pint.UnitRegistry()
        default_energy_unit = GROMACSPotential.default_energy_unit(units)
        default_forces_unit = default_energy_unit / GROMACSPotential.default_positions_unit(units)

        # Convert to unitless Tensor before returning.
        if not precompute_gradient:
            energies = np.array(result)
        else:
            # Convert from a list of shape (batch_size, 2) to arrays (2, batch_size).
            energies, forces = [np.array(res) for res in zip(*result)]
            # Convert the forces to a flattened tensor before storing it in ctx
            # to compute the gradient during backpropagation.
            forces = forces_array_to_tensor(forces*default_forces_unit, positions_unit, energy_unit)
            ctx.save_for_backward(forces.to(batch_positions))
        energies = energies_array_to_tensor(energies*default_energy_unit, energy_unit)
        return energies.to(batch_positions)

    @staticmethod
    def backward(ctx, grad_output):
        """Compute the gradient of the potential energy."""
        # We still need to return a None gradient for each
        # input of forward() beside batch_positions.
        n_input_args = 13
        grad_input = [None for _ in range(n_input_args)]

        # Compute gradient w.r.t. batch_positions.
        if ctx.needs_input_grad[0]:
            # Retrieve pre-computed forces.
            if len(ctx.saved_tensors) == 1:
                forces, = ctx.saved_tensors
            else:
                raise ValueError('Cannot compute gradients if precompute_gradient '
                                 'option is set to False.')

            # Accumulate gradient, which has opposite sign of the forces.
            grad_input[0] = -forces * grad_output[:, None]

        return tuple(grad_input)


def gromacs_potential_energy(
        batch_positions: torch.Tensor,
        batch_cell: torch.Tensor,
        tpr_file_path: str,
        launcher: Optional[Launcher] = None,
        positions_unit: Optional[pint.Unit] = None,
        energy_unit: Optional[pint.Unit] = None,
        precompute_gradient: bool = True,
        working_dir_path: Optional[Union[str, List[str]]] = None,
        cleanup_working_dir: bool = False,
        parallelization_strategy: Optional[ParallelizationStrategy] = None,
        launcher_kwargs: Optional[Dict[str, Any]] = None,
        mdrun_kwargs: Optional[Dict[str, Any]] = None,
        on_mdrun_error: Literal['raise', 'nan'] = 'raise',
):
    """PyTorch-differentiable QM/MM potential energy using GROMACS.

    PyTorch ``Function``s do not accept keyword arguments. This function wraps
    :func:`.GROMACSPotentialEnergyFunc.apply` to enable standard functional notation.
    See the documentation on the original function for the input parameters.

    See Also
    --------
    :class:`.GROMACSPotentialEnergyFunc`
        More details on input parameters and implementation details.

    """
    # apply() does not accept keyword arguments.
    return GROMACSPotentialEnergyFunc.apply(
        batch_positions,
        batch_cell,
        tpr_file_path,
        launcher,
        positions_unit,
        energy_unit,
        precompute_gradient,
        working_dir_path,
        cleanup_working_dir,
        parallelization_strategy,
        launcher_kwargs,
        mdrun_kwargs,
        on_mdrun_error,
    )


# =============================================================================
# MAIN FUNCTIONS WRAPPING GROMACS
# =============================================================================

def _to_gromacs_units(x, positions_unit):
    """Convert x from positions_unit to nanometers."""
    default_positions_unit = GROMACSPotential.default_positions_unit(positions_unit._REGISTRY)
    return (x * positions_unit).to(default_positions_unit).magnitude


def _run_gromacs_task(
        tpr_file_path,
        return_forces,
        cleanup_working_dir,
        launcher_kwargs,
        mdrun_kwargs,
        on_mdrun_error,
        positions_nm,
        box_vectors_nm,
        launcher,
        working_dir_path,
):
    """This is the task passed to the ``ParallelizationStrategy`` to run MiMiC.

    Parameters
    ----------
    tpr_file_path : str
        The path to the ``.tpr`` file holding the information on topology and
        the simulation parameters. The coordinates in this file are not important
        as they will be overwritten by ``positions_nm``.
    return_forces : bool, optional
        If ``True``, the forces are returned.
    cleanup_working_dir : bool, optional
        If ``True`` and ``working_dir_path`` is passed, all the files inside the
        working directory are removed after executing GROMACS. The directory(s)
        itself is not deleted.
    launcher_kwargs : Dict, optional
        Other kwargs for ``launcher`` (with the exception of ``cwd`` which is
        automatically determined based on ``working_dir_path``).
    mdrun_kwargs : Dict, optional
        Other kwargs for ``GmxMdrun``.
    on_mdrun_error : Literal['raise', 'nan'], optional
        Whether to raise an exception or return NaN potential when the single-
        point energy calculation with mdrun fails. In the latter case, the returned
        forces are set to zero.
    positions_nm : np.ndarray
        Shape ``(n_atoms, 3)``. The positions in nanometers.
    box_vectors_nm : np.ndarray or None, optional
        Shape ``(3, 3)``. Box vectors in nanomters.
    launcher : tfep.utils.cli.Launcher
        The ``Launcher`` to use to run ``mdrun``. If ``None``, a new instance of
        :class:`tfep.utils.cli.Launcher` is used.
    working_dir_path : str, optional
        The working directory to be used to run GROMACS. This must exist.

    Returns
    -------
    energy_kJ_mol : np.ndarray, optional
        ``energy_kJ_mol`` is the potential energy of the configuration in kJ/mol.
    forces_kJ_mol_nm : np.ndarray, optional
        Shape ``(n_atoms, 3)``. ``forces_kJ_mol_nm[i]`` are the forces of the configuration
        in kJ/mol/nm. This is returned only if ``return_forces`` is ``True``.

    """
    # Mutable default arguments.
    if launcher is None:
        launcher = Launcher()
    if launcher_kwargs is None:
        launcher_kwargs = {}
    if mdrun_kwargs is None:
        mdrun_kwargs = {}

    try:
        # Create a temporary working directory if not given.
        if working_dir_path is None:
            tmp_dir = tempfile.TemporaryDirectory()
            working_dir_path = tmp_dir.name
        else:
            tmp_dir = None
        working_dir_path = os.path.realpath(working_dir_path)

        # Create a coordinate file to evaluate.
        g96_file_path = _create_g96_file(working_dir_path, positions_nm, box_vectors_nm)

        # Run the mdrun command.
        edr_file_path = os.path.join(working_dir_path, 'energy.edr')
        traj_file_path = os.path.join(working_dir_path, 'traj.trr')
        mdrun_cmd = GmxMdrun(
            tpr_file_path=tpr_file_path,  # input
            rerun_traj_file_path=g96_file_path,  # input
            traj_file_path=traj_file_path,  # output
            edr_file_path=edr_file_path,  # output
            **mdrun_kwargs,
        )

        # Single-point calculation.
        completed_process = launcher.run(mdrun_cmd, cwd=working_dir_path, **launcher_kwargs)

        # Handle errors.
        if completed_process.returncode != 0:
            if on_mdrun_error == 'raise':
                raise RuntimeError('Single-point energy with mdrun returned non-zero exit code.')

            # Return NaN.
            assert on_mdrun_error == 'nan'
            energy_kJ_mol = np.nan
            if return_forces:
                forces_kJ_mol_nm = np.zeros_like(positions_nm)
        else:
            # Read energies and forces.
            energy_kJ_mol = _read_energy(edr_file_path)
            if return_forces:
                forces_kJ_mol_nm = _read_forces(traj_file_path, tpr_file_path, working_dir_path)
    finally:
        if tmp_dir is None and cleanup_working_dir:
            # Clean up user-given working directory.
            for file_name in os.listdir(working_dir_path):
                file_path = os.path.join(working_dir_path, file_name)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
        elif tmp_dir is not None:
            tmp_dir.cleanup()

    if return_forces:
        return energy_kJ_mol, forces_kJ_mol_nm
    return energy_kJ_mol


def _create_g96_file(dir_path, positions_nm, box_vectors_nm):
    """Save a g96 file called configuration.g96 in the given directory holding positions and box vectors.

    Example file (the BOX section is optional)

    TITLE
    Ligand in water
    END
    POSITIONRED
        1.470000029    3.907000065    0.718999982
        1.463000059    4.256000042    1.024999976
        1.544000030    4.171000004    0.797999978
    END
    BOX
        4.383590221    3.099669933    2.821249962    0.000000000    0.000000000    3.099669933    0.000000000    3.099669933    1.283920050
    END

    Parameters
    ----------
    dir_path : str
        The path where to save the g96 file.
    positions_nm : np.ndarray, optional
        Shape ``(n_atoms, 3)``.
    box_vectors_nm : np.ndarray or None
        Shape ``(3, 3)`` defining the box shape and dimension.

    Returns
    -------
    g96_file_path : str
        The path to the created g96 file.

    """
    g96_file_path = os.path.realpath(os.path.join(dir_path, 'configuration.g96'))
    with open(g96_file_path, 'w') as f:
        f.write('TITLE\ntfep\nEND\nPOSITIONRED\n')
        np.savetxt(f, positions_nm, fmt='%15.9f', delimiter='')
        f.write('END\n')

        # Box is optional.
        if box_vectors_nm is not None:
            f.write('BOX\n')
            # Format is v1(x) v2(y) v3(z) v1(y) v1(z) v2(x) v2(z) v3(x) v3(y) (see https://manual.gromacs.org/current/reference-manual/file-formats.html#gro)
            box_vectors_nm = box_vectors_nm.reshape(-1, 9)[:, [0, 4, 8, 1, 2, 3, 5, 6, 7]]
            np.savetxt(f, box_vectors_nm, fmt='%15.9f', delimiter='')
            f.write('END\n')

    return g96_file_path


def _read_energy(edr_file_path):
    """Extract the potential energies from a binary edr file."""
    # Do not convert the units so that we return in native GROMACS units (rather than MDAnalysis).
    reader = MDAnalysis.auxiliary.EDR.EDRReader(edr_file_path, convert_units=False)
    potential = reader.get_data('Potential')['Potential']
    assert potential.shape == (1,)
    return potential[0]


def _read_forces(traj_file_path, tpr_file_path, working_dir_path):
    """Extract the forces from a binary trajectory file."""
    # # Do not convert the units so that we return in native GROMACS units (rather than MDAnalysis).
    # # TRRReader returns forces always in single precision, regardless of how they are saved.
    # with MDAnalysis.coordinates.TRR.TRRReader(traj_file_path, convert_units=False) as reader:
    #     assert len(reader) == 1
    #     forces = reader.ts.forces
    # return forces

    # Extract forces in double precision.
    xvg_file_path = os.path.join(working_dir_path, 'forces.xvg')
    gmx_traj = GmxTraj(
        traj_file_path=traj_file_path,
        tpr_file_path=tpr_file_path,
        force_xvg_file_path=xvg_file_path,
        full_precision=True,
    )

    # echo "System" | gmx traj -f traj.trr -s gromacs.tpr -fp -of forces.xvg
    echo_cmd = ['echo', 'System']
    gmx_traj_cmd = gmx_traj.to_subprocess()
    with subprocess.Popen(echo_cmd, stdout=subprocess.PIPE) as p1:
        with subprocess.Popen(gmx_traj_cmd, stdin=p1.stdout) as p2:
            p2.communicate()

    # Read the resulting xvg file. The first column is always the time.
    forces = flattened_to_atom(np.loadtxt(xvg_file_path, comments=['#', '@'])[1:])
    return forces
