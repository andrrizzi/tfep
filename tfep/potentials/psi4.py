#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Modules and functions to compute QM energies and gradients with Psi4.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

# DO NOT IMPORT PSI4 HERE! Psi4 is an optional dependency of tfep.
import collections
import functools
import warnings

import numpy as np
import pint
import torch

from tfep.potentials.base import PotentialBase
from tfep.utils.misc import flattened_to_atom, energies_array_to_tensor, forces_array_to_tensor
from tfep.utils.parallel import SerialStrategy


# =============================================================================
# UTILITIES
# =============================================================================

def create_psi4_molecule(positions, fix_com=True, fix_orientation=True, **kwargs):
    """Create a Psi4 molecule object.

    This is a wrapper around ``psi4.core.Molecule.from_arrays`` that handles the
    units of ``positions`` when creating a ``psi4.core.Molecule`` object.

    The returned molecule is not activated in Psi4. It is activated when passed
    to Psi4's specific functions or by calling ``psi4.core.set_active_molecule``.

    .. note::
        Note that the defaults for ``fix_com`` and ``fix_orientation`` are different
        in this function than in ``psi4.core.Molcule.from_arrays``. By default,
        Psi4 removes translational/rotational degrees of freedom for efficiency
        reason. This does not affect energies, but it forces some of the force
        components to 0 and the final positions of the ``Molecule``, which might
        be unexpected behavior.

    .. note::
        Currently, if the molecule has a net charge, this must be passed through
        the ``molecular_charge`` argument.

    Parameters
    ----------
    positions : pint.Quantity
        The coordinates of the molecules as an array of shape ``(n_atoms, 3)``.
    **kwargs
        Other keyword arguments to pass to ``psi4.core.Molecule.from_arrays``
        except for ``geom`` and ``units`` which are handled by this method.
        Note that one between ``elem``, ``elez``, or ``elbl`` is mandatory.

    Returns
    -------
    psi4_molecule : psi4.core.Molecule
        A Psi4 Molecule object.

    See Also
    --------
    ``psi4.core.set_active_molecule`` `documentation <https://psicode.org/psi4manual/master/external_apis.html#qcelemental.molparse.from_arrays>`_:
        for more information on the supported parameters.

    """
    import psi4
    return psi4.core.Molecule.from_arrays(
        geom=positions.magnitude,
        units=str(positions.units),
        fix_com=fix_com,
        fix_orientation=fix_orientation,
        **kwargs
    )


def configure_psi4(
        memory=None,
        n_threads=None,
        psi4_output_file_path=None,
        psi4_scratch_dir_path=None,
        active_molecule=None,
        global_options=None,
):
    """Helper function to set common configurations of Psi4.

    Parameters
    ----------
    memory : str, optional
        Set the total memory available to psi4 (e.g., ``'1 KiB'`` or ``'1 KB'``).
        Note that this is the memory per job, not per thread. This sets a limit
        on the memory used by the major data structures in psi4, but the actual
        memory consuption is slightly higher.
    n_threads : int, optional
        Number of MP threads available to psi4.
    psi4_output_file_path : str, optional
        Redirect stdout to this file. If the string "quiet" is passed, the output
        is suppressed.
    psi4_scratch_dir_path : str, optional
        Path to the scratch directory. It is recommended that this directory
        allows fast reading/writing operations.
    active_molecule : psi4.core.Molecule, optional
        If given, the active molecule is set to this.
    global_options : dict
        General global options.

    """
    import psi4

    if memory is not None:
        psi4.set_memory(memory)
    if n_threads is not None:
        psi4.core.set_num_threads(n_threads)

    # Output file.
    if psi4_output_file_path == 'quiet':
        psi4.core.be_quiet()
    elif psi4_output_file_path is not None:
        psi4.core.set_output_file(psi4_output_file_path)

    # Scratch dir.
    if psi4_scratch_dir_path is not None:
        psi4_io = psi4.core.IOManager.shared_object()
        psi4_io.set_default_path(psi4_scratch_dir_path)

    # Active molecule.
    if active_molecule is not None:
        psi4.core.set_active_molecule(active_molecule)

    # Global options.
    if global_options is not None:
        psi4.set_options(global_options)


# =============================================================================
# TORCH MODULE API
# =============================================================================

class Psi4Potential(PotentialBase):
    """Potential energy and forces with Psi4.

    This ``Module`` wraps :class:``.Psi4PotentialEnergyFunc`` to provide a
    differentiable potential energy function for training. It also provides an
    API to compute energies and forces with Psi4 from batches of coordinates in
    ``numpy`` arrays in standard format (i.e., shape ``(n_atoms, 3)``) rather
    than flattened ``torch.Tensor``s (i.e., shape ``(n_atoms*3,)``).

    See Also
    --------
    :class:`.Psi4PotentialEnergyFunc`
        More details on input parameters and implementation details.
    ``psi4.energy`` `documentation <https://psicode.org/psi4manual/master/api/psi4.driver.energy.html>`_:
        More information on the supported keyword arguments.
    ``psi4.gradient`` `documentation <https://psicode.org/psi4manual/master/api/psi4.driver.gradient.html>`_:
        More information on the supported keyword arguments.

    """

    #: The default energy unit.
    DEFAULT_ENERGY_UNIT : str = 'hartree'

    #: The default positions unit.
    DEFAULT_POSITIONS_UNIT : str = 'bohr'

    def __init__(
            self,
            name,
            molecule=None,
            positions_unit=None,
            energy_unit=None,
            precompute_gradient=True,
            parallelization_strategy=None,
            on_unconverged='raise',
            **kwargs
    ):
        """Constructor

        Parameters
        ----------
        name : str
            The name of the potential to pass to ``psi4.energy()``.
        molecule : psi4.core.Molecule, optional
            If not ``None``, this will be set as the currently activated molecule
            in Psi4. Note that the old active molecule is not restored at the end
            of the execution.
        positions_unit : pint.Unit, optional
            The unit of the positions passed to the class methods. Since ``Tensor``s
            and positions returned by MDAnalysis normally do not have ``pint``
            units attached, this is used to appropriately convert ``batch_positions``
            to Psi4 units. If ``None``, no conversion is performed, which assumes
            that the input positions are in the same units used by Psi4.
        energy_unit : pint.Unit, optional
            The unit used for the returned energies (and as a consequence forces).
            Since ``Tensor``s and positions returned by MDAnalysis normally do
            not have ``pint`` units attached, this is used to appropriately convert
            Psi4 energies into the desired units. If ``None``, no conversion is
            performed, which means that energies and forces will be returned in
            Psi4 units.
        precompute_gradient : bool, optional
            If ``True``, the gradient is computed in the forward pass and saved
            to be consumed during backward.
        parallelization_strategy : tfep.utils.parallel.ParallelizationStrategy, optional
            The parallelization strategy used to distribute batches of energy and
            gradient calculations. By default, these are executed serially using
            the thread-based parallelization native in psi4.
        on_unconverged : str, optional
            Specifies how to handle the case in which the calculation did not
            converge. It can have the following values:
            - ``'raise'``: Raise the Psi4 exception.
            - ``'nan'``: Return ``float('nan')`` energy and zero forces.
            To treat the calculation as converged and return the latest energy,
            force, and/or wavefunction, simply set the psi4 global option
            ``'fail_on_maxiter'``.
        **kwargs
            Other keyword arguments to pass to :class:``.Psi4PotentialEnergyFunc``,
            ``psi4.energy``, and ``psi4.gradient``.

        """
        super().__init__(positions_unit=positions_unit, energy_unit=energy_unit)

        # Handle mutable default arguments.
        if parallelization_strategy is None:
            parallelization_strategy = SerialStrategy()

        self.name = name
        self.molecule = molecule
        self.precompute_gradient = precompute_gradient
        self.parallelization_strategy = parallelization_strategy
        self.on_unconverged = on_unconverged
        self.kwargs = kwargs

    def forward(self, batch_positions):
        """Compute a differential potential energy for a batch of configurations.

        Parameters
        ----------
        batch_positions : torch.Tensor
            A tensor of positions in flattened format (i.e., with shape
            ``(batch_size, 3*n_atoms)``) in units of ``self.positions_unit``
            (or Psi4 units if ``positions_unit`` is not provided).

        Returns
        -------
        potential_energy : torch.Tensor
            ``potential_energy[i]`` is the potential energy of configuration
            ``batch_positions[i]`` in units of ``self.energy_unit``.

        """
        return psi4_potential_energy(
            batch_positions=batch_positions,
            name=self.name,
            molecule=self.molecule,
            positions_unit=self._positions_unit,
            energy_unit=self._energy_unit,
            precompute_gradient=self.precompute_gradient,
            parallelization_strategy=self.parallelization_strategy,
            on_unconverged=self.on_unconverged,
            **self.kwargs
        )

    def energy(self, batch_positions):
        """Compute a the potential energy of a batch of configurations.

        Parameters
        ----------
        batch_positions : numpy.ndarray or pint.Quantity
            A batch of configurations in standard format (i.e., with shape
            ``(batch_size, n_atoms, 3)`` or ``(n_atoms, 3)``). If no units are
            attached to the array, it is assumed the positions are is in
            ``self.positions_unit`` units (or Psi4 units if ``positions_unit`` is
            not provided).

        Returns
        -------
        potential_energies : pint.Quantity
            ``potential_energies[i]`` is the potential energy of configuration
            ``batch_positions[i]``.

        """
        # Add units.
        batch_positions = self._ensure_positions_has_units(batch_positions)
        return _run_psi4(
            name=self.name,
            batch_positions=batch_positions,
            molecule=self.molecule,
            return_energy=True,
            parallelization_strategy=self.parallelization_strategy,
            on_unconverged=self.on_unconverged,
            **self.kwargs
        )

    def force(self, batch_positions):
        """Compute the force for a batch of configurations.

        Parameters
        ----------
        batch_positions : numpy.ndarray or pint.Quantity
            A batch of configurations in standard format (i.e., with shape
            ``(batch_size, n_atoms, 3)`` or ``(n_atoms, 3)``). If no units are
            attached to the array, it is assumed the positions are is in
            ``self.positions_unit`` units (or Psi4 units if ``positions_unit`` is
            not provided).

        Returns
        -------
        forces : pint.Quantity
            ``forces[i]`` is the force of configuration ``batch_positions[i]``.

        """
        # Add units.
        batch_positions = self._ensure_positions_has_units(batch_positions)
        return _run_psi4(
            name=self.name,
            batch_positions=batch_positions,
            molecule=self.molecule,
            return_force=True,
            parallelization_strategy=self.parallelization_strategy,
            on_unconverged=self.on_unconverged,
            **self.kwargs
        )

    def _ensure_positions_has_units(self, batch_positions):
        """Add units to an array of positions."""
        try:
            batch_positions.units
        except AttributeError:
            return batch_positions * self.positions_unit
        return batch_positions


# =============================================================================
# TORCH FUNCTIONAL API
# =============================================================================

class Psi4PotentialEnergyFunc(torch.autograd.Function):
    """PyTorch-differentiable potential energy of a Psi4 molecule.

    This is essentially a wrapper of ``psi4.energy``, but it provides additional
    functionalities:
    - Handle batches of coordinate tensors of shape ``(batch_size, 3*n_atoms)``.
    - Provides sample-specific restart capabilities for samples within the batch.
    - Implement the ``torch.autograd.Function`` interface to enable the calculation
      of the potential energy gradients used for backpropagation through
      ``psi4.gradient``.

    For efficiency reasons, by default the function computes and cache the gradient
    (i.e., the forces) during the forward pass so that it can be used during
    backpropagation. This gives better performance overall when backpropagation
    is necessary as the wavefunction is converged just once. Even when a restart
    file is provided, the restart wavefunction correspond only to that of the
    Hartree-Fock SCF procedure so if another potential is used (e.g., MP2), it
    requires another self-consistent calculation. If backpropagation is not
    necessary, set ``precompute_gradient`` to ``False``.

    Double backpropagation (sometimes necessary, for example, to train on forces)
    is supported by estimating the vector-Hessian product with finite-differences [1].

    By default, the perform the batch of energy/gradient calculations serially,
    using the native thread parallelization implemented in Psi4. This scheme is,
    however, not embarassingly parallel. Thus, the module supports batch
    parallelization schemes through :class:``tfep.utils.parallel.ParallelizationStrategy``s
    Note that, because a psi4 ``Molecule`` is not picklable, it cannot be sent
    to multiple processes for the purpose of batch parallelization (e.g., using
    :class:``~tfep.utils.parallel.ProcessPoolStrategy``). A work-around is to
    use an ``initializer`` for the ``multiprocessing.Pool`` that creates and
    activates the molecule in each subprocess (see example below).

    Parameters
    ----------
    ctx : torch.autograd.function._ContextMethodMixin
        A context to save information for the gradient.
    batch_positions : torch.Tensor
        A tensor of positions in flattened format (i.e., with shape
        ``(batch_size, 3*n_atoms)``).
    name : str
        The name of the potential to pass to ``psi4.energy()``.
    molecule : psi4.core.Molecule, optional
        If not ``None``, this will be set as the currently activated molecule
        in Psi4. Note that the old active molecule is not restored at the end
        of the execution.
    positions_unit : pint.Unit, optional
        The unit of the positions passed. This is used to appropriately convert
        ``batch_positions`` to Psi4 units. If ``None``, no conversion is performed,
        which assumes that the input positions are in the same units used by
        Psi4.
    energy_unit : pint.Unit, optional
        The unit used for the returned energies (and as a consequence forces).
        This is used to appropriately convert Psi4 energies into the desired
        units. If ``None``, no conversion is performed, which means that energies
        and forces will use Psi4 units.
    write_orbitals : bool or str or List[str], optional
        This option is passed to ``psi4.energy`` to store the wavefunction on
        disk at each Hartree-Fock SCF iteration, which can later be used to
        restart the calculation. If a ``list``, it must specify a path to the
        path to the restart file to write for each batch sample.
    restart_file : str or List[str], optional
        A Psi4 restart file path (or a list of restart file paths, one for each
        batch sample) storing a wavefunction that can be used as a starting point
        for the Hartree-Fock SCF optimization.
    precompute_gradient : bool, optional
        If ``True``, the gradient is computed in the forward pass and saved to
        be consumed during backward.
    parallelization_strategy : tfep.utils.parallel.ParallelizationStrategy, optional
        The parallelization strategy used to distribute batches of energy and
        gradient calculations. By default, these are executed serially using
        the thread-based parallelization native in psi4.
    on_unconverged : str, optional
        Specifies how to handle the case in which the calculation did not converge.
        It can have the following values:
        - ``'raise'``: Raise the Psi4 exception.
        - ``'nan'``: Return ``float('nan')`` energy and zero forces.
        To treat the calculation as converged and return the latest energy, force,
        and/or wavefunction, simply set the psi4 global option ``'fail_on_maxiter'``.
    kwargs : dict, optional
        Other keyword arguments to pass to ``psi4.energy`` and ``psi4.gradient``.

    Returns
    -------
    potentials : torch.Tensor
        ``potentials[i]`` is the potential energy of configuration
        ``batch_positions[i]``.

    See Also
    --------
    :class:`.Psi4Potential`
        ``Module`` API for computing potential energies with Psi4.
    ``psi4.energy`` `documentation <https://psicode.org/psi4manual/master/api/psi4.driver.energy.html>`_:
        More information on the supported keyword arguments.
    ``psi4.gradient`` `documentation <https://psicode.org/psi4manual/master/api/psi4.driver.gradient.html>`_:
        More information on the supported keyword arguments.

    Examples
    --------

    The example sets up a parallelization strategy based on a pool of processes
    for the calculation of energies and gradients of a water molecule. Note that
    molecules cannot be sent between processes with ``pickle`` so it is convenient
    to create the molecule and activate it in the process through an ``initializer``.
    Note that the functional syntax ``psi4_potential_energy()`` is used rather
    than ``Psi4PotentialEnergyFunc.apply()``, which do not support keyword
    arguments.

    .. code-block:: python

       import numpy as np
       import pint
       from torch.multiprocessing import Pool
       from tfep.utils.parallel import ProcessPoolStrategy

       def pool_process_initializer(positions):
           # Create a water molecule.
           molecule = create_psi4_molecule(positions=positions, activate=True, elem=['O', 'H', 'H'])

           # Create a scratch directory.
           scratch_dir_path = os.path.join('tmp/', str(os.getpid()))
           os.makedirs(scratch_dir_path, exist_ok=True)

           # Configure psi4 and activate the molecule.
           configure_psi4(
               n_threads=1,
               psi4_output_file_path='quiet',
               psi4_scratch_dir_path=scratch_dir_path,
               active_molecule=molecule,
               global_options=dict(basis='cc-pvtz', reference='RHF'),
           )

       # A batch of size 2 with identical positions.
       ureg = pint.UnitRegistry()
       positions = [
           [-0.2950, -0.2180, 0.1540],
           [-0.0170, 0.6750, 0.4080],
           [0.3120, -0.4570, -0.5630],
       ]
       batch_positions =  np.array([positions, positions], dtype=np.double) * ureg.angstrom

       with Pool(2, pool_process_initializer, initargs=[batch_positions[0]]) as p:
           strategy = ProcessPoolStrategy(p)
           energy = psi4_potential_energy(batch_positions, name='scf', positions_unit=ureg.angstrom)

    References
    ----------
    [1] Putrino A, Sebastiani D, Parrinello M. Generalized variational density
        functional perturbation theory. The Journal of Chemical Physics. 2000
        Nov 1;113(17):7102-9.

    """

    @staticmethod
    def forward(
            ctx,
            batch_positions,
            name,
            molecule=None,
            positions_unit=None,
            energy_unit=None,
            write_orbitals=False,
            restart_file=None,
            precompute_gradient=True,
            parallelization_strategy=None,
            on_unconverged='raise',
            kwargs=None,
    ):
        """Compute the potential energy of the molecule with Psi4."""
        # Handle mutable default arguments.
        if kwargs is None:
            kwargs = {}
        if parallelization_strategy is None:
            parallelization_strategy = SerialStrategy()

        # Check for unit registry.
        if positions_unit is not None:
            unit_registry = positions_unit._REGISTRY
        elif energy_unit is not None:
            unit_registry = energy_unit._REGISTRY
        else:
            unit_registry = pint.UnitRegistry()

        # Convert tensor to numpy array with shape (batch_size, n_atoms, 3) with attached units.
        batch_positions_arr = flattened_to_atom(batch_positions.detach().cpu().numpy())
        if positions_unit is None:
            batch_positions_arr *= Psi4Potential.default_positions_unit(unit_registry)
        else:
            batch_positions_arr *= positions_unit

        # Determine kwargs to pass to _run_psi4.
        run_psi4_kwargs = dict(
            name=name,
            batch_positions=batch_positions_arr,
            molecule=molecule,
            return_energy=True,
            write_orbitals=write_orbitals,
            restart_file=restart_file,
            unit_registry=unit_registry,
            parallelization_strategy=parallelization_strategy,
            on_unconverged=on_unconverged,
            **kwargs
        )

        if precompute_gradient:
            # The gradient computation already computes the potentials so
            # we do everything in a single pass and avoid re-doing the
            # MP2 wavefunction convergence twice.
            energies, forces = _run_psi4(return_force=True, **run_psi4_kwargs)

            # Save the pre-computed forces used for backpropagation.
            forces = forces_array_to_tensor(forces, positions_unit, energy_unit).to(batch_positions)

            # In this case we won't need the SCF wavefunctions.
            ctx.wavefunctions = None

            # The original input vector is required in case of double backprop
            # to tell PyTorch that _Psi4PotentialEnergyFuncBackward enters the
            # computational graph correctly. It won't be actually used since we
            # are also passing batch_positions_arr.
            ctx.save_for_backward(batch_positions, forces)
        else:
            # Compute the potential energies. Save the wavefunction so that
            # the gradient computation will avoid re-doing the SCF later.
            energies, wavefunctions = _run_psi4(return_wfn=True, **run_psi4_kwargs)

            # Make sure these are HF SCF wavefunction.
            if name != 'scf':
                wavefunctions = [w.reference_wavefunction() for w in wavefunctions]

            ctx.wavefunctions = wavefunctions
            ctx.save_for_backward(batch_positions)

        # Save other variables used for backprop and/or double backprop.
        ctx.name = name
        ctx.batch_positions_arr = batch_positions_arr
        ctx.molecule = molecule
        ctx.energy_unit = energy_unit
        ctx.positions_unit = positions_unit
        ctx.write_orbitals = write_orbitals
        ctx.restart_file = restart_file
        ctx.parallelization_strategy = parallelization_strategy
        ctx.on_unconverged = on_unconverged
        ctx.kwargs = kwargs

        # Convert to unitless tensor.
        energies = energies_array_to_tensor(energies, energy_unit).to(batch_positions)
        return energies

    @staticmethod
    def backward(ctx, grad_output):
        """Compute the gradient of the potential energy."""
        # We still need to return a None gradient for each
        # input of forward() beside batch_positions.
        n_input_args = 11
        grad_input = [None for _ in range(n_input_args)]

        # Check if we need a double backward (i.e. create_graph == True).
        if torch.is_grad_enabled() and (isinstance(ctx.write_orbitals, bool) or ctx.restart_file is None):
            warnings.warn('Psi4PotentialEnergyFunc.backward() was requested to '
                          'create the computational graph to perform double '
                          'backprop, but write_orbitals or restart_file were not '
                          'given. These should point to the same path or the '
                          'performance will be degraded.')

        # Compute gradient w.r.t. batch_positions.
        if ctx.needs_input_grad[0]:
            # Check if we have already computed the forces.
            if len(ctx.saved_tensors) == 2:
                # Retrieve pre-computed forces.
                batch_positions, precomputed_forces = ctx.saved_tensors
            else:
                batch_positions, = ctx.saved_tensors
                precomputed_forces = None

            # We don't really need write_orbitals=True since we have already
            # saved the SCF orbitals during the energy/gradient calculation.
            # If the paths in write_orbitals coincide with restart_file, then
            # they will be read.
            grad_input[0] = _Psi4PotentialEnergyFuncBackward.apply(
                batch_positions,
                grad_output,
                precomputed_forces,
                ctx.name,
                ctx.batch_positions_arr,
                ctx.molecule,
                ctx.wavefunctions,
                ctx.positions_unit,
                ctx.energy_unit,
                ctx.restart_file,
                ctx.parallelization_strategy,
                ctx.on_unconverged,
                ctx.kwargs,
            )

        return tuple(grad_input)


class _Psi4PotentialEnergyFuncBackward(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            batch_positions,
            back_grad_output,
            precomputed_forces,
            name,
            batch_positions_arr,
            molecule,
            wavefunctions,
            positions_unit,
            energy_unit,
            restart_file,
            parallelization_strategy,
            on_unconverged,
            kwargs,
    ):
        # Compute the forces if they weren't computed in forward().
        if precomputed_forces is None:
            precomputed_forces = _run_psi4(
                name=name,
                batch_positions=batch_positions_arr,
                molecule=wavefunctions,
                return_force=True,
                write_orbitals=False,
                restart_file=restart_file,
                parallelization_strategy=parallelization_strategy,
                on_unconverged=on_unconverged,
                **kwargs,
            )

            # From Quantity[numpy] to Tensor and fix units.
            precomputed_forces = forces_array_to_tensor(
                precomputed_forces, positions_unit, energy_unit).to(back_grad_output)

        # We shouldn't pass an SCF wavefunction here since we need the wave
        # function for a perturbed configuration.
        ctx.save_for_backward(batch_positions, back_grad_output, precomputed_forces)

        ctx.name = name
        ctx.batch_positions_arr = batch_positions_arr
        ctx.molecule = molecule
        ctx.energy_unit = energy_unit
        ctx.positions_unit = positions_unit
        ctx.restart_file = restart_file
        ctx.parallelization_strategy = parallelization_strategy
        ctx.on_unconverged = on_unconverged
        ctx.kwargs = kwargs

        # Compute the backward gradient.
        grad_input = precomputed_forces * back_grad_output[:, None]
        return grad_input

    @staticmethod
    def backward(ctx, back_back_grad_output):
        n_input_args = 13
        grad_back = [None for _ in range(n_input_args)]

        batch_positions, back_grad_output, forces = ctx.saved_tensors

        if ctx.needs_input_grad[0]:
            # Perturbation vector for finite-differences estimation of the
            # Hessian-vector product v^T \dot H .
            # v has shape (batch_size, n_atoms*3).
            v = back_back_grad_output * back_grad_output[:, None]

            # Determine epsilon so that the maximum displacement used to perturb
            # the positions is 1e-3 Bohr. max_disp is the maximum displacement
            # in the same units of batch_positions.
            # TODO: Make this a parameter?
            max_disp = 1e-3
            ureg = ctx.batch_positions_arr._REGISTRY
            default_positions_unit = Psi4Potential.default_positions_unit(ureg)
            if ctx.positions_unit is None:
                positions_unit = default_positions_unit
            else:
                # batch_positions is not in bohr.
                positions_unit = ctx.positions_unit
                max_disp = (max_disp * default_positions_unit).to(positions_unit).magnitude

            # epsilon[i] is the scalar multiplying the displacement for batch i.
            # shape: (batch_size, 1).
            epsilon = max_disp / torch.max(batch_positions, dim=1, keepdim=True).values

            # espilon_v shape: (batch_size, n_atoms*3).
            epsilon_v = epsilon * v
            batch_positions_plus = flattened_to_atom(batch_positions + epsilon_v)
            batch_positions_plus = batch_positions_plus.detach().cpu().numpy() * positions_unit
            batch_positions_minus = flattened_to_atom(batch_positions - epsilon_v)
            batch_positions_minus = batch_positions_minus.detach().cpu().numpy() * positions_unit

            # Shared kwargs for _run_psi().
            run_psi4_kwargs = dict(
                name=ctx.name,
                molecule=ctx.molecule,
                return_force=True,
                write_orbitals=False,
                restart_file=ctx.restart_file,
                parallelization_strategy=ctx.parallelization_strategy,
                on_unconverged=ctx.on_unconverged,
            )

            # Compute the two forces.
            forces_plus = _run_psi4(batch_positions=batch_positions_plus,
                                      **run_psi4_kwargs, **ctx.kwargs)
            forces_minus = _run_psi4(batch_positions=batch_positions_minus,
                                       **run_psi4_kwargs, **ctx.kwargs)

            # Convert units.
            forces_plus = forces_array_to_tensor(
                forces_plus, ctx.positions_unit, ctx.energy_unit).to(forces)
            forces_minus = forces_array_to_tensor(
                forces_minus, ctx.positions_unit, ctx.energy_unit).to(forces)

            # Compute the Hessian-vector product.
            hessian_v = (forces_plus - forces_minus) / (2 * epsilon)
            grad_back[0] = hessian_v

        if ctx.needs_input_grad[1]:
            grad_back[1] = back_back_grad_output * forces

        return tuple(grad_back)


def psi4_potential_energy(
        batch_positions,
        name,
        molecule=None,
        positions_unit=None,
        energy_unit=None,
        write_orbitals=False,
        restart_file=None,
        precompute_gradient=True,
        parallelization_strategy=None,
        on_unconverged='raise',
        **kwargs
):
    """PyTorch-differentiable potential energy of a Psi4 molecule.

    PyTorch ``Function``s do not accept keyword arguments. This function wraps
    :func:`.Psi4PotentialEnergyFunc.apply` to enable standard functional notation.
    See the documentation on the original function for the input parameters.

    See Also
    --------
    :class:`.Psi4PotentialEnergyFunc`
        More details on input parameters and implementation details.

    """
    # apply() does not accept keyword arguments.
    return Psi4PotentialEnergyFunc.apply(
        batch_positions,
        name,
        molecule,
        positions_unit,
        energy_unit,
        write_orbitals,
        restart_file,
        precompute_gradient,
        parallelization_strategy,
        on_unconverged,
        kwargs
    )


# =============================================================================
# RUNNING UTILITY FUNCTIONS
# =============================================================================

def _run_psi4(
        name,
        batch_positions=None,
        molecule=None,
        return_energy=False,
        return_force=False,
        return_wfn=False,
        write_orbitals=False,
        restart_file=None,
        unit_registry=None,
        parallelization_strategy=None,
        on_unconverged='raise',
        **kwargs
):
    """Compute the potential energy and gradient of a Psi4 ``Molecule``.

    This function is a wrapper of ``psi4.energy`` and ``psi4.gradient`` that
    handles:
    - Positions and energy units.
    - Batches of positions.
    - Positions with ``Tensor`` (flattened) or numpy (standard 2D) shape.
    - Restart/write wavefunction that is sample-specific within a batch.
    - Batch parallelization implementation through ``ParallelizationStrategy``
      objects.

    Whether ``energy`` or ``gradient`` is called depends on the value of
    ``return_force``.

    Some notes to remember about the internals of Psi4.

    Restart files are normally serialized ``Wavefunction``s created either by
    setting ``writer_orbitals`` to ``True`` or by calling ``Wavefunction.to_file``.
    They are read/written only for the Hartree-Fock SCF calculation so they are
    ignored when ``ref_wfn`` is passed to ``psi4.gradient``. Also, this means
    that even with a restart file an SCF calculation is required for post-HF
    methods (e.g., MP2).

    If ``writer_orbitals`` is ``True``, the wavefunction files at a path hardcoded
    in its function here

        https://github.com/psi4/psi4/blob/9485035a0cd5d9a39582c9d7c4406f64aa12b838/psi4/driver/p4util/python_helpers.py#L137-L145

    and the file is deleted at the end of the execution. The files are preserved only
    if ``write_orbitals`` is the path to the saved file. See

        https://github.com/psi4/psi4/blob/9485035a0cd5d9a39582c9d7c4406f64aa12b838/psi4/driver/procrouting/proc.py#L1309-L1317
        https://github.com/psi4/psi4/blob/9485035a0cd5d9a39582c9d7c4406f64aa12b838/psi4/driver/procrouting/proc.py#L1655-L1663

    Internally, when the ``restart_file`` option is passed, Psi4 automatically
    copy it to the ``Wavefunction`` hardcoded path and sets
    ``psi4.core.set_local_option('SCF', 'GUESS', 'READ')``. See

        https://github.com/psi4/psi4/blob/9485035a0cd5d9a39582c9d7c4406f64aa12b838/psi4/driver/driver.py#L558-L563

    Contrarily to ``psi4.gradient``, ``psi4.energy`` cannot take a reference
    wavefunction as a ``ref_wfn`` argument. See ``psi4.driver.procrouting.proc.scf_helper``).

        https://github.com/psi4/psi4/blob/9485035a0cd5d9a39582c9d7c4406f64aa12b838/psi4/driver/procrouting/proc.py#L1305-L1307

    To restart a calculation, the only available route for ``energy`` is the
    restart file.

    Parameters
    ----------
    name : str
        The name of the potential to pass to ``psi4.energy()``.
    batch_positions : pint.Quantity, optional
        An array of positions with units and shape: ``(batch_size, n_atoms, 3)`` or
        ``(n_atoms, 3)``. If ``None``, the geometry of the active Psi4 ``Molecule``
        is evaluated. Otherwise, this overwrites the geometry in the active
        ``Molecule``. If ``Wavefunction`` objects are passed in the ``molecule``
        argument, ``batch_positions`` is ignored.
    molecule : psi4.core.Molecule or psi4.core.Wavefunction or List[psi4.core.Wavefunction], optional
        If not ``None``, this will be set as the currently activated molecule
        in Psi4. Note that the old active molecule is not restored at the end
        of the execution.

        If this is a ``Wavefunction`` object (or a list of wavefunctions, one
        for each batch sample), this is assumed to be the converged Hartree-Fock
        SCF wavefunction at the correct geometry and the HF SCF calculation is
        skipped (i.e., the equivalent of the ``ref_wfn`` in ``psi4.gradient``.
        This will cause an error if done with potential energy calculations. Note
        that if you compute the gradient of a method different than SCF, this
        does not return the SCF wavefunction. To obtain it you must call
        ``Wavefunction.reference_wavefunction()``.
    return_energy : bool, optional
        If ``True``, the potential energies are returned.
    return_forces : bool, optional
        If ``True``, ``psi4.gradient`` is called instead of ``psi4.energy`` and
        the forces are returned.
    return_wfn : bool, optional
        If ``True``, the wavefunctions are also returned.
    write_orbitals : bool or str or List[str], optional
        This option is passed to ``psi4.energy`` to store the wavefunction on
        disk at each Hartree-Fock SCF iteration, which can later be used to
        the calculation. If a ``list``, it must specify a path to the path to the
        restart file to write for each batch sample.
    restart_file : str or List[str], optional
        A Psi4 restart file path (or a list of restart file paths, one for each
        batch sample) that can be used as a starting point for the Hartree-Fock
        SCF optimization.
    unit_registry : pint.UnitRegistry, optional
        The unit registry to use for the energy units. If ``None`` and
        ``batch_positions`` has units attached, the unit registry of the positions
        is used. Otherwise, the forces will be returned as a standard numpy array
        in the units returned by Psi4.
    parallelization_strategy : tfep.utils.parallel.ParallelizationStrategy, optional
        The parallelization strategy used to distribute batches of energy and
        gradient calculations. By default, these are executed serially using
        the thread-based parallelization native in psi4.
    on_unconverged : str, optional
        Specifies how to handle the case in which the calculation did not converge.
        It can have the following values:
        - ``'raise'``: Raise the Psi4 exception.
        - ``'nan'``: Return ``float('nan')`` energy, zero forces, and the latest
                     wavefunction.
        To treat the calculation as converged and return the latest energy, force,
        and/or wavefunction, simply set the psi4 global option ``'fail_on_maxiter'``.
    **kwargs
        Other keyword arguments to forward to ``psi4.energy`` or ``psi4.gradient``.

    Returns
    -------
    energies : pint.Quantity, optional
        If ``batch_positions`` was not given or if it is of shape ``(n_atoms, 3)``,
        this is a single float. Otherwise, ``energies[i]`` is the potential of
        configuration ``batch_positions[i]``.

        This is returned only if ``return_energy`` is ``True``.
    forces : torch.Tensor or pint.Quantity, optional
        If ``batch_positions`` was not given or if it is of shape ``(n_atoms, 3)``,
        this is a numpy array of shape ``(n_atoms, 3)`` with the molecule's force.
        Otherwise, ``forces[i]`` is the force of configuration ``batch_positions[i]``.

        This is returned only if ``return_force`` is ``True``.
    wavefunctions : psi4.core.Wavefunction or List[psi4.core.Wavefunction], optional
        The wavefunction or, if ``batch_positions`` is given, a list of the HF-SCF
        optimized wavefunctions for each batch sample.

        This is returned only if ``return_wfn`` is ``True``.

    """
    import psi4

    # Check input arguments.
    if on_unconverged not in {'raise', 'nan'}:
        raise ValueError('on_unconverged must be one of "raise" or "nan".')

    # Determine which psi4 function to call.
    if return_force:
        func = psi4.gradient
    else:
        func = psi4.energy

    # Handle mutable default arguments.
    if parallelization_strategy is None:
        parallelization_strategy = SerialStrategy()

    # Create a unit energy.
    if unit_registry is None:
        try:
            unit_registry = batch_positions._REGISTRY
        except AttributeError:
            unit_registry = pint.UnitRegistry()

    if batch_positions is None:
        # Convert to a list to avoid code branching.
        batch_positions_bohr = [None]
    else:
        batch_positions_bohr = batch_positions.to(Psi4Potential.DEFAULT_POSITIONS_UNIT).magnitude
        if len(batch_positions.shape) < 3:
            # Convert batch_positions to a unitless (in Psi4 units) numpy array
            # of shape (batch_size, n_atoms, 3).
            batch_positions_bohr = np.expand_dims(batch_positions_bohr, axis=0)

    # Check the length of all lists.
    batch_size = len(batch_positions_bohr)
    for x in [molecule, write_orbitals, restart_file]:
        if isinstance(x, list) and len(x) != batch_size:
            raise ValueError('molecule, write_orbitals, and restart_file must '
                             'have the same length as batch_positions.')

    # Convert all sample-specific options to lists to avoid code branching.
    if not isinstance(write_orbitals, list):
        write_orbitals = [write_orbitals] * batch_size
    if not isinstance(restart_file, list):
        restart_file = [restart_file] * batch_size

    # Handle the reference wavefunctions and the active molecule.
    if isinstance(molecule, psi4.core.Wavefunction):
        # Taken care in next if-clause.
        molecule = [molecule] * batch_size
    if isinstance(molecule, collections.abc.Sequence):
        # molecule is list-like of psi4.core.Wavefunctions.
        ref_wfn = molecule
        molecule = ref_wfn[0].molecule()
    else:
        # molecule is psi4.core.Molecule or None.
        ref_wfn = [None] * batch_size

    # Check consistent input.
    use_ref_wfn = not all([x is None] for x in ref_wfn)
    use_restart_file = not all([x is None] for x in restart_file)
    if use_ref_wfn and use_restart_file:
        raise ValueError('Cannot pass both ref_wfn and restart_file.')

    # Run all batches with the provided parallelization strategy.
    # We use functools.partial to encode the arguments that are common to all tasks.
    task = functools.partial(
        _run_psi4_task, func, molecule, name, return_energy, return_force, return_wfn, on_unconverged, kwargs)
    distributed_args = zip(batch_positions_bohr, ref_wfn, write_orbitals, restart_file)
    batch_results = parallelization_strategy.run(task, distributed_args)

    # Unpack the results.
    batch_results = list(zip(*batch_results))
    if return_energy:
        energies = batch_results[0]
        if return_force:
            forces = batch_results[1]
    elif return_force:
        forces = batch_results[0]
    if return_wfn:
        wavefunctions = batch_results[-1]

    # Prepare returned values and handle units.
    returned_values = []
    if return_energy:
        returned_values.append(energies * Psi4Potential.default_energy_unit(unit_registry))
    if return_force:
        returned_values.append(forces * Psi4Potential.default_energy_unit(unit_registry) / Psi4Potential.default_positions_unit(unit_registry))
    if return_wfn:
        returned_values.append(wavefunctions)

    # Return values must have the correct shape.
    is_single_configuration = batch_positions is None or len(batch_positions.shape) < 3
    if is_single_configuration:
        for i, returned_value in enumerate(returned_values):
            returned_values[i] = returned_value[0]

    if len(returned_values) == 1:
        return returned_values[0]
    return returned_values


def _run_psi4_task(func, molecule, name, return_energy, return_force, return_wfn, on_unconverged, kwargs,
                   positions_bohr, ref_wfn, write_orbitals, restart_file):
    """This is the task that is parallelized with ``ParallelizationStrategy``."""
    import psi4

    # Activate the molecule.
    if molecule is None:
        molecule = psi4.core.get_active_molecule()
    else:
        psi4.core.set_active_molecule(molecule)

    # If positions is None, then we need to use the molecule's internal geometry.
    if positions_bohr is not None:
        molecule.set_geometry(psi4.core.Matrix.from_array(positions_bohr))
        molecule.update_geometry()

    # We cannot pass restart_file = None as psi4 will crash.
    if restart_file is None:
        more_kwargs = {}
    else:
        more_kwargs = {'restart_file': restart_file}

    # Run the function.
    needs_wfn = True if return_force and return_energy else return_wfn

    # Handle unconverged calculations.
    try:
        result = func(name=name, return_wfn=needs_wfn, ref_wfn=ref_wfn,
                      write_orbitals=write_orbitals, **kwargs, **more_kwargs)
    except psi4.ConvergenceError as e:
        if on_unconverged == 'raise':
            raise
        else:  # on_unconverged == 'nan':
            result = []
            if return_energy:
                result.append(float('nan'))
            if return_force:
                result.append(np.zeros_like(e.wfn.molecule().geometry().to_array()))
            if return_wfn:
                result.append(e.wfn)
            return result

    # Because pickle cannot send Psi4 Matrix and Wavefunction objects, we convert
    # them in the subprocess before sending them back.
    return_values = []

    if needs_wfn:
        result_wfn = result[1]
        result = result[0]

    if return_force:
        if return_energy:
            return_values.append(result_wfn.energy())
        # If func is psi4.gradient, the result is a Psi4 Matrix object.
        force = result.to_array()
        return_values.append(force)
    elif return_energy:
        return_values.append(result)

    if return_wfn:
        return_values.append(result_wfn)

    return return_values
