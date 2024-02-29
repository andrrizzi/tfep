#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""Base ``LightningModule`` class to implement TFEP maps."""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import lightning
import pint
import MDAnalysis
import torch

import tfep.loss
import tfep.io.sampler
from tfep.utils.misc import atom_to_flattened_indices


# =============================================================================
# TFEP MAP BASE CLASS
# =============================================================================

class TFEPMapBase(ABC, lightning.LightningModule):
    """A ``LightningModule`` to run TFEP calculations.

    This abstract class implements several data-related utilities that are shared
    by all TFEP maps. In particular to:

    - Support correct mid-epoch resuming.
    - Log vectorial quantities such as the calculated potential energies and the
      absolute Jacobian terms that can later be used to estimate the free energy.
    - Identify and perform consistency checks on three atoms (origin, axis, and
      plane) that can be used to define a relative frame of reference for the flow.
    - Identify the mapped, conditioning, and fixed atom indices and handle fixed
      atoms.

    For the latter, mapped atoms are defined as those that the flow maps.
    Conditioning atoms are not mapped but are given as input to the flow to
    condition the mapping. Fixed atoms are instead ignored. Note that the flow
    defined child class must handle only the mapped and conditioning atoms. The
    flow will be automatically wrapped in a :class:`~tfep.nn.flows.PartialFlow`
    to handle the fixed atoms.

    The class further provides convenience methods to retrieve the indices of
    the mapped and conditioning atoms/degrees of freedom after the fixed atoms
    are removed through the methods :func:`~TFEPMapBase.get_mapped_indices` and
    :func:`~TFEPMapBase.get_conditioning_indices` (see example below). A similar
    function exist for the atom indices of the reference frame atoms called
    :func:`~TFEPMapBase.get_reference_atoms_indices`

    The only required method to implement a concrete class is :func:`~TFEPMapBase.configure_flow`.

    .. warning::

        Currently, this class is not multi-process or thread safe. Running with
        multiple processes may result in the corrupted logging of the potentials
        and Jacobians.

    Examples
    --------

    Here is an example of how to implement a working map using ``TFEPMapBase``.

    >>> class TFEPMap(TFEPMapBase):
    ...
    ...     def configure_flow(self):
    ...         # A simple 1-layer affine autoregressive flow.
    ...         # The flow must fix the relative frame of reference or raise
    ...         # an error when origin and axes atoms are set.
    ...         reference_atoms_indices = self.get_reference_atoms_indices()
    ...         if reference_atoms_indices is not None:
    ...             raise NotImplementedError('Relative frame of reference is not supported.')
    ...
    ...         # The flow must take care only of the mapped and conditioning atoms.
    ...         conditioning_indices = self.get_conditioning_indices(
    ...             idx_type="dof", remove_fixed=True, remove_reference=True)
    ...         return tfep.nn.flows.MAF(
    ...             dimension_in=self.n_nonfixed_dofs,
    ...             conditioning_indices=conditioning_indices,
    ...         )
    ...

    After this, the TFEP calculation can be run using.

    >>> from tfep.potentials.psi4 import Psi4Potential
    >>> units = pint.UnitRegistry()
    >>>
    >>> tfep_map = TFEPMap(
    ...     potential_energy_func=Psi4Potential(name='mp2'),
    ...     topology_file_path='path/to/topology.psf',
    ...     coordinates_file_path='path/to/trajectory.dcd',
    ...     temperature=300*units.kelvin,
    ...     batch_size=64,
    ...     mapped_atoms='resname MOL',  # MDAnalysis selection syntax.
    ...     conditioning_atoms=range(10, 20),
    ... )
    >>>
    >>> # Train the flow and save the potential energies.
    >>> trainer = lightning.Trainer()
    >>> trainer.fit(tfep_map)  # doctest: +SKIP

    """

    def __init__(
            self,
            potential_energy_func: torch.nn.Module,
            topology_file_path: str,
            coordinates_file_path: Union[str, Sequence[str]],
            temperature: pint.Quantity,
            batch_size: int = 1,
            mapped_atoms: Optional[Union[Sequence[int], str]] = None,
            conditioning_atoms: Optional[Union[Sequence[int], str]] = None,
            origin_atom: Optional[Union[int, str]] = None,
            axes_atoms: Optional[Union[Sequence[int], str]] = None,
            tfep_logger_dir_path: str = 'tfep_logs',
            dataloader_kwargs: Optional[Dict] = None,
    ):
        """Constructor.

        Parameters
        ----------
        potential_energy_func : torch.nn.Module
            A PyTorch module encapsulating the target potential energy function
            (e.g. :class:`tfep.potentials.psi4.ASEPotential`).
        topology_file_path : str
            The path to the topology file. The file can be in `any format supported
            by MDAnalysis <https://docs.mdanalysis.org/stable/documentation_pages/topology/init.html#supported-topology-formats>`__
            which is automatically detected from the file extension.
        coordinates_file_path : str or Sequence[str]
            The path(s) to the trajectory file(s). If a sequence of files is given,
            the trajectories are concatenated into a single large dataset. The
            file(s) can be in `any format supported by MDAnalysis <https://docs.mdanalysis.org/stable/documentation_pages/coordinates/init.html#id2>`__
            which is automatically detected from the file extension.
        temperature : pint.Quantity
            The temperature of the ensemble.
        batch_size : int, optional
            The batch size.
        mapped_atoms : Sequence[int] or str or None, optional
            The indices (0-based) of the atoms to map or a selection string in
            MDAnalysis syntax. If not passed, all atoms that are not conditioning
            are mapped (i.e., all atoms are mapped if also ``conditioning_atoms``
            is not given.
        conditioning_atoms : Sequence[int] or str or None, optional
            The indices (0-based) of the atoms conditioning the mapping or a
            selection string in MDAnalysis syntax. If not passed, no atom will
            condition the map.
        origin_atom : int or str or None, optional
            The index (0-based) or a selection string in MDAnalysis syntax of an
            atom on which to center the origin of the relative frame of reference.
            While this atom affects the mapping of the mapped atoms, its position
            will be constrained during the mapping, and thus it must be a conditioning
            atom by definition.
        axes_atoms : Sequence[int] or str or None, optional
            A pair of indices (0-based) or a selection string in MDAnalysis syntax
            for the two atoms determining the relative frame of reference.

            The details of how these are used to fix the orientation of the frame
            of reference depend on the implementation of :func:`~TFEPMapBase.configure_flow`.
            For example, ``axes_atoms[0]``-th atom may lay on the ``z`` axis ,
            and the ``axes_atoms[1]``-th atom may lay on the plane spanned by
            the ``x`` and ``z`` axes.

            These atoms can be either conditioning or mapped. ``axes_atoms[0]``
            has only 1 degree of freedom (DOF) while ``axes_atoms[1]`` has 2.
            Whether these DOFs are mapped or not depends on whether their atoms
            are indicated as mapped or conditioning, respectively.
        tfep_logger_dir_path : str, optional
            The path where to save TFEP-related information (potential energies,
            sample indices, etc.).
        dataloader_kwargs : Dict, optional
            Extra keyword arguments to pass to ``torch.utils.data.DataLoader``.

        See Also
        --------
        `MDAnalysis Universe object <https://docs.mdanalysis.org/2.6.1/documentation_pages/core/universe.html#MDAnalysis.core.universe.Universe>`_

        """
        super().__init__()

        # Make sure coordinates_file_paths is a sequence.
        if isinstance(coordinates_file_path, str):
            coordinates_file_path = [coordinates_file_path]

        # batch_size is saved as a hyperparmeter of the module.
        self.save_hyperparameters('batch_size', 'mapped_atoms', 'conditioning_atoms', 'origin_atom', 'axes_atoms')

        # Potential energy.
        self._potential_energy_func = potential_energy_func

        # Paths to files.
        self._topology_file_path = topology_file_path
        self._coordinates_file_path = coordinates_file_path
        self._tfep_logger_dir_path = tfep_logger_dir_path

        # Internally, rather than the temperature, save the (unitless) value of
        # kT, in the same units of energy returned by potential_energy_func.
        units = temperature._REGISTRY
        try:
            kT = (temperature * units.molar_gas_constant).to(potential_energy_func.energy_unit)
        except pint.errors.DimensionalityError:
            kT = (temperature * units.boltzmann_constant).to(potential_energy_func.energy_unit)
        self.register_buffer('_kT', torch.tensor(kT.magnitude))

        # KL divergence loss function.
        self._loss_func = tfep.loss.BoltzmannKLDivLoss()

        # Dataloader kwargs.
        self._dataloader_kwargs = dataloader_kwargs

        # The following variables can be data-dependent and are thus initialized
        # dynamically in setup(), which is called by Lightning by all processes.
        # This class is not currently parallel-safe as the TFEPLogger is not, but
        # I organized the code as suggested by Lightning's docs anyway as I plan
        # to add support for this at some point.
        self.dataset: Optional[tfep.io.dataset.TrajectoryDataset] = None  #: The dataset.

        # Register buffers so that they get automatically moved to the correct device.
        self.register_buffer('_mapped_atom_indices', None)  # The indices of the mapped atoms.
        self.register_buffer('_conditioning_atom_indices', None)  # The indices of the conditioning atoms.
        self.register_buffer('_fixed_atom_indices', None)  # The indices of the fixed atoms.
        self.register_buffer('_origin_atom_idx', None)  # The index of the origin atom.
        self.register_buffer('_axes_atoms_indices', None)  # The indices of the axis and plane atoms.
        self._flow = None  # The normalizing flow model.
        self._stateful_batch_sampler = None  # Batch sampler for mid-epoch resuming.
        self._tfep_logger = None  # The logger where to save the potentials.

    def setup(self, stage: str = 'fit'):
        """Lightning method.

        This is executed on all processes by Lightning in DDP mode (contrary to
        ``__init__``) and can be used to initialize objects like the ``Dataset``
        and all data-dependent objects.

        """
        # Create TrajectoryDataset. This sets self.dataset.
        self.dataset = self.create_dataset()

        # Identify mapped, conditioning, and fixed atom indices.
        self.determine_atom_indices()

        # Create model.
        flow = self.configure_flow()

        # Wrap in partial flow(s) to carry over the fixed degrees of freedom.
        self._flow = self.create_partial_flow(flow)

    @abstractmethod
    def configure_flow(self) -> torch.nn.Module:
        """Initialize the normalizing flow.

        Note that the flow must handle only the mapped and conditioning atoms.
        The fixed atoms will be instead automatically wrapped in a
        :class:`~tfep.nn.flows.PartialFlow`.

        The method must also set the flow to fix the reference frame of reference
        based on the origin and axes atoms. For example by using
        :class`~tfep.nn.flows.OrientedFlow` and :class`~tfep.nn.flows.CenteredCentroidFlow`.

        Returns
        -------
        flow : torch.nn.Module
            The normalizing flow.

        """

    def configure_optimizers(self):
        """Lightning method.

        Returns
        -------
        optimizer : torch.optim.optimizer.Optimizer
            The optimizer to use for the training.

        """
        return torch.optim.AdamW(self._flow.parameters())

    @property
    def n_mapped_atoms(self) -> int:
        """The number of mapped atoms."""
        return len(self._mapped_atom_indices)

    @property
    def n_mapped_dofs(self) -> int:
        """The number of mapped degrees of freedom (excluding the constrained DOFs of the reference frame atoms)."""
        n_mapped_dofs = 3 * self.n_mapped_atoms

        # Check if the unconstrained DOFs of the axes atoms are mapped (the origin
        # atom is always conditioning).
        if self._axes_atoms_indices is not None:
            is_atom_0_mapped, is_atom_1_mapped = self.are_axes_atoms_mapped()
            if is_atom_0_mapped:
                n_mapped_dofs -= 2
            if is_atom_1_mapped:
                n_mapped_dofs -= 1

        return n_mapped_dofs

    @property
    def n_conditioning_atoms(self) -> int:
        """The number of conditioning atoms."""
        if self._conditioning_atom_indices is None:
            return 0
        return len(self._conditioning_atom_indices)

    @property
    def n_conditioning_dofs(self) -> int:
        """The number of conditioning degrees of freedom (excluding the constrained DOFs of the reference frame atoms)."""
        n_conditioning_dofs = 3 * self.n_conditioning_dofs

        # Remove constrained DOFs of the origin atom which is always conditioning.
        if self._origin_atom_idx is not None:
            n_conditioning_dofs -= 3

        # Remove constrained DOFs of the axes atoms.
        if self._axes_atoms_indices is not None:
            is_atom_0_mapped, is_atom_1_mapped = self.are_axes_atoms_mapped()
            if not is_atom_0_mapped:
                n_conditioning_dofs -= 2
            if not is_atom_1_mapped:
                n_conditioning_dofs -= 1

        return n_conditioning_dofs

    @property
    def n_fixed_atoms(self) -> int:
        """The number of fixed atoms."""
        if self._fixed_atom_indices is None:
            return 0
        return len(self._fixed_atom_indices)

    @property
    def n_nonfixed_atoms(self) -> int:
        """Total number of mapped and conditioning atoms."""
        return self.n_mapped_atoms + self.n_conditioning_atoms

    @property
    def n_nonfixed_dofs(self) -> int:
        """Total number of mapped and conditioning degrees of freedom (excluding the constrained DOFs of the reference frame atoms)."""
        n_nonfixed_dofs = 3 * self.n_nonfixed_atoms
        if self._origin_atom_idx is not None:
            n_nonfixed_dofs -= 3
        if self._axes_atoms_indices is not None:
            n_nonfixed_dofs -= 3
        return n_nonfixed_dofs

    def are_axes_atoms_mapped(self):
        """Return whether the two axes atoms (if any) are mapped.

        Returns
        -------
        are_mapped : None or Tuple[bool]
            A pair ``(is_axes_atom_0_mapped, is_axes_atom_1_mapped)`` or ``None``
            if there are no axes atoms.

        """
        if self._axes_atoms_indices is None:
            return None

        if self.n_conditioning_atoms == 0:
            return True, True
        elif self.n_conditioning_atoms > self.n_mapped_atoms:
            is_atom_0_mapped = torch.any(self._mapped_atom_indices == self._axes_atoms_indices[0])
            is_atom_1_mapped = torch.any(self._mapped_atom_indices == self._axes_atoms_indices[1])
        else:
            is_atom_0_mapped = not torch.any(self._conditioning_atom_indices == self._axes_atoms_indices[0])
            is_atom_1_mapped = not torch.any(self._conditioning_atom_indices == self._axes_atoms_indices[1])

        return is_atom_0_mapped, is_atom_1_mapped

    def get_mapped_indices(
            self,
            idx_type: Literal['atom', 'dof'],
            remove_fixed: bool,
    ) -> torch.Tensor:
        """Return the indices of the mapped atom or degrees of freedom (DOF).

        Each atom generally has 3 degrees of freedom, except for the atoms used
        to set the relative frame of reference. If the ``axes_atoms`` (or only
        one of them) have been indicated as mapped, the returned conditioning
        DOFs indices also include the DOFs of the ``axes_atoms`` that are not
        constrained, i.e., the ``x`` coordinate of ``axes_atoms[0]``, and the
        ``x,y`` coordinates of ``axes_atoms[1]``.

        Parameters
        ----------
        idx_type : Literal['atom', 'dof']
            Whether to return the indices of the atom or the degrees of freedom.
        remove_fixed : bool
            If ``True``, the returned tensor represent the indices after the
            fixed atoms have been removed.

        Returns
        -------
        indices : torch.Tensor
            The mapped atom/DOFs indices.

        """
        return self._get_nonfixed_indices(self._mapped_atom_indices, idx_type, remove_fixed)

    def get_conditioning_indices(
            self,
            idx_type: Literal['atom', 'dof'],
            remove_fixed: bool,
    ) -> torch.Tensor:
        """Return the indices of the conditioning atom or degrees of freedom (DOF).

        Each atom generally has 3 degrees of freedom, except for the atoms used
        to set the relative frame of reference. If the ``axes_atoms`` (or only
        one of them) have been indicated as conditioning, the returned conditioning
        DOFs indices also include the DOFs of the ``axes_atoms`` that are not
        constrained, i.e., the ``x`` coordinate of ``axes_atoms[0]``, and the
        ``x,y`` coordinates of ``axes_atoms[1]``. The ``origin_atom`` is always
        a conditioning atom by definition, and it is thus included in the returned
        indices.

        Parameters
        ----------
        idx_type : Literal['atom', 'dof']
            Whether to return the indices of the atom or the degrees of freedom.
        remove_fixed : bool
            If ``True``, the returned tensor represent the indices after the
            fixed atoms have been removed.

        Returns
        -------
        indices : torch.Tensor
            The conditioning atom/DOFs indices.

        """
        # Conditioning atoms might be None.
        if self.n_conditioning_atoms == 0:
            return None
        return self._get_nonfixed_indices(self._conditioning_atom_indices, idx_type, remove_fixed)

    def get_reference_atoms_indices(
            self,
            remove_fixed: bool,
            separate_origin_axes: bool = False,
    ) -> Union[torch.Tensor, None, List[Union[torch.Tensor, None]]]:
        """Return the atom indices of the origin and axes atoms.

        Parameters
        ----------
        remove_fixed : bool
            If ``True``, the returned tensor represent the indices after the
            fixed atoms have been removed.
        separate_origin_axes : bool, optional
            If ``True``, the origin and axes atoms are returned separately in
            two ``Tensors``. Otherwise, a single ``Tensor`` is returned.

        Returns
        -------
        reference_atom_indices : torch.Tensor or None or List[torch.Tensor | None]
            If ``separate_origin_axes is False``, a single ``Tensor`` including
            the indices, in this order, of the origin, axis, and plane atoms (if
            they exist) or ``None`` if there are no origin and axes atoms.

            If ``separate_origin_axes is False``, this is a pair of ``Tensors``
            (or ``None``) holding the origin atom index and the axes atom indices.

        """
        # Shortcuts.
        has_origin = self._origin_atom_idx is not None
        has_axes = self._axes_atoms_indices is not None

        # Return None if no origin/axes atoms are given.
        if not (has_origin or has_axes):
            if separate_origin_axes:
                return [None, None]
            return None

        # Initialize return value.
        reference_atom_indices = [
            self._origin_atom_idx if has_origin else None,
            self._axes_atoms_indices if has_axes else None,
        ]

        # Remove fixed atoms.
        if remove_fixed and self.n_fixed_atoms > 0:
            for i, atom_indices in enumerate(reference_atom_indices):
                if atom_indices is not None:
                    shift = torch.searchsorted(self._fixed_atom_indices, atom_indices)
                    reference_atom_indices[i] = atom_indices - shift

        # Concatenate if requested.
        if not separate_origin_axes:
            # The returned tensor must be at least 1D.
            if has_origin:
                reference_atom_indices[0] = reference_atom_indices[0].unsqueeze(0)

            # Concatenate if both origin and axes are present.
            if has_origin and has_axes:
                reference_atom_indices = torch.cat(reference_atom_indices)
            else:
                reference_atom_indices.remove(None)
                reference_atom_indices = reference_atom_indices[0]

        return reference_atom_indices

    def create_universe(self):
        """Create and return the MDAnalysis ``Universe``.

        Returns
        -------
        universe : MDAnalysis.Universe
            The MDAnalysis ``Universe`` object.

        """
        return MDAnalysis.Universe(self._topology_file_path, *self._coordinates_file_path)

    def create_dataset(self):
        """Create and return the ``Dataset`` object.

        Returns
        -------
        dataset : torch.utils.data.Dataset
            The PyTorch dataset.

        """
        universe = self.create_universe()
        return tfep.io.TrajectoryDataset(universe=universe)

    def create_partial_flow(self, flow: torch.nn.Module, return_partial: bool = False) -> torch.nn.Module:
        """Wrap the flow to remove the fixed DOFs.

        Parameters
        ----------
        flow : torch.nn.Module
            The flow to be wrapped in the Partial and/or Oriented/CenteredCentroid
            flows.
        return_partial : bool, optional
            The ``return_partial`` flag of the :class:`~tfep.nn.flows.PartialFlow`.

        Returns
        -------
        flow : torch.nn.Module
            The wrapped flow.

        """
        # Wrap in a partial flow to carry over fixed degrees of freedom.
        if self.n_fixed_atoms > 0:
            fixed_dof_indices = atom_to_flattened_indices(self._fixed_atom_indices)
            flow = tfep.nn.flows.PartialFlow(
                flow,
                fixed_indices=fixed_dof_indices,
                return_partial=return_partial,
            )

        return flow

    def determine_atom_indices(self):
        """Determine mapped, conditioning, fixed, and reference frame atom indices.

        This initializes the following attributes
        - ``self._mapped_atom_indices``
        - ``self._conditioning_atom_indices``
        - ``self._fixed_atom_indices``
        - ``self._origin_atom_idx``
        - ``self._axes_atoms_indices``

        """
        # Shortcuts.
        mapped = self.hparams.mapped_atoms
        conditioning = self.hparams.conditioning_atoms
        origin = self.hparams.origin_atom
        axes = self.hparams.axes_atoms
        n_atoms = self.dataset.n_atoms

        # Used to check for duplicate selected atoms.
        mapped_set = None
        conditioning_set = None
        non_fixed_set = None

        if (mapped is None) and (conditioning is None):
            # Everything is mapped.
            mapped_atom_indices = torch.tensor(range(n_atoms))
            conditioning_atom_indices = None
            fixed_atom_indices = None
        elif conditioning is None:
            # Everything that is not mapped is fixed (no conditioning.
            mapped_atom_indices = self._get_selected_indices(mapped)
            conditioning_atom_indices = None
            mapped_set = set(mapped_atom_indices.tolist())
            fixed_atom_indices = torch.tensor(
                [idx for idx in range(n_atoms) if idx not in mapped_set])
        elif mapped is None:
            # Everything that is not conditioning is mapped (no fixed).
            conditioning_atom_indices = self._get_selected_indices(conditioning)
            conditioning_set = set(conditioning_atom_indices.tolist())
            mapped_atom_indices = torch.tensor(
                [idx for idx in range(n_atoms) if idx not in conditioning_set])
            fixed_atom_indices = None
        else:
            # Everything needs to be selected.
            mapped_atom_indices = self._get_selected_indices(mapped)
            conditioning_atom_indices = self._get_selected_indices(conditioning)

            # Make sure that there are no overlapping atoms.
            mapped_set = set(mapped_atom_indices.tolist())
            conditioning_set = set(conditioning_atom_indices.tolist())
            if len(mapped_set & conditioning_set) > 0:
                raise ValueError('Mapped and conditioning selections cannot have overlapping atoms.')

            non_fixed_set = mapped_set | conditioning_set
            fixed_atom_indices = torch.tensor(
                [idx for idx in range(n_atoms) if idx not in non_fixed_set])

        # Make sure conditioning and fixed atoms are None if they are empty.
        if (conditioning_atom_indices is not None) and (len(conditioning_atom_indices) == 0):
            conditioning_atom_indices = None
        if (fixed_atom_indices is not None) and (len(fixed_atom_indices) == 0):
            fixed_atom_indices = None

        # Make sure there are atoms to map.
        if len(mapped_atom_indices) == 0:
            raise ValueError('There are no atoms to map.')

        # Check that there are no duplicate atoms.
        if (mapped_set is not None and
                    len(mapped_set) != len(mapped_atom_indices)):
                raise ValueError('There are duplicate mapped atom indices.')
        if (conditioning_set is not None and
                    len(conditioning_set) != len(conditioning_atom_indices)):
                raise ValueError('There are duplicate conditioning atom indices.')

        # Select origin atom.
        if origin is None:
            origin_atom_idx = None
        else:
            origin_atom_idx = self._get_selected_indices(origin, sort=False)

            # String selections are returned as an array containing 1 index.
            if len(origin_atom_idx.shape) > 0:
                if origin_atom_idx.numel() > 1:
                    raise ValueError('Selected multiple atoms as the origin atom')
                origin_atom_idx = origin_atom_idx[0]

            # Make sure origin is a fixed atom.
            if (conditioning_atom_indices is None or
                        origin_atom_idx not in conditioning_atom_indices):
                raise ValueError("origin_atom is not a conditioning atom. origin_atom "
                                 "affects the mapping but its position is constrained.")

        # Select axes atoms.
        if axes is None:
            axes_atoms_indices = None
        else:
            # In this case we must maintain the given order.
            axes_atoms_indices = self._get_selected_indices(axes, sort=False)
            if len(axes_atoms_indices) != 2:
                raise ValueError('Exactly 2 axes atoms must be given.')

            # Check that the atoms don't overlap.
            reference_atoms = axes_atoms_indices
            if origin is not None:
                reference_atoms = torch.cat((origin_atom_idx.unsqueeze(0), reference_atoms))
            if len(reference_atoms.unique()) != len(reference_atoms):
                raise ValueError("center, axis, and plane atoms must be different")

            # Check that the axes atoms are not flagged as fixed.
            if fixed_atom_indices is None:
                are_axes_atom_fixed = False
            else:
                if non_fixed_set is None:
                    if mapped_set is None:
                        mapped_set = set(mapped_atom_indices.tolist())
                    if (conditioning_set is None) and (conditioning_atom_indices is not None):
                        conditioning_set = set(conditioning_atom_indices.tolist())
                        non_fixed_set = mapped_set | conditioning_set
                    else:
                        non_fixed_set = mapped_set

                axes_atom_indices_set = set(axes_atoms_indices.tolist())
                are_axes_atom_fixed = len(axes_atom_indices_set & non_fixed_set) != 2

            if are_axes_atom_fixed:
                raise ValueError("axis and plane atoms must be mapped or conditioning "
                                 "atoms as they affect the mapping.")

        # Register as buffer so that they get automatically moved to the correct device.
        self._mapped_atom_indices = mapped_atom_indices
        self._conditioning_atom_indices = conditioning_atom_indices
        self._fixed_atom_indices = fixed_atom_indices
        self._origin_atom_idx = origin_atom_idx
        self._axes_atoms_indices = axes_atoms_indices

    def forward(self, x):
        """Execute the normalizing flow in the forward direction.

        Parameters
        ----------
        x : torch.Tensor
            Input coordinates for the flow.


        Returns
        -------
        y : torch.Tensor
            Mapped coordinates for the flow.
        log_det_J : torch.Tensor
            Logarithm of the absolute value of the Jacobian determinant.

        """
        # Continuous flows return also the regularization, which is important only for training.
        return self._flow(x)[:2]

    def inverse(self, y):
        """Execute the normalizing flow in the inverse direction.

        Parameters
        ----------
        y : torch.Tensor
            Mapped coordinates for the flow.


        Returns
        -------
        x : torch.Tensor
            Input coordinates for the flow.
        log_det_J : torch.Tensor
            Logarithm of the absolute value of the Jacobian determinant.

        """
        # Continuous flows return also the regularization, which is important only for training.
        return self._flow.inverse(y)[:2]

    def training_step(self, batch, batch_idx):
        """Lightning method.

        Execute a training step.

        """
        x = batch['positions']

        # Forward.
        result = self._flow(x)

        # Continuous flows also return a regularization term.
        try:
            y, log_det_J = result
            reg = None
        except ValueError:
            y, log_det_J, reg = result

        # Compute potentials and loss.
        try:
            potential_y = self._potential_energy_func(y, batch['dimensions'])
        except KeyError:
            # There are no box vectors.
            potential_y = self._potential_energy_func(y)

        # Convert potentials to units of kT.
        potential_y = potential_y / self._kT

        # Convert bias to units of kT.
        try:
            log_weights = batch['log_weights']
        except KeyError:  # Unbiased simulation.
            try:
                log_weights = batch['bias'] / self._kT
            except KeyError:
                log_weights = None

        # Compute loss.
        loss = self._loss_func(target_potentials=potential_y, log_det_J=log_det_J, log_weights=log_weights)

        # Add regularization for continuous flows.
        if reg is not None:
            loss = loss + reg.mean()

        # Log potentials.
        self._tfep_logger.save_train_tensors(
            tensors={
                'dataset_sample_index': batch['dataset_sample_index'],
                'trajectory_sample_index': batch['trajectory_sample_index'],
                'potential': potential_y,
                'log_det_J': log_det_J,
            },
            epoch_idx=self.trainer.current_epoch,
            batch_idx=batch_idx,
        )

        # Log loss.
        self._tfep_logger.save_train_metrics(
            tensors={'loss': loss},
            epoch_idx=self.trainer.current_epoch,
            batch_idx=batch_idx,
        )

        return loss

    def train_dataloader(self):
        """Lightning method.

        Returns
        -------
        data_loader : torch.utils.data.DataLoader
            The training data loader.

        """
        # If this was loaded from a checkpoint, we need to restore the state of the batch sampler.
        if isinstance(self._stateful_batch_sampler, dict):
            batch_sampler_state = self._stateful_batch_sampler
        else:
            batch_sampler_state = None

        # Initialize the batch sampler for a correct mid-epoch resuming.
        self._stateful_batch_sampler = tfep.io.StatefulBatchSampler(
            self.dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            drop_last=False,
            trainer=self.trainer,
        )
        if batch_sampler_state is not None:
            self._stateful_batch_sampler.load_state_dict(batch_sampler_state)

        # Create the training dataloader.
        if self._dataloader_kwargs is None:
            dataloader_kwargs = {}
        else:
            dataloader_kwargs = self._dataloader_kwargs

        data_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_sampler=self._stateful_batch_sampler,
            **dataloader_kwargs,
        )

        # Initialize the TFEPLogger.
        self._tfep_logger = tfep.io.TFEPLogger(
            save_dir_path=self._tfep_logger_dir_path,
            data_loader=data_loader,
        )

        return data_loader

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]):
        """Lightning hook.

        Used to restore the state of the batch sampler for mid-epoch resuming.

        """
        # Normally, if this is a resumed training after a crash, StatefulBatchSampler
        # won't be initialized at this point, so we just save the state.
        self._stateful_batch_sampler = checkpoint['stateful_batch_sampler']

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        """Lightning hook.

        Used to store the state of the batch sampler for mid-epoch resuming.

        """
        checkpoint['stateful_batch_sampler'] = self._stateful_batch_sampler.state_dict()

    def _get_selected_indices(self, selection: Union[str, int, Sequence[int]], sort: bool = True) -> torch.Tensor:
        """Return selected indices as a sorted Tensor of integers.

        This does not set the device of the returned tensor.

        Parameters
        ----------
        selection : str, int, or array-like of ints
            An MDAnalysis string selection, an integer, or a sequence of integers
            representing the indices of the atoms.
        sort : bool, optional
            If ``True``, the returned atom indices are sorted.

        Returns
        -------
        atom_indices : torch.Tensor
            The atom indices of the selection.

        """
        if isinstance(selection, str):
            selection = self.dataset.universe.select_atoms(selection).ix
        if not torch.is_tensor(selection):
            selection = torch.tensor(selection)
        if selection.numel() == 0:
            raise ValueError('Selection contains 0 atoms.')
        if sort:
            selection = selection.sort()[0]
        return selection

    def _get_nonfixed_indices(
            self,
            atom_indices: torch.Tensor,
            idx_type: Literal['atom', 'dof'],
            remove_fixed: bool,
    ) -> torch.Tensor:
        """Return the atom/DOFs indices.

        atom_indices should be either self._mapped_atom_indices or self._conditioning_atom_indices.

        """
        # Shortcuts.
        assert idx_type in {"atom", "dof"}
        is_dof = idx_type == "dof"

        # Returned value (could be atom or dof indices).
        indices = atom_indices

        # We search the fixed indices by atom index so that searchsorted takes
        # three times less and we expect the fixed indices (when present) to be
        # the most numerous.
        if remove_fixed and self.n_fixed_atoms > 0:
            indices = indices - torch.searchsorted(self._fixed_atom_indices, indices)

        # Convert to DOF indices.
        if is_dof:
            indices = atom_to_flattened_indices(indices)

        return indices
