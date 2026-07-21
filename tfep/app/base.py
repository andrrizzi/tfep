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
import math
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import lightning
import MDAnalysis
import numpy as np
import pint
import torch

import tfep.loss
import tfep.io.sampler
from tfep.utils.misc import atom_to_flattened_indices, remove_and_shift_sorted_indices
from tfep.io.log import TMBARLogger


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
    ...             degrees_in=tfep.nn.conditioners.generate_degrees(
    ...                 n_features=self.n_nonfixed_dofs, conditioning_indices=conditioning_indices)
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
        return torch.optim.AdamW(self.parameters())

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
        n_conditioning_dofs = 3 * self.n_conditioning_atoms

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

    def get_nonfixed_indices(
            self,
            idx_type: Literal['atom', 'dof'],
            remove_fixed: bool,
    ) -> torch.Tensor:
        """Return the indices of the mapped and conditioning atom or degrees of freedom (DOF).

        This is a more efficient way of obtaining all mapped and conditioning
        indices tha concatenating and sorting the results of
        :func:`.TFEPMapBase.get_mapped_indices` and
        :func:`.TFEPMapBase.get_conditioning_indices`.

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
            return self.get_mapped_indices(idx_type=idx_type, remove_fixed=remove_fixed)

        # Merge and sort atom indices before removing the fixed ones (which
        # requires the indices to be sorted).
        nonfixed_atom_indices = torch.cat([
            self._mapped_atom_indices,
            self._conditioning_atom_indices
        ]).sort().values
        return self._get_nonfixed_indices(nonfixed_atom_indices, idx_type, remove_fixed)

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
                    # Reference atoms are conditioning or mapped. We can set remove=False.
                    reference_atom_indices[i] = remove_and_shift_sorted_indices(
                        atom_indices, removed_indices=self._fixed_atom_indices, remove=False)

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

    def forward(self, batch: Dict) -> dict[str, torch.Tensor]:
        """Execute the normalizing flow in the forward direction.

        Parameters
        ----------
        batch : dict[str, torch.Tensor]
            Batch data. Must have the key ``'positions'``.

        Returns
        -------
        result : dict[str, torch.Tensor]
            The output of the normalizing flow with at least the following keys:

            - ``'positions'``: Shape ``(batch_size, n_atoms*3)``. The mapped
              coordinates of the flow.
            - ``'log_det_J'``: Shape ``(batch_size,)``. The log weight of the
              transformation. This is usually the logarithm of the absolute value
              of the Jacobian determinant for deterministic maps, but it can be
              also the trace for continuous flows, or the log-ratio of the
              forward and backward paths in stochastic flows.
            - ``'regularization'``: Shape ``(batch_size,)``. Optional. Arbitrary
              regularization terms. The mean across batches is added to the loss.

            Any other tensor in this dictionary that is a scalar or of shape
            ``(batch_size,)`` is automatically logged. Any other tensor is ignored.

        """
        out = self._flow(batch['positions'])
        result = dict(positions=out[0], log_det_J=out[1])
        # Continuous flows also return a regularization term.
        if len(out) > 2:
            result['regularization'] = out[2]
        return result

    def inverse(self, batch: Dict) -> dict[str, torch.Tensor]:
        """Execute the normalizing flow in the inverse direction.

        See :func:`.TFEPMapBase.forward` for the documentation on the input
        parameters and returned value.

        """
        # Continuous flows return also the regularization, which is important only for training.
        out = self._flow.inverse(batch['positions'])
        result = dict(positions=out[0], log_det_J=out[1])
        # Continuous flows also return a regularization term.
        if len(out) > 2:
            result['regularization'] = out[2]
        return result

    def training_step(self, batch, batch_idx):
        """Lightning method.

        Execute a training step.

        """
        # Forward.
        result = self(batch)

        # Compute potentials and loss.
        try:
            potential = self._potential_energy_func(result['positions'], batch['dimensions'])
        except KeyError:
            # There are no box vectors.
            potential = self._potential_energy_func(result['positions'])

        # Convert potentials to units of kT.
        potential = potential / self._kT

        # Convert bias to units of kT.
        try:
            log_weights = batch['log_weights']
        except KeyError:
            try:
                log_weights = batch['bias'] / self._kT
            except KeyError:  # Unbiased simulation.
                log_weights = None

        # Compute loss.
        loss = self._loss_func(
            target_potentials=potential,
            log_det_J=result['log_det_J'],
            log_weights=log_weights,
        )

        # Add regularization for continuous flows.
        if 'regularization' in result:
            loss = loss + result['regularization'].mean()

        # Log potentials.
        self._tfep_logger.save_train_tensors(
            tensors={
                'dataset_sample_index': batch['dataset_sample_index'],
                'trajectory_sample_index': batch['trajectory_sample_index'],
                'potential': potential,
                # Save here any other tensor of shape (batch_size,).
                **{k: v for k, v in result.items() if v.shape == result['log_det_J'].shape},
            },
            epoch_idx=self.trainer.current_epoch,
            batch_idx=batch_idx,
        )

        # Log loss.
        self.log('loss', loss)

        # Log any other scalar tensor.
        for k, v in result.items():
            if len(v.shape) == 0:
                self.log(k, v)

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

    def _get_selected_indices(
            self,
            selection: Union[str, int, Sequence[int]],
            sort: bool = True,
            allow_empty: bool = False,
    ) -> torch.Tensor:
        """Return selected indices as a sorted Tensor of (unique) integers.

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
        if selection.numel() == 0 and not allow_empty:
            raise ValueError('Selection contains 0 atoms.')
        # Remove duplicated values. torch.Tensor.unique() always sorts on the
        # CUDA and CPU platforms, even if unique(sorted=False).
        if sort:
            selection = selection.unique()
        elif selection.numel() > 1:
            selection = selection.detach().numpy()
            sorter = np.unique(selection, return_index=True)[1]
            selection = torch.tensor(selection[np.sort(sorter)])
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

        # We remove the atom indices (not DOF indices) which is 3 times faster.
        if remove_fixed and self.n_fixed_atoms > 0:
            # We already know that atom_indices do not contain any fixed atom
            # index so we just need to shift them.
            indices = remove_and_shift_sorted_indices(
                indices, removed_indices=self._fixed_atom_indices, remove=False)

        # Convert to DOF indices.
        if is_dof:
            indices = atom_to_flattened_indices(indices)

        return indices




# =============================================================================
# TMBAR HELPERS
# =============================================================================

class _LenDataset:
    """A minimal object implementing only ``__len__``.

    Used to provide a ``dataset`` attribute for dataloaders that are not real
    ``torch.utils.data.DataLoader`` instances (e.g., combined loaders).
    """

    def __init__(self, n: int):
        self._n = int(n)

    def __len__(self):
        return self._n


class CombinedDataLoaderCompat:
    """A minimal combined dataloader that exposes metadata used by TFEP/TMBAR loggers.

    Lightning supports returning arbitrary iterables from ``train_dataloader``.
    However, :class:`tfep.io.TMBARLogger` expects the object passed as
    ``data_loader`` to expose ``batch_size``, ``drop_last``, and ``dataset``.

    This wrapper yields a dict with keys ``'batch_1'`` and ``'batch_2'`` and
    iterates for ``min(len(loader0), len(loader1))`` batches.
    """

    def __init__(self, loader0, loader1):
        self.loader0 = loader0
        self.loader1 = loader1

        # Prefer explicit batch_size; fall back to batch_sampler metadata.
        bs = getattr(loader0, 'batch_size', None)
        if bs is None and getattr(loader0, 'batch_sampler', None) is not None:
            bs = getattr(loader0.batch_sampler, 'batch_size', None)
        self.batch_size = bs

        drop_last = getattr(loader0, 'drop_last', None)
        if drop_last is None and getattr(loader0, 'batch_sampler', None) is not None:
            drop_last = getattr(loader0.batch_sampler, 'drop_last', None)
        self.drop_last = bool(drop_last)

        # The logger uses len(dataset) to compute buffer sizes; use the min.
        n0 = len(getattr(loader0, 'dataset', _LenDataset(len(loader0))))
        n1 = len(getattr(loader1, 'dataset', _LenDataset(len(loader1))))
        self.dataset = _LenDataset(min(n0, n1))

    def __len__(self):
        return min(len(self.loader0), len(self.loader1))

    def __iter__(self):
        it0 = iter(self.loader0)
        it1 = iter(self.loader1)
        for _ in range(len(self)):
            yield {'batch_1': next(it0), 'batch_2': next(it1)}


# =============================================================================
# TMBAR MAP BASE CLASS
# =============================================================================

class TMBARMapBase(TFEPMapBase):
    """Base class for bidirectional TFEP maps logged with :class:`tfep.io.TMBARLogger`.

    This class extends :class:`~tfep.app.base.TFEPMapBase` to support training a
    **single** invertible model in both directions (0→1 and 1→0) using two
    independent datasets (typically trajectories sampled at two different
    potentials).

    Key guarantees (to match the historical monolithic scripts):
    - Correct **target** potential is used for each direction.
    - The combined dataloader exposes metadata required by :class:`TMBARLogger`.
    - Batch samplers use ``drop_last=True`` so logger preallocation matches
      the number of samples actually produced.

    To compute generalized works outside the model, you typically use the saved
    tensors ``potential`` (destination reduced potential) and ``log_det_J`` and
    subtract the appropriate source reduced potentials computed on the original
    frames.
    """

    def __init__(
        self,
        potential_energy_func: torch.nn.Module,
        topology_file_path: str,
        coordinates_file_path: Union[str, Sequence[str]],
        coordinates_file_path_2: Union[str, Sequence[str]],
        temperature: pint.Quantity,
        batch_size: int = 1,
        mapped_atoms: Optional[Union[Sequence[int], str]] = None,
        conditioning_atoms: Optional[Union[Sequence[int], str]] = None,
        origin_atom: Optional[Union[int, str]] = None,
        axes_atoms: Optional[Union[Sequence[int], str]] = None,
        tfep_logger_dir_path: str = 'tfep_logs',
        dataloader_kwargs: Optional[Dict] = None,
        n_states: int = 2,
        state_names: Optional[list[str]] = None,
        bar_regularizer: Optional[object] = None,
        *,
        objective: Literal['kl', 'bar', 'hybrid'] = 'kl',
        lambda_bar: float = 1.0,
        bar_detach_df: bool = True,
        bar_warm_start: bool = True,
        bar_max_iter: int = 25,
        bar_tol: float = 1e-10,
        bar_df_solver: Literal['newton', 'robust'] = 'newton',
        logJ_penalty_weight: float = 0.0,
        allow_reweighted_bar: bool = False,
        reweighted_bar_weight_normalization: Literal['minibatch', 'global'] = 'minibatch',
        reweighted_bar_global_stats: Optional[Dict[str, Dict[str, float]]] = None,
        stochastic_training_config: Optional[Any] = None,
        **kwargs,
    ):
        super().__init__(
            potential_energy_func=potential_energy_func,
            topology_file_path=topology_file_path,
            coordinates_file_path=coordinates_file_path,
            temperature=temperature,
            batch_size=batch_size,
            mapped_atoms=mapped_atoms,
            conditioning_atoms=conditioning_atoms,
            origin_atom=origin_atom,
            axes_atoms=axes_atoms,
            tfep_logger_dir_path=tfep_logger_dir_path,
            dataloader_kwargs=dataloader_kwargs,
        )

        if n_states != 2:
            raise ValueError(
                'TMBARMapBase currently supports exactly two states (n_states=2). '
                'For >2 states, use/implement a multi-dataset variant.'
            )

        self.n_states = int(n_states)
        self.state_names = state_names if state_names is not None else [f'state_{i}' for i in range(self.n_states)]
        self.state_mappings = [(0, 1), (1, 0)]

        self._coordinates_file_path_2 = coordinates_file_path_2
        self.dataset_2 = None

        self._tmb_logger: Optional[TMBARLogger] = None

        # Mid-epoch resuming: we need one sampler per dataset.
        self._stateful_batch_sampler_0 = None
        self._stateful_batch_sampler_1 = None

        # Optional minibatch regularizer (e.g., BAR-like consistency term).
        self._bar_regularizer = bar_regularizer

        # Keep extra kwargs for child classes.
        self._extra_kwargs = kwargs

        self._objective = str(objective)

        if self._objective not in ('kl', 'bar', 'hybrid'):
            raise ValueError(
                f"Unsupported objective {self._objective!r}. "
                "Expected one of: 'kl', 'bar', 'hybrid'."
            )

        self._lambda_bar = float(lambda_bar)
        self._bar_detach_df = bool(bar_detach_df)
        self._bar_warm_start = bool(bar_warm_start)
        self._bar_max_iter = int(bar_max_iter)
        self._bar_tol = float(bar_tol)
        self._bar_df_solver = str(bar_df_solver).lower()
        if self._bar_df_solver not in ('newton', 'robust'):
            raise ValueError(
                f"Unsupported bar_df_solver {bar_df_solver!r}. "
                "Expected one of: 'newton', 'robust'."
            )
        self._logJ_penalty_weight = float(logJ_penalty_weight)
        self._allow_reweighted_bar = bool(allow_reweighted_bar)
        self._reweighted_bar_weight_normalization = str(reweighted_bar_weight_normalization).lower()
        if self._reweighted_bar_weight_normalization not in ('minibatch', 'global'):
            raise ValueError(
                "Unsupported reweighted_bar_weight_normalization "
                f"{reweighted_bar_weight_normalization!r}. Expected 'minibatch' or 'global'."
            )
        self._reweighted_bar_global_stats = {
            str(key): dict(value) for key, value in (reweighted_bar_global_stats or {}).items()
        }
        from tfep.stochastic.training import StochasticTrainingConfig
        self._stochastic_training_config = StochasticTrainingConfig.from_any(stochastic_training_config)
        self._stochastic_training_config.validate()
        if self._stochastic_training_config.enabled and self._objective not in ('bar', 'hybrid'):
            raise ValueError(
                "stochastic_training_config.enabled=True requires objective='bar' or objective='hybrid'."
            )
        self.register_buffer('_last_df_bar_obj', torch.zeros(()), persistent=False)

        if self._objective in ('bar', 'hybrid') and bar_regularizer is not None:
            raise ValueError(
                "bar_regularizer is set but objective is 'bar'/'hybrid'. "
                "This would double-count a BAR-like term. Set bar_regularizer=None."
            )
        self._bar_regularizer = bar_regularizer
    
    def setup(self, stage: str = 'fit'):
        """Lightning method. Create both datasets and the flow."""
        self.dataset = self.create_dataset()
        self.dataset_2 = self.create_dataset_2()
        if self._reweighted_bar_weight_normalization == 'global':
            self._complete_reweighted_bar_global_stats()

        # Identify mapped/conditioning/fixed indices using the first dataset.
        self.determine_atom_indices()

        flow = self.configure_flow()
        self._flow = self.create_partial_flow(flow)

    @staticmethod
    def _uniform_reweighted_bar_stats(population_size: int) -> Dict[str, float]:
        n = int(population_size)
        if n <= 0:
            raise ValueError("Global weighted BAR requires a positive training population")
        return {
            'population_size': n,
            'log_normalizer': float(math.log(float(n))),
            'ess': float(n),
            'ess_ratio': 1.0,
            'weighted': False,
        }

    def _complete_reweighted_bar_global_stats(self) -> None:
        """Validate global constants against the two actual training datasets."""
        for key, dataset in (('state0', self.dataset), ('state1', self.dataset_2)):
            n_dataset = int(len(dataset))
            stats = self._reweighted_bar_global_stats.get(key)
            if stats is None:
                stats = self._uniform_reweighted_bar_stats(n_dataset)
                self._reweighted_bar_global_stats[key] = stats
            n_stats = int(stats.get('population_size', -1))
            if n_stats != n_dataset:
                raise ValueError(
                    f"Global weighted BAR {key} population size {n_stats} does not match "
                    f"the training dataset size {n_dataset}"
                )
            for field in ('log_normalizer', 'ess'):
                value = float(stats.get(field, float('nan')))
                if not math.isfinite(value) or (field == 'ess' and value <= 0.0):
                    raise ValueError(f"Invalid global weighted BAR {key} {field}: {value}")

    def _global_reweighted_bar_terms(self, *, device, dtype) -> Optional[Dict[str, Any]]:
        if self._reweighted_bar_weight_normalization != 'global':
            return None
        missing = [key for key in ('state0', 'state1') if key not in self._reweighted_bar_global_stats]
        if missing:
            raise RuntimeError(
                "Global weighted BAR statistics were not initialized for " + ", ".join(missing)
            )
        state0 = self._reweighted_bar_global_stats['state0']
        state1 = self._reweighted_bar_global_stats['state1']
        ess0 = torch.as_tensor(float(state0['ess']), device=device, dtype=dtype)
        ess1 = torch.as_tensor(float(state1['ess']), device=device, dtype=dtype)
        return {
            # ESS is a precision diagnostic, not an additive free-energy term.
            'log_ratio': torch.zeros((), device=device, dtype=dtype),
            'log_weight_normalizer_forward': torch.as_tensor(
                float(state0['log_normalizer']), device=device, dtype=dtype
            ),
            'log_weight_normalizer_reverse': torch.as_tensor(
                float(state1['log_normalizer']), device=device, dtype=dtype
            ),
            'population_size_forward': int(state0['population_size']),
            'population_size_reverse': int(state1['population_size']),
            'ess_forward': ess0,
            'ess_reverse': ess1,
        }

    def create_dataset_2(self):
        universe = self.create_universe_2()
        return tfep.io.TrajectoryDataset(universe=universe)

    def create_universe_2(self):
        if isinstance(self._coordinates_file_path_2, str):
            coords = [self._coordinates_file_path_2]
        else:
            coords = list(self._coordinates_file_path_2)
        return MDAnalysis.Universe(self._topology_file_path, *coords)

    @staticmethod
    def _strip_batch_sampler_kwargs(d: Dict) -> Dict:
        """Remove DataLoader kwargs that conflict with passing ``batch_sampler``."""
        out = d.copy()
        for bad in ('batch_size', 'shuffle', 'sampler', 'drop_last', 'batch_sampler'):
            out.pop(bad, None)
        return out

    def train_dataloader(self):
        """Lightning method. Return a combined loader (state0/state1) with logger metadata."""
        # Restore sampler states if loaded from checkpoint.
        state0 = self._stateful_batch_sampler_0 if isinstance(self._stateful_batch_sampler_0, dict) else None
        state1 = self._stateful_batch_sampler_1 if isinstance(self._stateful_batch_sampler_1, dict) else None

        dataloader_kwargs = {} if self._dataloader_kwargs is None else self._strip_batch_sampler_kwargs(self._dataloader_kwargs)

        self._stateful_batch_sampler_0 = tfep.io.StatefulBatchSampler(
            self.dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            drop_last=True,
            trainer=self.trainer,
        )
        self._stateful_batch_sampler_1 = tfep.io.StatefulBatchSampler(
            self.dataset_2,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            drop_last=True,
            trainer=self.trainer,
        )
        if state0 is not None:
            self._stateful_batch_sampler_0.load_state_dict(state0)
        if state1 is not None:
            self._stateful_batch_sampler_1.load_state_dict(state1)

        loader0 = torch.utils.data.DataLoader(self.dataset, batch_sampler=self._stateful_batch_sampler_0, **dataloader_kwargs)
        loader1 = torch.utils.data.DataLoader(self.dataset_2, batch_sampler=self._stateful_batch_sampler_1, **dataloader_kwargs)

        combined_loader = CombinedDataLoaderCompat(loader0, loader1)

        self._tmb_logger = TMBARLogger(
            save_dir_path=self._tfep_logger_dir_path,
            data_loader=combined_loader,
            n_states=self.n_states,
            state_names=self.state_names,
        )

        return combined_loader

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]):
        """Lightning hook. Used to restore batch sampler state for mid-epoch resuming."""
        # Backward compatibility: accept either the new two-sampler key or the old single one.
        if 'stateful_batch_samplers' in checkpoint:
            s = checkpoint['stateful_batch_samplers']
            self._stateful_batch_sampler_0 = s.get('state0')
            self._stateful_batch_sampler_1 = s.get('state1')
        else:
            # Old checkpoints (if any) stored a single sampler; treat it as state0.
            self._stateful_batch_sampler_0 = checkpoint.get('stateful_batch_sampler', None)
            self._stateful_batch_sampler_1 = None

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        """Lightning hook. Store batch sampler state for mid-epoch resuming."""
        checkpoint['stateful_batch_samplers'] = {
            'state0': (self._stateful_batch_sampler_0.state_dict() if hasattr(self._stateful_batch_sampler_0, 'state_dict') else self._stateful_batch_sampler_0),
            'state1': (self._stateful_batch_sampler_1.state_dict() if hasattr(self._stateful_batch_sampler_1, 'state_dict') else self._stateful_batch_sampler_1),
        }

    def _eval_potential(self, state: int, positions: torch.Tensor, dimensions: Optional[torch.Tensor]):
        """Evaluate the potential for a specific state.

        Supports:
        - modules exposing ``energy(state, positions, dimensions=None)``;
        - indexable containers of per-state potentials (list/tuple/dict);
        - plain single-state potentials (state is ignored).
        """
        pot = self._potential_energy_func

        if hasattr(pot, 'energy'):
            try:
                return pot.energy(state, positions, dimensions)
            except TypeError:
                try:
                    return pot.energy(state, positions)
                except TypeError:
                    return pot.energy(positions)

        # Indexable container of potentials.
        if isinstance(pot, (list, tuple)):
            pot_state = pot[int(state)]
        elif isinstance(pot, dict):
            pot_state = pot[int(state)]
        else:
            pot_state = pot

        if dimensions is None:
            return pot_state(positions)
        try:
            return pot_state(positions, dimensions)
        except TypeError:
            return pot_state(positions)

    def _compute_direction_step(self, batch_data, direction_func, state_mapping, batch_idx):
        """Compute loss and log tensors for a single direction."""
        result = direction_func(batch_data)
        from_state, to_state = int(state_mapping[0]), int(state_mapping[1])
        dims = batch_data.get('dimensions', None)

        u_to = self._eval_potential(to_state, result['positions'], dims) / self._kT
        log_det_J = result['log_det_J']

        # Optional: for BAR-like minibatch penalties.
        u_from = None
        need_u_from = (self._bar_regularizer is not None) or (self._objective in ('bar', 'hybrid'))
        if need_u_from:
            u_from = self._eval_potential(from_state, batch_data['positions'], dims) / self._kT


        # Convert bias to units of kT.
        if 'log_weights' in batch_data:
            log_weights = batch_data['log_weights']
        elif 'bias' in batch_data:
            log_weights = batch_data['bias'] / self._kT
        else:
            log_weights = None

        loss = self._loss_func(
            target_potentials=u_to,
            log_det_J=log_det_J,
            log_weights=log_weights,
        )

        # Add regularization for continuous flows.
        if 'regularization' in result:
            loss = loss + result['regularization'].mean()

        # Log tensors.
        if self._tmb_logger is not None:
            self._tmb_logger.save_train_tensors(
                tensors={
                    'dataset_sample_index': batch_data['dataset_sample_index'],
                    'trajectory_sample_index': batch_data['trajectory_sample_index'],
                    'potential': u_to,
                    'log_det_J': log_det_J,
                    **({'log_weights': log_weights} if log_weights is not None else {}),
                    **{k: v for k, v in result.items() if hasattr(v, 'shape') and v.shape == log_det_J.shape},
                },
                epoch_idx=self.trainer.current_epoch,
                batch_idx=batch_idx,
                state_mapping=state_mapping,
            )

        self.log(f'loss_{from_state}_{to_state}', loss)
        return loss, u_to, log_det_J, u_from, result

    @staticmethod
    def _compute_bidirectional_works(
            u1_y0: torch.Tensor,
            logJ01: torch.Tensor,
            u0_x0: torch.Tensor,
            u0_y1: torch.Tensor,
            logJ10: torch.Tensor,
            u1_x1: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return forward/reverse generalized works in dimensionless kT units."""
        w01 = (u1_y0 - logJ01) - u0_x0
        w10 = (u0_y1 - logJ10) - u1_x1
        return w01, w10

    @staticmethod
    def _log_sample_ratio(n0: int, n1: int, *, device, dtype) -> torch.Tensor:
        """Return ``log(n1 / n0)`` as a tensor in the target device/dtype."""
        return torch.log(
            torch.as_tensor(float(n1), device=device, dtype=dtype)
            / torch.as_tensor(float(n0), device=device, dtype=dtype)
        )

    def _bar_objective_term(
            self,
            *,
            w01: torch.Tensor,
            w10: torch.Tensor,
            log_weights01: Optional[torch.Tensor] = None,
            log_weights10: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Any]]:
        """Compute BAR objective and detached df estimate for objective=bar/hybrid."""
        from tfep.regularizers import bar as barlib
        from tfep.analysis import reweighting as rwlib

        n0 = int(w01.numel())
        n1 = int(w10.numel())
        log_ratio = self._log_sample_ratio(n0, n1, device=w01.device, dtype=w01.dtype)
        weighted = log_weights01 is not None or log_weights10 is not None

        if weighted:
            global_terms = self._global_reweighted_bar_terms(device=w01.device, dtype=w01.dtype)
            log_ratio = (
                torch.zeros((), device=w01.device, dtype=w01.dtype)
                if global_terms is None else global_terms['log_ratio']
            )
            global_kwargs = {} if global_terms is None else {
                key: global_terms[key] for key in (
                    'log_weight_normalizer_forward',
                    'log_weight_normalizer_reverse',
                    'population_size_forward',
                    'population_size_reverse',
                )
            }

            df_init = self._last_df_bar_obj if self._bar_warm_start else None
            solver_info = None
            if self._bar_df_solver == 'robust':
                solver_info = rwlib.weighted_bar_robust_solve_detached(
                    w01.detach(),
                    w10.detach(),
                    log_weights_forward=log_weights01.detach() if log_weights01 is not None else None,
                    log_weights_reverse=log_weights10.detach() if log_weights10 is not None else None,
                    log_ratio=log_ratio.detach(),
                    df_init=df_init.detach() if df_init is not None else None,
                    max_iter=self._bar_max_iter,
                    tol=self._bar_tol,
                    **global_kwargs,
                )
                df_bar = solver_info.df.detach()
            else:
                df_bar = rwlib.weighted_bar_newton_solve_detached(
                    w01.detach(),
                    w10.detach(),
                    log_weights_forward=log_weights01.detach() if log_weights01 is not None else None,
                    log_weights_reverse=log_weights10.detach() if log_weights10 is not None else None,
                    log_ratio=log_ratio.detach(),
                    df_init=df_init.detach() if df_init is not None else None,
                    max_iter=self._bar_max_iter,
                    tol=self._bar_tol,
                    **global_kwargs,
                ).detach()
            self._last_df_bar_obj = df_bar

            df_for_obj = df_bar.detach() if self._bar_detach_df else df_bar
            bar_obj = rwlib.weighted_bar_objective(
                w01,
                w10,
                df_for_obj,
                log_ratio,
                log_weights_forward=log_weights01,
                log_weights_reverse=log_weights10,
                **global_kwargs,
            )
            return bar_obj, df_bar, solver_info

        # Detached Newton solve (optionally warm-started).
        df_init = self._last_df_bar_obj if self._bar_warm_start else None
        solver_info = None
        if self._bar_df_solver == 'robust':
            solver_info = rwlib.weighted_bar_robust_solve_detached(
                w01.detach(),
                w10.detach(),
                log_ratio=log_ratio.detach(),
                df_init=df_init.detach() if df_init is not None else None,
                max_iter=self._bar_max_iter,
                tol=self._bar_tol,
            )
            df_bar = solver_info.df.detach()
        else:
            df_bar = barlib._bar_newton_solve_detached(
                w01.detach(),
                w10.detach(),
                log_ratio=log_ratio.detach(),
                df_init=df_init.detach() if df_init is not None else None,
                max_iter=self._bar_max_iter,
                tol=self._bar_tol,
            ).detach()
        self._last_df_bar_obj = df_bar

        df_for_obj = df_bar.detach() if self._bar_detach_df else df_bar
        bar_obj = barlib.bar_objective(w01, w10, df_for_obj, log_ratio)
        return bar_obj, df_bar, solver_info

    def _jacobian_penalty_term(self, logJ01: torch.Tensor, logJ10: torch.Tensor) -> torch.Tensor:
        """Quadratic penalty guarding against runaway Jacobian magnitudes."""
        logJ_sq = 0.5 * (logJ01.pow(2).mean() + logJ10.pow(2).mean())
        return torch.as_tensor(
            self._logJ_penalty_weight,
            device=logJ_sq.device,
            dtype=logJ_sq.dtype,
        ) * logJ_sq

    def _log_finite_tensor_stats(self, name: str, values: torch.Tensor) -> None:
        """Log passive diagnostics without altering the training objective."""
        flat = torch.as_tensor(values).detach().reshape(-1)
        finite = flat[torch.isfinite(flat)]
        if finite.numel() == 0:
            nan = torch.as_tensor(float("nan"), device=flat.device, dtype=flat.dtype)
            self.log(f"{name}_mean", nan)
            self.log(f"{name}_std", nan)
            self.log(f"{name}_max_abs", nan)
            return
        self.log(f"{name}_mean", finite.mean())
        self.log(f"{name}_std", finite.std(unbiased=False))
        self.log(f"{name}_max_abs", finite.abs().max())

    def _log_weight_training_diagnostics(
            self,
            name_suffix: str,
            log_weights: Optional[torch.Tensor],
            *,
            n_samples: int,
            device: torch.device,
            dtype: torch.dtype,
    ) -> None:
        """Log ESS/span diagnostics for biased-training weights."""
        if log_weights is None:
            ess = torch.as_tensor(float(n_samples), device=device, dtype=dtype)
            ess_ratio = torch.as_tensor(1.0, device=device, dtype=dtype)
            span = torch.as_tensor(0.0, device=device, dtype=dtype)
        else:
            logw = torch.as_tensor(log_weights, device=device, dtype=dtype).detach().reshape(-1)
            finite = logw[torch.isfinite(logw)]
            if finite.numel() == 0:
                ess = torch.as_tensor(float("nan"), device=device, dtype=dtype)
                ess_ratio = torch.as_tensor(float("nan"), device=device, dtype=dtype)
                span = torch.as_tensor(float("nan"), device=device, dtype=dtype)
            else:
                norm = finite - torch.logsumexp(finite, dim=0)
                weights = torch.exp(norm)
                ess = 1.0 / torch.sum(weights * weights)
                ess_ratio = ess / torch.as_tensor(float(finite.numel()), device=device, dtype=dtype)
                span = finite.max() - finite.min()
        self.log(f"log_weight_ess{name_suffix}", ess)
        self.log(f"log_weight_ess_ratio{name_suffix}", ess_ratio)
        self.log(f"log_weight_span{name_suffix}", span)

    def _bar_log_ratio_for_diagnostics(
            self,
            w01: torch.Tensor,
            w10: torch.Tensor,
            log_weights01: Optional[torch.Tensor],
            log_weights10: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Return the balance offset used by BAR training diagnostics."""
        n0 = int(w01.numel())
        n1 = int(w10.numel())
        device = w01.device
        dtype = w01.dtype
        if log_weights01 is None and log_weights10 is None:
            return self._log_sample_ratio(n0, n1, device=device, dtype=dtype).detach()

        global_terms = self._global_reweighted_bar_terms(device=device, dtype=dtype)
        if global_terms is not None:
            return global_terms['log_ratio'].detach()

        return torch.zeros((), device=device, dtype=dtype)

    def _log_bar_training_diagnostics(
            self,
            *,
            w01: torch.Tensor,
            w10: torch.Tensor,
            df_bar: torch.Tensor,
            log_weights01: Optional[torch.Tensor],
            log_weights10: Optional[torch.Tensor],
    ) -> None:
        """Log BAR/work diagnostics without changing estimator or gradients."""
        self._log_finite_tensor_stats("w01", w01)
        self._log_finite_tensor_stats("w10", w10)

        has_reweighting = log_weights01 is not None or log_weights10 is not None
        if has_reweighting:
            self._log_weight_training_diagnostics(
                "01",
                log_weights01,
                n_samples=int(w01.numel()),
                device=w01.device,
                dtype=w01.dtype,
            )
            self._log_weight_training_diagnostics(
                "10",
                log_weights10,
                n_samples=int(w10.numel()),
                device=w10.device,
                dtype=w10.dtype,
            )
            global_terms = self._global_reweighted_bar_terms(device=w01.device, dtype=w01.dtype)
            if global_terms is not None:
                self.log('global_log_weight_ess01', global_terms['ess_forward'])
                self.log('global_log_weight_ess10', global_terms['ess_reverse'])
                self.log('global_bar_log_ratio', global_terms['log_ratio'])
                self.log(
                    'global_weight_normalization_enabled',
                    torch.as_tensor(1.0, device=w01.device, dtype=w01.dtype),
                )

        log_ratio = self._bar_log_ratio_for_diagnostics(w01, w10, log_weights01, log_weights10)
        df = torch.as_tensor(df_bar, device=w01.device, dtype=w01.dtype).detach().reshape(())
        arg01 = w01.detach().reshape(-1) - df - log_ratio
        arg10 = w10.detach().reshape(-1) + df + log_ratio

        def _finite_max_abs(values: torch.Tensor) -> torch.Tensor:
            finite = values[torch.isfinite(values)]
            if finite.numel() == 0:
                return torch.as_tensor(float("nan"), device=values.device, dtype=values.dtype)
            return finite.abs().max()

        self.log("bar_arg_forward_max_abs", _finite_max_abs(arg01))
        self.log("bar_arg_reverse_max_abs", _finite_max_abs(arg10))

    def _log_bar_solver_diagnostics(
            self,
            solver_info: Optional[Any],
            df_bar: torch.Tensor,
            *,
            device: torch.device,
            dtype: torch.dtype,
    ) -> None:
        """Log detached BAR solver diagnostics when the robust path is enabled."""
        if self._bar_df_solver != 'robust':
            return

        def _scalar(value: float) -> torch.Tensor:
            return torch.as_tensor(float(value), device=device, dtype=dtype)

        self.log("bar_df_value", torch.as_tensor(df_bar, device=device, dtype=dtype).detach().reshape(()))
        if solver_info is None:
            self.log("bar_df_solver_used_fallback", _scalar(float("nan")))
            self.log("bar_df_solver_converged", _scalar(float("nan")))
            self.log("bar_df_solver_iterations", _scalar(float("nan")))
            self.log("bar_df_solver_residual_abs", _scalar(float("nan")))
            return
        self.log("bar_df_solver_used_fallback", _scalar(1.0 if solver_info.used_fallback else 0.0))
        self.log("bar_df_solver_converged", _scalar(1.0 if solver_info.converged else 0.0))
        self.log("bar_df_solver_iterations", _scalar(float(solver_info.iterations)))
        self.log(
            "bar_df_solver_residual_abs",
            torch.as_tensor(solver_info.residual_abs, device=device, dtype=dtype).detach().reshape(()),
        )

    def training_step(self, batch, batch_idx):
        """Lightning method. Compute forward + reverse losses (and optional minibatch regularizer)."""
        step01 = self._compute_direction_step(batch['batch_1'], self.forward, (0, 1), batch_idx)
        step10 = self._compute_direction_step(batch['batch_2'], self.inverse, (1, 0), batch_idx)
        if len(step01) == 4:
            loss_01, u1_y0, logJ01, u0_x0 = step01
            result01 = None
        else:
            loss_01, u1_y0, logJ01, u0_x0, result01 = step01
        if len(step10) == 4:
            loss_10, u0_y1, logJ10, u1_x1 = step10
        else:
            loss_10, u0_y1, logJ10, u1_x1, _ = step10

        total_loss = 0.5 * (loss_01 + loss_10)
        
        
        if self._objective in ('bar', 'hybrid'):
            b0 = batch['batch_1']
            b1 = batch['batch_2']
            logw0 = b0.get('log_weights', None)
            if logw0 is None and 'bias' in b0:
                logw0 = b0['bias'] / self._kT
            logw1 = b1.get('log_weights', None)
            if logw1 is None and 'bias' in b1:
                logw1 = b1['bias'] / self._kT
            has_reweighting = logw0 is not None or logw1 is not None

            # Reweighted BAR is experimental and must be explicitly selected so
            # biased trajectories cannot silently enter the ordinary BAR path.
            if has_reweighting and not self._allow_reweighted_bar:
                raise NotImplementedError(
                    "BAR-matched objective currently assumes unbiased sampling in each state. "
                    "Pass --allow-reweighted-bar to enable the experimental weighted BAR objective."
                )

            # Safety if u_from wasn't returned for some reason
            if u0_x0 is None or u1_x1 is None:
                u0_x0 = self._eval_potential(0, b0['positions'], b0.get('dimensions', None)) / self._kT
                u1_x1 = self._eval_potential(1, b1['positions'], b1.get('dimensions', None)) / self._kT

            snf_training = None
            if self._stochastic_training_config.enabled:
                if result01 is None:
                    raise RuntimeError("Stochastic training requires _compute_direction_step to return mapped forward results.")
                from tfep.stochastic.training import compute_bidirectional_stochastic_training_works
                snf_training = compute_bidirectional_stochastic_training_works(
                    model=self,
                    batch0=b0,
                    batch1=b1,
                    mapped01=result01,
                    u0_x0=u0_x0,
                    u1_x1=u1_x1,
                    config=self._stochastic_training_config,
                )
                w01, w10 = snf_training.w01, snf_training.w10
                for metric_name, metric_value in snf_training.metrics().items():
                    self.log(metric_name, metric_value)
            else:
                w01, w10 = self._compute_bidirectional_works(
                    u1_y0=u1_y0,
                    logJ01=logJ01,
                    u0_x0=u0_x0,
                    u0_y1=u0_y1,
                    logJ10=logJ10,
                    u1_x1=u1_x1,
                )
            bar_obj, df_bar, solver_info = self._bar_objective_term(
                w01=w01,
                w10=w10,
                log_weights01=logw0,
                log_weights10=logw1,
            )
            self._log_bar_training_diagnostics(
                w01=w01,
                w10=w10,
                df_bar=df_bar,
                log_weights01=logw0,
                log_weights10=logw1,
            )
            self._log_bar_solver_diagnostics(
                solver_info,
                df_bar,
                device=bar_obj.device,
                dtype=bar_obj.dtype,
            )

            self.log('df_bar_obj', df_bar)
            self.log('bar_obj', bar_obj)
            if has_reweighting:
                self.log('reweighted_bar_enabled', torch.as_tensor(1.0, device=bar_obj.device, dtype=bar_obj.dtype))
            if snf_training is not None:
                self.log('snf_df_bar_obj', df_bar)
                self.log('snf_bar_obj', bar_obj)

            if self._objective == 'bar':
                total_loss = bar_obj
            else:  # hybrid
                total_loss = total_loss + torch.as_tensor(self._lambda_bar, device=bar_obj.device, dtype=bar_obj.dtype) * bar_obj

        # Jacobian penalty: penalise large |log det J| to prevent Jacobian hacking.
        # When the BAR/hybrid objective is used, the gradient w.r.t. log|J| is
        # antisymmetric between forward and reverse directions.  Without a guard
        # the optimizer can exploit this by driving <log|J|> to ±∞, collapsing
        # the overlap to zero while the per-direction work variances shrink.
        # A penalty lambda_J * mean(logJ^2) keeps the Jacobian bounded.
        if self._logJ_penalty_weight > 0.0:
            logJ_pen = self._jacobian_penalty_term(logJ01, logJ10)
            total_loss = total_loss + logJ_pen
            self.log('logJ_penalty', logJ_pen)
            self.log('logJ01_rms', logJ01.pow(2).mean().sqrt())
            self.log('logJ10_rms', logJ10.pow(2).mean().sqrt())

        # Optional BAR-like minibatch penalty.
        if self._bar_regularizer is not None:
            # u_from tensors were computed in _compute_direction_step when regularizer is enabled.
            if u0_x0 is None or u1_x1 is None:
                # Safety: compute if not provided.
                b0 = batch['batch_1']
                b1 = batch['batch_2']
                u0_x0 = self._eval_potential(0, b0['positions'], b0.get('dimensions', None)) / self._kT
                u1_x1 = self._eval_potential(1, b1['positions'], b1.get('dimensions', None)) / self._kT

            w01, w10 = self._compute_bidirectional_works(
                u1_y0=u1_y0,
                logJ01=logJ01,
                u0_x0=u0_x0,
                u0_y1=u0_y1,
                logJ10=logJ10,
                u1_x1=u1_x1,
            )
            reg = self._bar_regularizer(w01, w10)
            total_loss = total_loss + reg
            self.log('bar_reg', reg)

        self.log('loss', total_loss)
        return total_loss
        
