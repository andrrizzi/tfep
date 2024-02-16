#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""Base ``LightningModule`` class to implement TFEP maps."""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

from collections.abc import Sequence
from typing import Literal, Optional, Union

import pint
import torch

from tfep.app.base import TFEPMapBase
import tfep.nn.flows
from tfep.utils.misc import atom_to_flattened_indices


# =============================================================================
# TFEP MAP BASE CLASS
# =============================================================================

class CartesianMAFMap(TFEPMapBase):
    """A TFEP map using a masked autoregressive flow in Cartesian coordinates.

    The class divides the atoms of the entire system in mapped, conditioning,
    and fixed. Mapped atoms are defined as those that the flow maps. Conditioning
    atoms are not mapped but are given as input to the flow to condition the
    mapping. Fixed atoms are instead ignored.

    Optionally, the flow can map the atoms in a relative frame of reference
    which based on the position of an ``origin_atom`` and two ``axes_atoms``
    that determine the origin and the orientation of the axes, respectively.

    The class further supports logging the potential energies computed during
    training (required for the multimap TFEP analysis) and mid-epoch resuming.

    .. warning::

        Currently, this class is not multi-process or thread safe. Running with
        multiple processes may result in the corrupted logging of the potentials
        and Jacobians.

    See Also
    --------
    :class:`tfep.app.base.TFEPMapBase`
        The base class for TFEP maps with more detailed explanations on the
        division in mapped/conditioning/fixed atoms.

    Examples
    --------

    >>> from tfep.potentials.psi4 import Psi4Potential
    >>> units = pint.UnitRegistry()
    >>>
    >>> tfep_map = CartesianMAFMap(
    ...     potential_energy_func=Psi4Potential(name='mp2'),
    ...     topology_file_path='path/to/topology.psf',
    ...     coordinates_file_path='path/to/trajectory.dcd',
    ...     temperature=300*units.kelvin,
    ...     batch_size=64,
    ...     mapped_atoms='resname MOL',  # MDAnalysis selection syntax.
    ...     conditioning_atoms=range(10, 20),
    ...     origin_atom=12,  # Fix the origin of the relative reference frame on atom 123.
    ...     axes_atoms=[13, 16],  # Determine the orientation of the reference frame.
    ... )
    >>>
    >>> # Train the flow and save the potential energies.
    >>> import lightning
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
            n_maf_layers: int = 6,
            **kwargs,
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
        mapped_atoms : Sequence[int] or str, optional
            The indices (0-based) of the atoms to map or a selection string in
            MDAnalysis syntax. If not passed, all atoms that are not conditioning
            are mapped (i.e., all atoms are mapped if also ``conditioning_atoms``
            is not given.
        conditioning_atoms : Sequence[int] or str, optional
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
            for the two atoms determining the relative frame of reference. The
            ``axes_atoms[0]``-th atom will lay on the ``z`` axis , and the
            ``axes_atoms[1]``-th atom will lay on the plane spanned by the ``x``
            and ``z`` axes. The ``y`` axis will be set as the cross product of
            ``x`` and ``z``.

            These atoms can be either conditioning or mapped. ``axes_atoms[0]``
            has only 1 degree of freedom (DOF) while ``axes_atoms[1]`` has 2.
            Whether these DOFs are mapped or not depends on whether their atoms
            are indicated as mapped or conditioning, respectively.
        tfep_logger_dir_path : str, optional
            The path where to save TFEP-related information (potential energies,
            sample indices, etc.).
        n_maf_layers : int, optional
            The number of MAF layers.
        **kwargs
            Other keyword arguments to pass to the constructor of :class:`tfep.nn.flows.MAF`.

        See Also
        --------
        `MDAnalysis Universe object <https://docs.mdanalysis.org/2.6.1/documentation_pages/core/universe.html#MDAnalysis.core.universe.Universe>`_

        """
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

        )
        self.save_hyperparameters('n_maf_layers')
        self.kwargs = kwargs

    def configure_flow(self) -> torch.nn.Module:
        """Initialize the normalizing flow.

        Returns
        -------
        flow : torch.nn.Module
            The normalizing flow.

        """
        conditioning_indices = self.get_conditioning_indices(
            idx_type='dof', remove_fixed=True, remove_reference=True)

        # Build MAF layers.
        maf_layers = []
        for layer_idx in range(self.hparams.n_maf_layers):
            maf_layers.append(tfep.nn.flows.MAF(
                dimension_in=self.n_nonfixed_dofs,
                conditioning_indices=conditioning_indices,
                degrees_in='input' if (layer_idx%2 == 0) else 'reversed',
                **self.kwargs,
            ))
        flow = tfep.nn.flows.SequentialFlow(*maf_layers)

        # Determine origin and axes atom indices after the fixed DOFs have been removed.
        origin_atom_idx, axes_atoms_indices = self.get_reference_atoms_indices(
            remove_fixed=True, separate_origin_axes=True)

        # If the removed origin atom is before the axes atom, their indices shift.
        if (origin_atom_idx is not None) and (axes_atoms_indices is not None):
            for axes_atom_idx in range(2):
                if origin_atom_idx < axes_atoms_indices[axes_atom_idx]:
                    axes_atoms_indices[axes_atom_idx] = axes_atoms_indices[axes_atom_idx] - 1

        # Set the axes orientation of the relative reference frame.
        if axes_atoms_indices is not None:
            flow = tfep.nn.flows.OrientedFlow(
                flow,
                axis_point_idx=axes_atoms_indices[0],
                plane_point_idx=axes_atoms_indices[1],
                axis='z',
                plane='xz',
            )

        # Set the origin of the relative reference frame.
        if origin_atom_idx is not None:
            flow = tfep.nn.flows.CenteredCentroidFlow(
                flow,
                space_dimension=3,
                subset_point_indices=[origin_atom_idx],
            )

        return flow

    def get_mapped_indices(
            self,
            idx_type: Literal['atom', 'dof'],
            remove_fixed: bool,
            remove_reference: bool = False,
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
        remove_reference : bool, optional
            If ``True``, the returned tensor represent the indices after the
            reference frame atoms (i.e., origin and axes atoms) have been removed.
            Note that if ``idx_type == 'dof'``, only the constrained DOFs of the
            reference frame atoms are removed.

        Returns
        -------
        indices : torch.Tensor
            The mapped atom/DOFs indices.

        """
        indices = super().get_mapped_indices(idx_type=idx_type, remove_fixed=remove_fixed)
        if remove_reference:
            indices = self._remove_reference_indices(indices, idx_type=idx_type, remove_fixed=remove_fixed)
        return indices

    def get_conditioning_indices(
            self,
            idx_type: Literal['atom', 'dof'],
            remove_fixed: bool,
            remove_reference: bool = False,
    ) -> torch.Tensor:
        """Return the indices of the conditioning atom or degrees of freedom (DOF).

        Each atom generally has 3 degrees of freedom, except for the atoms used
        to set the relative frame of reference. If the ``axes_atoms`` (or only
        one of them) have been indicated as conditioning, the returned conditioning
        DOFs indices also include the DOFs of the ``axes_atoms`` that are not
        constrained, i.e., the ``x`` coordinate of ``axes_atoms[0]``, and the
        ``x,y`` coordinates of ``axes_atoms[1]``. The ``origin_atom`` is always
        a conditioning atom by definition, and it is thus included in the returned
        indices unless ``remove_constrained is True``.

        Parameters
        ----------
        idx_type : Literal['atom', 'dof']
            Whether to return the indices of the atom or the degrees of freedom.
        remove_fixed : bool
            If ``True``, the returned tensor represent the indices after the
            fixed atoms have been removed.
        remove_reference : bool, optional
            If ``True``, the returned tensor represent the indices after the
            reference frame atoms (i.e., origin and axes atoms) have been removed.
            Note that if ``idx_type == 'dof'``, only the constrained DOFs of the
            reference frame atoms are removed.

        Returns
        -------
        indices : torch.Tensor
            The conditioning atom/DOFs indices.

        """
        indices = super().get_conditioning_indices(idx_type=idx_type, remove_fixed=remove_fixed)
        if remove_reference and (indices is not None):
            indices = self._remove_reference_indices(indices, idx_type=idx_type, remove_fixed=remove_fixed)
        return indices

    def _remove_reference_indices(
            self,
            indices: torch.Tensor,
            idx_type: Literal['atom', 'dof'],
            remove_fixed: bool,
    ):
        """Override _get_nonfixed_indices() to optionally remove the reference atoms.

        indices must represent indices of the same idx_type and with/without fixed atoms
        as parameters.

        """
        # Return if there's nothing to remove.
        removed_indices = self.get_reference_atoms_indices(remove_fixed=remove_fixed)
        if removed_indices is None:
            return indices

        # Convert to DOF indices.
        if idx_type == "dof":
            # Find all the constrained DOFs associated with the origin and axes atoms.
            removed_dof_indices = []

            has_origin = len(removed_indices) in {1, 3}
            if has_origin:
                # All DOFs of the origin atoms are constrained.
                removed_dof_indices.append(atom_to_flattened_indices(removed_indices[:1]))

            has_axes = len(removed_indices) > 1
            if has_axes:
                # axes_atom[0] is constrained on the z-axis so x,y coordinates are fixed.
                removed_dof_indices.append(atom_to_flattened_indices(removed_indices[-2:-1])[:2])
                # axes_atom[1] is constrained on the x-z plane so y coordinate is fixed.
                removed_dof_indices.append(atom_to_flattened_indices(removed_indices[-1:])[1:2])

            # Update from atom to DOF the indices to remove.
            removed_indices = removed_dof_indices

        # Remove the reference atom indices.
        if len(removed_indices) > 0:
            # removed_indices must be sorted for searchsorted.
            removed_indices = torch.cat(removed_indices).sort()[0]

            # We need to first to delete the constrained reference DOFs that belong to indices.
            mask = True
            for idx in removed_indices:
                mask &= indices != idx
            indices = indices[mask]

            # And now shift the indices to account for the removal of the
            # constrained indices (even if they don't belong to indices).
            indices = indices - torch.searchsorted(removed_indices, indices)

        return indices
