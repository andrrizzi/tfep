#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""Base ``LightningModule`` class to implement TFEP maps."""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

from collections.abc import Sequence
from typing import Optional, Tuple, Union

import MDAnalysis
import numpy as np
import pint
import torch

from tfep.app.base import TFEPMapBase
import tfep.nn.flows
from tfep.utils.misc import atom_to_flattened_indices


# =============================================================================
# TFEP MAP BASE CLASS
# =============================================================================

class MixedMAFMap(TFEPMapBase):
    """A TFEP map using a masked autoregressive flow in mixed internal and Cartesian coordinates.

    The class divides the atoms of the entire system in mapped, conditioning,
    and fixed. Mapped atoms are defined as those that the flow maps. Conditioning
    atoms are not mapped but are given as input to the flow to condition the
    mapping. Fixed atoms are instead ignored.

    The coordinates of each mapped molecule are transformed before going through
    the MAF. Three atoms (the molecule centroid and two atoms bonded to it)
    are mapped as Cartesian coordinates while the rest of the degrees of freedom
    is mapped in internal coordinates using a Z-matrix. The conditioning atoms
    are always represented as Cartesian coordinates.

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
        The base class for TFEP maps with more detailed explanations of how the
        relative reference frame and the division in mapped/conditioning/fixed
        atoms work.

    Examples
    --------

    >>> from tfep.potentials.psi4 import PotentialPsi4
    >>> units = pint.UnitRegistry()
    >>>
    >>> tfep_map = MixedMAFMap(
    ...     potential_energy_func=PotentialPsi4(name='mp2'),
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
            (e.g. :class:`tfep.potentials.psi4.PotentialASE`).
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
            ``axes_atoms[0]``-th atom will lay on the ``x`` axis , and the
            ``axes_atoms[1]``-th atom will lay on the plane spanned by the ``x``
            and ``y`` axes. The ``z`` axis will be set as the cross product of
            ``x`` and ``y``.

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
        # Determine Z-matrix and Cartesian mapped atoms.
        mapped_cartesian_atom_indices, z_matrix = self._build_z_matrix()
        cartesian_atom_indices = torch.from_numpy(mapped_cartesian_atom_indices)

        # Shortcuts.
        n_ic_atoms = len(z_matrix)

        # Handle conditioning DOFs.
        maf_conditioning_dof_indices = []
        if self.n_conditioning_atoms > 0:
            conditioning_atom_indices = self.get_conditioning_indices(
                idx_type='atom', remove_fixed=True, remove_reference=True)

            # All conditioning atoms enter the flow as Cartesian coordinates as well.
            cartesian_atom_indices = torch.concatenate([cartesian_atom_indices, conditioning_atom_indices])

            # Determine conditioning DOF indices for the MAF. The conditioning DOFs
            # are at the end of the output of _CartesianToMixedFlow, just before
            # the axes atoms DOFs.
            start_idx = 3 * (n_ic_atoms + len(mapped_cartesian_atom_indices))
            end_idx = start_idx + 3 * len(conditioning_atom_indices)
            maf_conditioning_dof_indices.extend(list(range(start_idx, end_idx)))

        # Now eventually add the axes atom DOFs.
        if self._axes_atom_indices is not None:
            # Axes atoms can be either mapped or conditioning.
            is_atom_0_mapped, is_atom_1_mapped = self._are_axes_atoms_mapped()
            if not is_atom_0_mapped:
                maf_conditioning_dof_indices.append(end_idx)
            if not is_atom_1_mapped:
                maf_conditioning_dof_indices.extend([end_idx+1, end_idx+2])

        # Pass None to MAF.
        if len(maf_conditioning_dof_indices) == 0:
            maf_conditioning_dof_indices = None

        # Determine the periodic degrees of freedom (i.e., angles and torsions).
        # These are put by _CartesianToMixedFlow right after the bonds.
        maf_periodic_dof_indices = list(range(n_ic_atoms, 3*n_ic_atoms))

        # Create the transformer.
        transformer = self._get_transformer(
            n_ic_atoms, len(mapped_cartesian_atom_indices), maf_periodic_dof_indices)

        # Build MAF layers.
        maf_layers = []
        for layer_idx in range(self.hparams.n_maf_layers):
            maf_layers.append(tfep.nn.flows.MAF(
                dimension_in=self.n_nonfixed_dofs,
                conditioning_indices=maf_conditioning_dof_indices,
                periodic_indices=maf_periodic_dof_indices,
                # The periodic limits are 0 to 1 if normalize_angles=True in _CartesianToMixedFlow
                periodic_limits=[0, 1],
                degrees_in='input' if (layer_idx%2 == 0) else 'reversed',
                transformer=transformer,
                **self.kwargs,
            ))
        flow = tfep.nn.flows.SequentialFlow(*maf_layers)

        # Get the (unconstrained) DOF indices of the axes atoms that will be sent to.
        # _CartesianToMixedFlow. The constrained DOFs are removed by TFEPMapBase.
        axes_dof_indices = self._get_axes_dof_indices()

        # Wrap the MAF into a flow performing the change of coordinates.
        flow = _CartesianToMixedFlow(
            flow=flow,
            cartesian_atom_indices=cartesian_atom_indices,
            z_matrix=z_matrix,
            axes_dof_indices=axes_dof_indices,
        )

        return flow

    def _build_z_matrix(self):
        """Build a Z-matrix for the system and determine the fixed atoms.

        Note that the returned indices refer to those after the fixed and reference
        atoms have been removed.

        Returns
        -------
        mapped_cartesian_atom_indices : numpy.ndarray
            Shape (n_cartesian_atoms,). The indices of the atoms represented by
            Cartesian coordinates, excluding the reference frame atoms. The array
            is sorted in ascending order.
        z_matrix : numpy.ndarray
            Shape (n_ic_atoms, 4). The Z-matrix for the atoms represented by
            internal coordinates. E.g., ``z_matrix[i] == [7, 2, 4, 8]``
            means that the distance, angle, and dihedral for atom ``7`` must be
            computed between atoms ``7-2``, ``7-2-4``, and ``7-2-4-8`` respectively.

        """
        import networkx as nx

        # We transform into internal coordinates only the mapped atoms, but not
        # the reference frame atoms which are treated in Cartesian coordinates
        # even if mapped. First we need to create a graph of these mapped atoms.
        mapped_atom_indices_w_fixed = self.get_mapped_indices(
            idx_type='atom', remove_fixed=False, remove_reference=False)

        # Remove the axes atoms that are treated in Cartesian coordinates.
        if self._axes_atom_indices is not None:
            mask = True
            for idx in self._axes_atom_indices:
                mask &= mapped_atom_indices_w_fixed != idx
            mapped_atom_indices_w_fixed = mapped_atom_indices_w_fixed[mask]

        # Select only the bonds in which both atoms are in the atom group.
        mapped_atoms = MDAnalysis.AtomGroup(mapped_atom_indices_w_fixed, self.dataset.universe)
        internal_bonds = [b for b in mapped_atoms.bonds
                          if (b.atoms[0] in mapped_atoms and b.atoms[1] in mapped_atoms)]

        # Build a networkx graph representing the topology of all the mapped atoms.
        system_graph = nx.Graph()
        system_graph.add_nodes_from(mapped_atoms)
        system_graph.add_edges_from(internal_bonds)

        # Initialize returned values.
        mapped_cartesian_atom_indices = []
        system_z_matrix = []

        # Build the Z-matrix for each connected subgraph.
        for graph_nodes in nx.connected_components(system_graph):
            graph = system_graph.subgraph(graph_nodes).copy()

            # graph_distances[i, j] is the distance (in number of edges) between atoms i and j.
            # We don't need paths longer than 3 edges as we'll select torsion atoms prioritizing
            # closer atoms.
            graph_distances = dict(nx.all_pairs_shortest_path_length(graph, cutoff=3))

            # Select the first atom as the graph center.
            center_atom = nx.center(graph)[0]
            sub_z_matrix = [[center_atom.index, -1, -1, -1]]

            # atom_order[atom_idx] is the row index of the Z-matrix defining its coords.
            atoms_order = {center_atom.index: 0}

            # We traverse the graph breadth first.
            for _, added_atom in nx.bfs_edges(graph, source=center_atom):
                z_matrix_row = [added_atom.index]

                # Find bond atom.
                is_h = _is_hydrogen(added_atom)
                priorities = self._get_atom_zmatrix_priorities(added_atom, graph_distances, atoms_order, is_h)
                z_matrix_row.append(priorities[0][0])

                # For angle and torsion atoms, adds also the distance to the bond atom
                # in the priorities. This reduces the chance of collinear torsions.
                bond_atom = self.dataset.universe.atoms[z_matrix_row[-1]]
                priorities = self._get_atom_zmatrix_priorities(added_atom, graph_distances, atoms_order, is_h, bond_atom)
                z_matrix_row.extend([p[0] for p in priorities[:2]])

                # The first two added atoms are the reference atoms.
                if len(z_matrix_row) < 4:
                    assert len(sub_z_matrix) < 4
                    z_matrix_row = z_matrix_row + [-1] * (4-len(z_matrix_row))

                # Add entry to Z-matrix.
                sub_z_matrix.append(z_matrix_row)

                # Add this atom to those added to the Z-matrix.
                atoms_order[added_atom.index] = len(atoms_order)

            # Now separate the fixed atoms from the Z-matrix ones.
            mapped_cartesian_atom_indices.extend([row[0] for row in sub_z_matrix[:3]])
            system_z_matrix.extend(sub_z_matrix[3:])

        # The atom indices and the Z-matrix so far use the atom indices of the
        # systems before the fixed and reference atoms have been removed. Now
        # we need to map the indices to those after they are removed since these
        # are not passed to _CartesianToMixed._rel_ic.
        mapped_atom_indices_w_fixed = mapped_atom_indices_w_fixed.numpy()
        mapped_atom_indices_wo_fixed = self.get_mapped_indices(
            idx_type='atom', remove_fixed=True, remove_reference=True).numpy()
        indices_map = {mapped_atom_indices_w_fixed[i]: mapped_atom_indices_wo_fixed[i]
                       for i in range(len(mapped_atom_indices_w_fixed))}

        # Convert indices.
        mapped_cartesian_atom_indices = [indices_map[i] for i in mapped_cartesian_atom_indices]
        for row_idx, z_matrix_row in enumerate(system_z_matrix):
            system_z_matrix[row_idx] = [indices_map[i] for i in z_matrix_row]

        # Sort atom indices and convert everything to numpy array (for RelativeInternalCoordinateTransformation).
        mapped_cartesian_atom_indices = np.array(mapped_cartesian_atom_indices)
        mapped_cartesian_atom_indices.sort()
        return mapped_cartesian_atom_indices, np.array(system_z_matrix)

    def _get_atom_zmatrix_priorities(self, atom, graph_distances, atoms_order, is_h, bond_atom=None):
        """Build priority list for this atom.

        Atoms are prioritized with the following criterias (in this order).
        1) Closest to ``atom``.
        2) Closest to the ``bond_atom`` if passed as an argument (for angle and torsion atoms).
        3) Most recently added to the Z-matrix.
        4) Prioritize heavy atoms if ``atom`` is not a hydrogen.

        """
        # priorities[i][0] is the atom index.
        # priorities[i][1] is the distance (in number of edges) from atom.
        # priorities[i][2] is the distance (in number of edges) from bond_atom.
        # priorities[i][3] is the index at which this atom has been added to the z-matrix.
        # priorities[i][4] is 1 if it is a hydrogen and atom_idx isn't, else 0.
        # This way we can select the reference atoms simply by sorting a list.
        priorities = []
        for prev_atom, dist in graph_distances[atom].items():
            # atom_idx cannot depend on itself and on atoms that are not already in the Z-matrix.
            if (prev_atom.index not in atoms_order) or (atom.index == prev_atom.index):
                continue

            if bond_atom is None:
                # Set all bond distances to the same value to avoid prioritization based on this criteria.
                bond_atom_dist = 0
            elif prev_atom.index == bond_atom.index:
                # Do not add bond_atom twice.
                continue
            elif prev_atom not in graph_distances[bond_atom]:
                # prev_atom needs to be close to the bond atom as well.
                continue
            else:
                bond_atom_dist = graph_distances[bond_atom][prev_atom]

            priorities.append([
                prev_atom.index,
                dist,
                bond_atom_dist,
                atoms_order[prev_atom.index],
                float(not is_h and _is_hydrogen(prev_atom)),
            ])

        # The minus sign of the atom order is because we want to prioritize atoms that have just been added.
        priorities.sort(key=lambda k: (k[1], k[2], -k[3], k[4]))
        return priorities

    def _get_transformer(
            self,
            n_ic_atoms,
            n_cartesian_atoms,
            maf_periodic_dof_indices,
    ):
        """Return the transformer for the MAF.

        Parameters
        ----------
        n_ic_atoms : int
            Number of atoms mapped in the internal coordinate representation.
        n_cartesian_atoms : int
            The number of mapped atoms (excluding reference frame atoms) that
            are mapped using Cartesian coordinates.
        maf_periodic_dof_indices : Sequence[int]
            The indices of the mapped DOFs that must be treated as periodic by
            the neural spline.

        """
        # Shortcut.
        n_mapped_dofs = self.n_mapped_dofs

        # We need to determine the limits only for the mapped (not conditioning)
        # DOFs. Initialize using the limits for the (normalized) angles.
        x0 = torch.zeros(n_mapped_dofs)
        xf = torch.ones(n_mapped_dofs)

        # Now fill the limits for the bonds.
        bond_dof_indices = list(range(n_ic_atoms))
        x0[bond_dof_indices] = 0.5  # in Angstrom
        xf[bond_dof_indices] = 3.0  # in Angstrom

        # Set the limits for the atoms mapped in Cartesian coordinates.
        # The mapped Cartesian DOFs are placed by _CartesianToMixed right after
        # the internal coordinates (bonds, angles, and torsions).
        n_ic_dofs = 3 * n_ic_atoms
        n_nonreference_dofs = n_ic_dofs + 3 * n_cartesian_atoms
        mapped_cartesian_dof_indices = list(range(n_ic_dofs, n_nonreference_dofs))

        # Also the reference frame atoms (if mapped) are treated as Cartesian.
        n_reference_dofs = n_mapped_dofs - n_nonreference_dofs
        if n_reference_dofs > 0:
            # The reference frame atoms are appended at the end, after the conditioning atoms.
            n_nonfixed_dofs = self.n_nonfixed_dofs  # Shortcut.
            assert n_reference_dofs <= 3

            # Add the axis atom DOF.
            if n_reference_dofs >= 1:
                mapped_cartesian_dof_indices.append(n_nonfixed_dofs-3)

            # Add the 2 DOFs of the plane atom.
            if n_reference_dofs > 1:
                # The plane atom (2 DOFs) is also mapped.
                mapped_cartesian_dof_indices.extend([n_nonfixed_dofs-2, n_nonfixed_dofs-1])

        # Set the limits for the mapped Cartesian coordinates.
        max_displacement = 1.5  # In Angstrom.
        min_vals, max_vals = self._get_min_max_dofs(mapped_cartesian_dof_indices)
        x0[mapped_cartesian_dof_indices] = min_vals - max_displacement  # In Angstrom.
        xf[mapped_cartesian_dof_indices] = max_vals + max_displacement  # In Angstrom.

        return tfep.nn.transformers.NeuralSplineTransformer(
            x0=x0,
            xf=xf,
            n_bins=5,
            circular=maf_periodic_dof_indices,
        )

    def _get_min_max_dofs(self, dof_indices: torch.Tensor) -> Tuple[torch.Tensor]:
        """Compute the minimum and maximum value of each DOF in the trajectory for the given atom indices.

        These are the coordinates in the relative frame of reference.

        This is useful to calculate appropriate values for the left/rightmost
        nodes of the neural spline transformer.

        Parameters
        ----------
        dof_indices : torch.Tensor
            The indices of the DOFs for which to compute the min and max values.

        Returns
        -------
        min_dofs : torch.Tensor
            ``min_dofs[i]`` is the minimum value of the ``dof_indices[i]``-th
            degree of freedom.
        max_dofs : torch.Tensor
            ``max_dofs[i]`` is the maximum value of the ``dof_indices[i]``-th
            degree of freedom.

        """
        # We need to calculate the minimum and maximum dof AFTER it has gone
        # through the partial flow and the relative frame of reference has been
        # fixed as this is the input that will be passed to the transformers.
        identity_flow = lambda x_: (x_, torch.zeros_like(x_[:, 0]))
        flow = self._create_change_of_frame_flow(identity_flow, restore=False)

        # Initialize returned values.
        n_dofs = len(dof_indices)
        min_dofs = torch.full((n_dofs,), float('inf'))
        max_dofs = torch.full((n_dofs,), -float('inf'))

        # Read the trajectory in batches.
        dataset = self._create_dataset()
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=1024, shuffle=False, drop_last=False
        )
        for batch_data in data_loader:
            # Go through partial flow.
            dofs, _ = flow(batch_data['positions'])

            # Take the min/max across the batch of the selected DOFs.
            batch_min = torch.min(dofs[:, dof_indices], dim=0).values
            batch_max = torch.max(dofs[:, dof_indices], dim=0).values

            # Update current min/max.
            min_dofs = torch.minimum(min_dofs, batch_min)
            max_dofs = torch.maximum(max_dofs, batch_max)

        return min_dofs, max_dofs

    def _get_axes_dof_indices(self):
        """Return the indices corresponding to the unconstrained DOFs of the axes atoms received by _CartesianToMixedFlow."""
        if self._axes_atom_indices is None:
            return None

        # Get the atom indices after the fixed and origin atom has been removed.
        axes_atom_indices = self._get_passed_reference_atom_indices(remove_origin_from_axes=True)[-2:]

        # axes_atom[0] is constrained on the x-axis so y,z coordinates are fixed.
        # axes_atom[1] is constrained on the x-y plane so z coordinate is fixed.
        axes_dof_indices = atom_to_flattened_indices(axes_atom_indices)
        axes_dof_indices = torch.concatenate([axes_dof_indices[:1], axes_atom_indices[:2]])

        # Shift the indices to account for the removed constrained DOFs.
        if axes_atom_indices[0] < axes_atom_indices[1]:
            axes_dof_indices[1:] -= 2
        else:
            axes_dof_indices[:1] -= 1

        return axes_dof_indices


# =============================================================================
# HELPER CLASSES AND FUNCTIONS
# =============================================================================

def _is_hydrogen(atom):
    """Return True if the atom is a hydrogen.

    atom is an MDAnalysis.Atom.

    Raises an exception if the atom has no information on the element.

    """
    try:
        element = atom.element
    except MDAnalysis.exceptions.NoDataError as e:
        err_msg = ("The topology files have no information on the atom elements. "
                   "This is required to infer a robust Z-matrix. You can either "
                   "provide a topology file that includes this info (e.g., a PDB) "
                   "or add this information programmatically by overwriting "
                   "MixedMAFMap._create_universe() and set, e.g., "
                   "universe.add_TopologyAttr('element', ['O', 'H', 'C', ...]).")
        raise ValueError(err_msg) from e
    else:
        element = element.upper()
    return element == 'H'


class _CartesianToMixedFlow(torch.nn.Module):
    """Utility flow to convert from Cartesian to mixed coordinates."""

    def __init__(
            self,
            flow: torch.nn.Module,
            cartesian_atom_indices: np.ndarray,
            z_matrix: np.ndarray,
            axes_dof_indices: Optional[torch.Tensor],
    ):
        """Constructor.

        Parameters
        ----------
        flow : torch.nn.Module
            The normalizing flow to wrap that takes as input mixed coordinates.
        cartesian_atom_indices : np.ndarray
            Shape ``(n_cartesian_atoms,)``. Indices of the atoms represented as
            Cartesian (both mapped and conditioning), except for the reference
            frame atoms.
        z_matrix : np.ndarray
            Shape ``(n_ic_atoms, 4)``. The Z-matrix for the atoms represented as
            internal coordinates.
        axes_dof_indices : torch.Tensor or None
            The indices of the unconstrained DOFs of the axes atoms determining
            the relative reference frame. These are also passed as Cartesian
            coordinates.

        """
        from bgflow.nn.flow.crd_transform.ic import RelativeInternalCoordinateTransformation

        super().__init__()

        #: Wrapped flow taking as input the mixed coordinates.
        self.flow = flow

        # Indices of the non-constrained reference frame atom DOFs (passed as Cartesian coordinates).
        self._axes_dof_indices = axes_dof_indices

        # Cache all the other indices.
        if self._n_axes_dofs > 0:
            axes_dof_indices_set = set(axes_dof_indices.tolist())
            n_dofs = 3 * (len(z_matrix) + len(cartesian_atom_indices)) + self._n_axes_dofs
            self._non_axes_dof_indices = torch.tensor([i for i in range(n_dofs)
                                                       if i not in axes_dof_indices_set])

        # The module performing the coordinate conversion.
        self._rel_ic = RelativeInternalCoordinateTransformation(
            z_matrix=z_matrix,
            fixed_atoms=cartesian_atom_indices,
            normalize_angles=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward transformation."""
        return self._pass(x, inverse=False)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Inverse transformation."""
        return self._pass(y, inverse=True)

    @property
    def _n_axes_dofs(self) -> int:
        """The number of reference DOFs to remove before the conversion."""
        if self._axes_dof_indices is None:
            return 0
        return 3

    def _pass(self, x, inverse):
        # Convert from Cartesian to mixed coordinates.
        y, cumulative_log_det_J = self._cartesian_to_mixed(x)

        # Run flow.
        if inverse:
            y, log_det_J = self.flow.inverse(y)
        else:
            y, log_det_J = self.flow(y)
        cumulative_log_det_J = cumulative_log_det_J + log_det_J

        # Convert from mixed to Cartesian coordinates.
        y, log_det_J = self._mixed_to_cartesian(y)
        cumulative_log_det_J = cumulative_log_det_J + log_det_J

        return y, cumulative_log_det_J

    def _cartesian_to_mixed(self, x: torch.Tensor) -> torch.Tensor:
        """Convert from Cartesian to mixed coordinates."""
        # Separate relative reference frame atoms.
        if self._n_axes_dofs > 0:
            x_ref = x[:, self._axes_dof_indices]
            x = x[:, self._non_axes_dof_indices]

        # Convert.
        bonds, angles, torsions, x_fixed, log_det_J = self._rel_ic(x)

        # Concatenate all DOFs.
        to_concatenate = [bonds, angles, torsions, x_fixed]
        if self._n_axes_dofs > 0:
            to_concatenate.append(x_ref)
        y = torch.cat(to_concatenate, dim=-1)

        return y, log_det_J

    def _mixed_to_cartesian(self, y: torch.Tensor) -> torch.Tensor:
        """Convert from mixed to Cartesian."""
        # Separate the mapped IC and Cartesian DOFs.
        n_z_matrix_atoms = len(self._rel_ic.z_matrix)
        bonds = y[:, :n_z_matrix_atoms]
        angles = y[:, n_z_matrix_atoms:2*n_z_matrix_atoms]
        torsions = y[:, 2*n_z_matrix_atoms:3*n_z_matrix_atoms]
        y_fixed = y[:, 3*n_z_matrix_atoms:]

        # Separate reference frame atom DOFs.
        if self._n_axes_dofs > 0:
            y_ref = y_fixed[:, -self._n_axes_dofs:]
            y_fixed = y_fixed[:, :-self._n_axes_dofs]

        # Convert back from mixed to Cartesian.
        x, log_det_J = self._rel_ic(bonds, angles, torsions, y_fixed, inverse=True)

        # Re-add reference frame atom DOFs.
        if self._n_axes_dofs > 0:
            tmp = x
            x = torch.empty_like(y)
            x[:, self._axes_dof_indices] = y_ref
            x[:, self._non_axes_dof_indices] = tmp

        return x, log_det_J
