#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""Base ``LightningModule`` class to implement TFEP maps."""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

from collections.abc import Sequence
import logging
from typing import Dict, List, Optional, Set, Tuple, Union

import MDAnalysis
import networkx as nx
import numpy as np
import pint
import torch

import tfep.nn.conditioners.made
import tfep.nn.embeddings
import tfep.nn.flows
from tfep.app.base import TFEPMapBase
from tfep.utils.geometry import (
    batchwise_dot,
    batchwise_rotate,
    get_axis_from_name,
    reference_frame_rotation_matrix,
    cartesian_to_polar,
    polar_to_cartesian,
)
from tfep.utils.misc import (
    atom_to_flattened_indices,
    atom_to_flattened,
    flattened_to_atom,
    ensure_tensor_sequence,
    remove_and_shift_sorted_indices,
)


# =============================================================================
# GLOBAL VARIABLES
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# TFEP MAP BASE CLASS
# =============================================================================

class MixedMAFMap(TFEPMapBase):
    """A TFEP map using a masked autoregressive flow in mixed internal and Cartesian coordinates.

    The class divides the atoms of the entire system in mapped, conditioning,
    and fixed. Mapped atoms are defined as those that the flow maps. Conditioning
    atoms are not mapped but are given as input to the flow to condition the
    mapping. Fixed atoms are instead ignored.

    Before going through the MAF, the coordinates of are transformed into a mixed
    Cartesian/internal coordinate system. To this end, the system is first
    divided into connected fragments based on the bond topology, and a separate
    Z-matrix is built for each fragment.

    The Z-matrix is automatically determined based on a heuristic. The first atom
    is chosen as the center of the graph representing the molecule. The graph is
    then traversed breath first from the center and, for each atom, the bond,
    angle, and dihedral atoms are selected from those in the current Z-matrix
    according to these priorities: 1) closest to the inserted atom; 2) only for
    angle and dihedral atoms, closest to the bond atom; 3) most recently added
    to the Z-matrix 4) only for heavy atoms, hydrogens are de-prioritized. In
    particular, 2) and 3) limit the occurrence of undefined angles and instability
    during training as a result of a triplet of collinear atoms.

    The first three atoms of each molecule's Z-matrix and all conditioning atoms
    are represented as Cartesian, while the remaining mapped atoms are converted
    to internal coordinates.

    The flow also rototranslates the Cartesian coordinates into a relative
    frame of reference which based on the position of an ``origin_atom`` and
    two ``axes_atoms`` that determine the origin and the orientation of the
    axes, respectively. When given, these atoms are prioritized for the choice
    of the first three atoms of a molecule's Z-matrix. If not passed, these 3
    atoms are automatically chosen as the first three atoms in the Z-matrix of
    the largest fragment. Optionally, the roto-translational degrees of freedom
    can be removed from the mapping with ``remove_translation`` and
    ``remove_rotation``.

    When ``remove_rotation is True`` the ``axes_atoms`` are represented in
    internal coordinates (2 distances w.r.t. the origin atom and 1 angle).
    When ``False``, the axis/plane atoms are represented in Cartesian/cylindrical
    coordinates. This is just because it simplifies the support for removing
    the global rotational degrees of freedom with ``remove_rotation``.

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

    >>> from tfep.potentials.psi4 import Psi4Potential
    >>> units = pint.UnitRegistry()
    >>>
    >>> tfep_map = MixedMAFMap(
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
            remove_translation: bool = False,
            remove_rotation: bool = False,
            tfep_logger_dir_path: str = 'tfep_logs',
            n_maf_layers: int = 6,
            distance_lower_limit_displacement: Optional[pint.Quantity] = None,
            dataloader_kwargs: Optional[Dict] = None,
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
            is not given).
        conditioning_atoms : Sequence[int] or str, optional
            The indices (0-based) of the atoms conditioning the mapping or a
            selection string in MDAnalysis syntax. If not passed, no atom will
            condition the map. These atoms cannot overlap with ``mapped_atoms``.
        origin_atom : int or str or None, optional
            The index (0-based) or a selection string in MDAnalysis syntax of an
            atom on which to center the origin of the relative frame of reference.
            If a conditioning atom, the coordinates are not passed to the flow
            as they would be always zero. By default, this is chosen as the 1st
            atom in the Z-matrix of the largest fragment.
        axes_atoms : Sequence[int] or str or None, optional
            A pair of indices (0-based) or a selection string in MDAnalysis syntax
            for the two atoms determining the relative frame of reference. The
            ``axes_atoms[0]``-th atom will lay on the ``z`` axis , and the
            ``axes_atoms[1]``-th atom will lay on the plane spanned by the ``x``
            and ``z`` axes. The ``y`` axis will be set as the cross product of
            ``x`` and ``y``.

            If conditioning atoms, the coordinates that after the rotation are
            0 are not passed to the flow. The other degrees of freedom are
            converted into two distances (from the origin atom) and a valence
            angle. By default, these are chosen as the 2nd and 3rd atoms in the
            Z-matrix of the largest fragment.
        remove_translation : bool, optional
            If ``True``, the 3 degrees of freedom of the ``origin_atom`` are
            not mapped even if ``origin_atom`` is mapped. When ``origin_atom``
            is conditioning, this option has no effect.
        remove_rotation : bool, optional
            If ``True``, the 3 rotational degrees of freedom of ``axes_atoms``
            are not mapped even if ``axes_atoms`` are mapped atoms. In this
            case, only their 2 distances from the origin atom and the valence
            angle between them is passed to the flow. When ``axes_atoms`` are
            conditioning, this option has no effect.
        tfep_logger_dir_path : str, optional
            The path where to save TFEP-related information (potential energies,
            sample indices, etc.).
        n_maf_layers : int, optional
            The number of MAF layers.
        distance_lower_limit_displacement : pint.Quantity or None
            This controls the (fixed) lower limit for the neural spline used to
            transform bond lengths. This lower limit is set to
            ``max(0, min_observed - distance_lower_limit_displacement)`` where
            ``min_observed`` is the minimum bond length observed for the specific
            bond on a random sample in the dataset. The default value is 0.3
            Angstrom.

            Note that the same maximum displacement is applied to control the
            two distances between the axes atoms and the origin.
        dataloader_kwargs : Dict, optional
            Extra keyword arguments to pass to ``torch.utils.data.DataLoader``.
        **kwargs
            Other keyword arguments to pass to the constructor of :class:`tfep.nn.flows.MAF`.

        See Also
        --------
        `MDAnalysis Universe object <https://docs.mdanalysis.org/2.6.1/documentation_pages/core/universe.html#MDAnalysis.core.universe.Universe>`_

        """
        # Check input.
        if axes_atoms is not None and origin_atom is None:
            raise ValueError('axes_atoms cannot be passed if origin_atom is None.')

        # Convert bond
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

        # Default value and unit conversion.
        positions_unit = potential_energy_func.positions_unit
        if distance_lower_limit_displacement is None:
            distance_lower_limit_displacement = 0.3 * positions_unit._REGISTRY.angstrom
        distance_lower_limit_displacement = distance_lower_limit_displacement.to(positions_unit).magnitude

        # Save hyperparameters.
        self.hparams['distance_lower_limit_displacement'] = distance_lower_limit_displacement
        self.save_hyperparameters('n_maf_layers', 'remove_translation', 'remove_rotation')

        # MAF kwargs.
        self._kwargs = kwargs

    def configure_flow(self) -> torch.nn.Module:
        """Initialize the normalizing flow.

        Returns
        -------
        flow : torch.nn.Module
            The normalizing flow.

        """
        # Determine Z-matrix and Cartesian atoms and (optionally)
        # automatically determine origin and axes atoms.
        cartesian_atom_indices, z_matrix = self._build_z_matrix()
        if len(z_matrix) == 0:
            raise ValueError('There are no internal coordinates to map. '
                             'Consider using a Cartesian flow.')

        # If the origin and orientation atoms are conditioning, we remove the
        # rototranslational DOFs in the coordinate conversion since they are constant.
        reference_atom_indices = self.get_reference_atoms_indices(remove_fixed=True)
        conditioning_atom_indices = self.get_conditioning_indices(idx_type='atom', remove_fixed=True)
        if conditioning_atom_indices is None:
            is_ref_conditioning = [False, False, False]
        else:
            is_ref_conditioning = torch.isin(reference_atom_indices, conditioning_atom_indices).tolist()

        # Initialize _CartesianToMixedFlow that will map the MAF. We will set
        # the wrapped flow after we have initialized the MAF.
        cartesian_to_mixed_flow = _CartesianToMixedFlow(
            flow=None,
            cartesian_atom_indices=cartesian_atom_indices,
            z_matrix=z_matrix,
            reference_atom_indices=reference_atom_indices,
            remove_ref_rototranslation=[
                self.hparams.remove_translation or is_ref_conditioning[0],
                self.hparams.remove_rotation or is_ref_conditioning[1],
                self.hparams.remove_rotation or is_ref_conditioning[2],
            ],
        )

        # Now we take a pass at the trajectory to check that the Z-matrix is robust
        # (i.e., no collinear atoms determining angles) and compute the min/max values
        # of the DOFs (after going through _CartesianToMixedFlow) to configure the
        # neural splines correctly.
        min_dof_vals, max_dof_vals = self._analyze_dataset(z_matrix, cartesian_to_mixed_flow)

        # Determine the various types of degrees of freedom after the
        # conversion from Cartesian to mixed coordinates.
        maf_dof_indices = cartesian_to_mixed_flow.get_dof_indices_by_type(conditioning_atom_indices)

        # Create the transformer.
        transformer = self._get_transformer(
            cartesian_to_mixed_flow=cartesian_to_mixed_flow,
            min_dof_vals=min_dof_vals,
            max_dof_vals=max_dof_vals,
            dof_indices=maf_dof_indices,
        )

        # Create the degrees_in argument for ascending (index 0) and descending
        # (index 1) order.
        degrees_in = self._get_maf_degrees_in(
            n_dofs_in=cartesian_to_mixed_flow.n_dofs_out,
            maf_dof_indices=maf_dof_indices
        )

        # Build MAF layers.
        maf_layers = []
        for layer_idx in range(self.hparams.n_maf_layers):
            maf_layers.append(tfep.nn.flows.MAF(
                degrees_in=degrees_in[layer_idx%2],
                transformer=transformer,
                embedding=tfep.nn.embeddings.PeriodicEmbedding(
                    n_features_in=cartesian_to_mixed_flow.n_dofs_out,
                    # The periodic limits are 0 to 1 if normalize_angles=True in _CartesianToMixedFlow
                    limits=[0, 1],
                    periodic_indices=maf_dof_indices['torsions'],
                ),
                **self._kwargs,
            ))
        flow = tfep.nn.flows.SequentialFlow(*maf_layers)

        # Wrap the MAF into the _CartesianToMixedFlow.
        cartesian_to_mixed_flow.flow = flow
        return cartesian_to_mixed_flow

    def _build_z_matrix(self):
        """Determine the Z-matrix, the Cartesian atoms, and (if not given) the reference frame atoms.

        See the class docstring for an overview of how the Z-matrix is determined.

        The Z-matrix is constructed so that the origin and axes atoms are always
        included among the Cartesian atoms.

        If not given by the user, this method also sets self._origin_atom_idx
        and self._axes_atoms_indices.

        Returns
        -------
        cartesian_atom_indices : list[int]
            Shape (n_cartesian_atoms,). The indices of the atoms represented by
            Cartesian coordinates (i.e., 3 reference atoms for each molecule,
            and all conditioning atoms). The array is sorted in ascending order.
        z_matrix : list[list[int]]
            Shape (n_ic_atoms, 4). The Z-matrix for the atoms represented by
            internal coordinates. E.g., ``z_matrix[i] == [7, 2, 4, 8]``
            means that the distance, angle, and dihedral for atom ``7`` must be
            computed between atoms ``7-2``, ``7-2-4``, and ``7-2-4-8`` respectively.

        """
        # First we need to create a graph representation of all the molecules
        # constituted by mapped and conditioning atoms. To build the graph
        # correctly, the indices must be those before the fixed atoms are removed.
        # Get the indices of the nonfixed atoms by merging mapped and condtioning.
        mapped_atom_indices_w_fixed = self.get_mapped_indices(idx_type='atom', remove_fixed=False)
        conditioning_atom_indices_w_fixed = self.get_conditioning_indices(idx_type='atom', remove_fixed=False)
        if conditioning_atom_indices_w_fixed is None:
            nonfixed_atom_indices_w_fixed = mapped_atom_indices_w_fixed
        else:
            nonfixed_atom_indices_w_fixed = torch.cat([
                mapped_atom_indices_w_fixed, conditioning_atom_indices_w_fixed]).sort()[0]

        # Build the graph.
        system_graph = self._create_networkx_graph(nonfixed_atom_indices_w_fixed.numpy())

        # Check if the user has provided reference frame atoms.
        ref_atom_indices = self.get_reference_atoms_indices(remove_fixed=False)
        if ref_atom_indices is None:
            ref_atom_indices = []
        else:
            ref_atom_indices = ref_atom_indices.tolist()

        nonfixed_atom_indices_w_fixed = nonfixed_atom_indices_w_fixed.tolist()
        if not set(ref_atom_indices).issubset(set(nonfixed_atom_indices_w_fixed)):
            raise ValueError('The origin and axes atoms must be mapped or conditioning.')

        # Only the mapped atoms are represented using internal coordinates
        # so we build a set to check for membership.
        mapped_atom_indices_w_fixed_set = set(mapped_atom_indices_w_fixed.tolist())

        # Build the Z-matrix for each connected subgraph.
        frags_z_matrices = []
        for graph_nodes in nx.connected_components(system_graph):
            graph = system_graph.subgraph(graph_nodes).copy()
            frags_z_matrices.append(self._build_connected_graph_z_matrix(graph, ref_atom_indices))

        # If not given, automatically determine the frame of reference. We need
        # to do it before we shift the indices to account for the removed fixed
        # atoms and before we remove the first three rows of the Z-matrix since
        # in the 4th row the order of origin/axes atoms is scrambled.
        z_matrix = frags_z_matrices[np.argmax([len(z) for z in frags_z_matrices])]
        if self._origin_atom_idx is None:
            self._origin_atom_idx = torch.tensor(z_matrix[0][0])
        if self._axes_atoms_indices is None:
            self._axes_atoms_indices = torch.tensor([z_matrix[1][0], z_matrix[2][0]])

        # Divide atoms treated as Cartesian and internal coords.
        cartesian_atom_indices = []
        ic_z_matrix = []
        for z_matrix in frags_z_matrices:
            # First three atoms of each Z-matrix are Cartesian.
            cartesian_atom_indices.extend([row[0] for row in z_matrix[:3]])

            # Only the mapped atoms are converted to internal coordinates.
            # Conditioning atoms are kept as Cartesian.
            is_mapped = False
            for z_matrix_row in z_matrix[3:]:
                if z_matrix_row[0] in mapped_atom_indices_w_fixed_set:
                    ic_z_matrix.append(z_matrix_row)
                    is_mapped = True
                else:
                    cartesian_atom_indices.append(z_matrix_row[0])

            # Test independence. No need to check purely conditioning fragments.
            if is_mapped:
                check_independent(z_matrix)

        # The atom indices and the Z-matrix so far use the atom indices of the
        # systems before the fixed and reference atoms have been removed. Now
        # we need to map the indices to those after they are removed since these
        # are not passed to _CartesianToMixed.rel_ic.
        indices_map = {nonfixed_atom_indices_w_fixed[idx]: idx for idx in range(self.n_nonfixed_atoms)}

        # Log Z-matrix.
        logger.info('Determined Z-Matrix:\n' + str(np.array(ic_z_matrix)))

        # Convert indices.
        cartesian_atom_indices = [indices_map[i] for i in cartesian_atom_indices]
        for row_idx, z_matrix_row in enumerate(ic_z_matrix):
            ic_z_matrix[row_idx] = [indices_map[i] for i in z_matrix_row]

        # Sort atom indices and convert everything to numpy array (for RelativeInternalCoordinateTransformation).
        cartesian_atom_indices = torch.tensor(cartesian_atom_indices).sort().values
        return cartesian_atom_indices, torch.tensor(ic_z_matrix)

    def _create_networkx_graph(self, atom_indices):
        """Return a networkx graph representing the given atoms."""
        # Select only the bonds in which both atoms are in the atom group.
        atoms = MDAnalysis.AtomGroup(atom_indices, self.dataset.universe)
        bonds = [bond for bond in atoms.bonds
                 if (bond.atoms[0] in atoms and bond.atoms[1] in atoms)]

        # Build a networkx graph representing the topology of all the nonfixed atoms.
        graph = nx.Graph()
        graph.add_nodes_from(atoms)
        graph.add_edges_from(bonds)

        return graph

    def _build_connected_graph_z_matrix(self, graph, ref_atom_indices):
        """Build the Z-matrix for a connected graph.

        Parameters
        ----------
        graph : networkx.Graph
            The graph with nodes and edges representing atoms and bonds.
        ref_atom_indices : list[int]
            The user-given reference atoms. These are prioritized when
            selecting the initial 3 atoms of the Z-matrix.

        Returns
        -------
        z_matrix : list[list[int]]
            The returned Z-matrix.

        """
        # For the first three atoms, we give priority to the user-defined
        # origin/axes atoms as long as they are within the molecule.
        ref_atoms_in_graph = [self.dataset.universe.atoms[i] for i in ref_atom_indices
                              if graph.has_node(self.dataset.universe.atoms[i])]

        # If origin/axes atom are not in the molecule, just set the first atom as the graph center.
        if len(ref_atoms_in_graph) == 0:
            ref_atoms_in_graph = [nx.center(graph)[0]]

        # Shortcuts.
        n_ref_atoms_in_graph = len(ref_atoms_in_graph)
        ref_atom_indices_in_graph = [atom.index for atom in ref_atoms_in_graph]

        # Add (up to) the first three atoms of the Z-matrix for the molecule.
        z_matrix = [[-1] * 4 for _ in range(n_ref_atoms_in_graph)]
        for row_idx in range(n_ref_atoms_in_graph):
            z_matrix[row_idx][:row_idx+1] = reversed(ref_atom_indices_in_graph[:row_idx+1])

        # atom_order[atom_idx] is the row index of the Z-matrix defining its coords.
        atoms_order = {}
        for row_idx, atom_idx in enumerate(ref_atom_indices_in_graph):
            atoms_order[atom_idx] = row_idx

        # graph_distances[i, j] is the distance (in number of edges) between atoms i and j.
        # We don't need paths longer than 3 edges as we'll select torsion atoms prioritizing
        # closer atoms.
        graph_distances = dict(nx.all_pairs_shortest_path_length(graph, cutoff=3))

        # Axes atoms that are in the molecule might be distant from the center
        # so we need to include the distances for these nodes as well.
        for axes_atom in ref_atoms_in_graph[1:]:
            axes_atom_distances = nx.single_source_shortest_path_length(graph, axes_atom)
            for target_atom, dist in axes_atom_distances.items():
                graph_distances[axes_atom][target_atom] = dist
                graph_distances[target_atom][axes_atom] = dist

        # We traverse the graph breadth first.
        for _, added_atom in nx.bfs_edges(graph, source=ref_atoms_in_graph[0]):
            # This might be an axes atoms that we have already added above.
            if added_atom.index in ref_atom_indices_in_graph[1:]:
                continue

            # Initialize Z-matrix row.
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
                assert len(z_matrix) < 4
                z_matrix_row = z_matrix_row + [-1] * (4-len(z_matrix_row))

            # Add entry to Z-matrix.
            z_matrix.append(z_matrix_row)

            # Add this atom to those added to the Z-matrix.
            atoms_order[added_atom.index] = len(atoms_order)

        return z_matrix

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

            # The minus sign of the atom order is because we want to prioritize
            # atoms that have just been added.
            priorities.append([
                prev_atom.index,
                dist,
                bond_atom_dist,
                -atoms_order[prev_atom.index],
                float(not is_h and _is_hydrogen(prev_atom)),
            ])

        # Sort the atoms by priority. The first element of the tuple is the atom
        # index. It is not used to assign priorities.
        priorities.sort(key=lambda k: tuple(k[1:]))
        return priorities

    def _analyze_dataset(
            self,
            z_matrix: Sequence[Sequence[int]],
            cartesian_to_mixed_flow: torch.nn.Module,
    ) -> Tuple[torch.Tensor]:
        """Check the Z-matrix robustness and compute the minimum and maximum value of each DOF in the trajectory.

        This function goes through the dataset and analyzes the structures to
        check that the angles in the Z-matrix are not defined by collinear angles
        and to compute the min/max values of the DOFs.

        The returned min/max values are for the coordinates in the relative
        frame of reference. This is useful to calculate appropriate initial
        values for the left/rightmost nodes of the neural spline transformer
        (e.g., for Cartesian coordinates).

        For this, we need to calculate the minimum and maximum dof AFTER it has
        gone through the partial flow removing the fixed atoms and the relative
        frame of reference has been set by _CartesianToMixedFlow since this is
        the input that will be passed to the transformers.

        Parameters
        ----------
        z_matrix : Sequence[Sequence[int]]
            The Z-matrix in the same format passed to _CartesianToMixedFlow.
            Atom indices must refer to those after the fixed atoms have been
            removed.
        cartesian_to_mixed_flow : torch.nn.Module
            The flow used to convert Cartesian into mixed coordinates.

        Returns
        -------
        min_dofs : torch.Tensor
            ``min_dofs[i]`` is the minimum value of the ``dof_indices[i]``-th
            degree of freedom.
        max_dofs : torch.Tensor
            ``max_dofs[i]`` is the maximum value of the ``dof_indices[i]``-th
            degree of freedom.

        """
        # This is needed to check for collinearity of the reference atoms.
        ref_atoms = self.get_reference_atoms_indices(remove_fixed=True)

        # We temporarily set the flow to the partial flow to process the positions.
        assert self._flow is None
        identity_flow = lambda x_: (x_, torch.zeros_like(x_[:, 0]))
        self._flow = self.create_partial_flow(identity_flow, return_partial=True)

        # Read the trajectory in batches.
        batch_size = 1024
        max_n_samples = 5 * batch_size
        dataset = self.create_dataset()
        if len(dataset) > max_n_samples:
            step = int(np.ceil(len(dataset) / max_n_samples))
            indices = list(range(0, len(dataset), step))
            dataset = torch.utils.data.Subset(dataset, indices)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, drop_last=False
        )
        for batch_data in data_loader:
            # Go through the partial flow.
            batch_positions = self(batch_data)['positions']

            # Test collinearity for the Z-matrix for these samples.
            batch_atom_positions = flattened_to_atom(batch_positions)
            for row_idx, zmatrix_row in enumerate(z_matrix):
                if (is_collinear(batch_atom_positions[:, zmatrix_row[:3]]) or
                        is_collinear(batch_atom_positions[:, zmatrix_row[1:]])):
                    raise RuntimeError(f'Row {row_idx+1} have collinear atoms.')

            # Test collinearity reference frame atoms.
            if is_collinear(batch_atom_positions[:, ref_atoms]):
                raise RuntimeError('Axes atoms are collinear!')

            # Go through the coordinate conversion flow.
            dofs = cartesian_to_mixed_flow.cartesian_to_mixed(batch_positions)[0]

            # Take the min/max across the batch of the selected DOFs.
            batch_min = torch.min(dofs, dim=0).values
            batch_max = torch.max(dofs, dim=0).values

            # Update current min/max.
            try:
                min_dofs = torch.minimum(min_dofs, batch_min)
                max_dofs = torch.maximum(max_dofs, batch_max)
            except NameError:  # First iteration.
                min_dofs = batch_min
                max_dofs = batch_max

        # Reset flow.
        self._flow = None

        return min_dofs, max_dofs

    def _get_transformer(
            self,
            cartesian_to_mixed_flow: torch.nn.Module,
            min_dof_vals: torch.Tensor,
            max_dof_vals: torch.Tensor,
            dof_indices: dict[str, Optional[torch.Tensor]],
    ) -> torch.nn.Module:
        """Return the transformer for the MAF.

        Parameters
        ----------
        cartesian_to_mixed_flow : torch.nn.Module
            The flow used to convert Cartesian into mixed coordinates.
        min_dof_vals : torch.Tensor
            Minimum values observed in the trajectory for the DOFs in mixed coordinates.
        max_dof_vals : torch.Tensor
            Maximum values observed in the trajectory for the DOFs in mixed coordinates.
        dof_indices : dict[str, Optional[torch.Tensor]]
            The indices of the DOFs after the conversion to mixed coordinates
            ``_CartesianToMixedFlow.get_dof_indices_by_type()``.

        """
        # We need to determine the limits only for the mapped (not conditioning)
        # DOFs since that's what the NeuralSplineTransformer will modify.
        # Moreover, for angles, these limits are fixed. We still compute them
        # for all DOFs because it's easier and then filter out conditioning and
        # angle DOFs.
        x0 = min_dof_vals
        xf = max_dof_vals

        # Set the limits for the bonds.
        x0[dof_indices['distances']] = torch.max(
            torch.tensor(0.0),
            x0[dof_indices['distances']] - self.hparams.distance_lower_limit_displacement,
        )

        # Now filter all conditioning dofs.
        if dof_indices['conditioning'] is not None:
            mask = torch.isin(torch.arange(cartesian_to_mixed_flow.n_dofs_out),
                              dof_indices['conditioning'], assume_unique=True, invert=True)
            x0 = x0[mask]
            xf = xf[mask]

            # We need to remove (and shift) the conditioning DOF from the type
            # indices. Copy to avoid modifying the input argument.
            dof_indices = dof_indices.copy()
            for idx_type in ['distances', 'angles', 'torsions', 'cartesians', 'reference']:
                dof_indices[idx_type] = remove_and_shift_sorted_indices(
                    dof_indices[idx_type], removed_indices=dof_indices['conditioning'])

        # Initialize the transformers. We set the limits of angles from 0 to 1
        # since they are normalized. Only torsions are periodic since valence
        # bond angles are [0, pi].
        assert cartesian_to_mixed_flow.rel_ic.normalize_angles
        transformer_indices = [
            dof_indices['distances'],
            dof_indices['angles'],
            dof_indices['torsions'],
        ]
        transformers = [
            tfep.nn.transformers.NeuralSplineTransformer(
                x0=x0[dof_indices['distances']].detach(),
                xf=xf[dof_indices['distances']].detach(),
                n_bins=5,
                circular=False,
                identity_boundary_slopes=True,
                learn_lower_bound=False,
                learn_upper_bound=True,
            ),
            tfep.nn.transformers.NeuralSplineTransformer(
                x0=torch.zeros(len(dof_indices['angles'])),
                xf=torch.ones(len(dof_indices['angles'])),
                n_bins=5,
                circular=False,
                identity_boundary_slopes=False,
                learn_lower_bound=False,
                learn_upper_bound=False,
            ),
            tfep.nn.transformers.NeuralSplineTransformer(
                x0=torch.zeros(len(dof_indices['torsions'])),
                xf=torch.ones(len(dof_indices['torsions'])),
                n_bins=5,
                circular=True,
                identity_boundary_slopes=False,
                learn_lower_bound=False,
                learn_upper_bound=False,
            ),
        ]

        # If there is only 1 fragment and rototranslational DOFs are removed
        # there are no Cartesian coordinates.
        if len(dof_indices['cartesians']) > 0:
            transformers.append(tfep.nn.transformers.NeuralSplineTransformer(
                x0=x0[dof_indices['cartesians']].detach(),
                xf=xf[dof_indices['cartesians']].detach(),
                n_bins=5,
                circular=False,
                identity_boundary_slopes=True,
                learn_lower_bound=True,
                learn_upper_bound=True,
            ))
            transformer_indices.append(dof_indices['cartesians'])

        # If rototranslational DOFs are removed, this is empty.
        if len(dof_indices['reference']) > 0:
            # The rototranslational DOFs are always 0. When this happen, a neural
            # spline can learn an identity function with an arbitrary Jacobian
            # adding a spurious contribution so we use a volume-preserving shift
            # transformer.
            transformers.append(tfep.nn.transformers.VolumePreservingShiftTransformer())
            transformer_indices.append(dof_indices['reference'])

        return tfep.nn.transformers.MixedTransformer(
            transformers=transformers,
            indices=transformer_indices,
        )

    def _get_maf_degrees_in(self, n_dofs_in, maf_dof_indices):
        """Assign degrees to each input DOFs.

        Returns a pair [degrees_in_ascending, degrees_in_descending] input
        arguments for MAF in the ascending and descending order.

        The rototranslational DOFs (if not removed) are always assigned the
        last degree, both in ascending and descending order, since they are
        constant.

        """
        # If we don't remove translation/rotation, we assign their degree later.
        # To make this work with generate_degrees() we set them as conditioning
        # which makes it temporarily assign a -1 degree to them.
        if len(maf_dof_indices['reference']) > 0:
            # Avoid modifying the original maf_dof_indices variable.
            maf_dof_indices = maf_dof_indices.copy()
            if maf_dof_indices['conditioning'] is None:
                maf_dof_indices['conditioning'] = torch.tensor([]).to(maf_dof_indices['reference'])

            # We always remove translation/rotation when reference atoms are
            # conditioning so these are not repeated indices.
            maf_dof_indices['conditioning'] = torch.cat([
                maf_dof_indices['conditioning'],
                maf_dof_indices['reference'],
            ])

        # Create the degrees for the ascending and descending order.
        degrees_in = []
        for order in ['ascending', 'descending']:
            degrees_in.append(tfep.nn.conditioners.made.generate_degrees(
                n_features=n_dofs_in,
                order=order,
                conditioning_indices=maf_dof_indices['conditioning'],
            ))

        # Now add the degree of the reference.
        last_degree = torch.max(degrees_in[0]) + 1
        for degrees in degrees_in:
            degrees[maf_dof_indices['reference']] = last_degree

        return degrees_in


# =============================================================================
# HELPER CLASSES AND FUNCTIONS TO CONSTRUCT THE Z-MATRIX
# =============================================================================

def check_independent(z_matrix):
    """Check that rows of the Z-matrix do not depend on the same atoms.

    This raises an error if the coordinates of two atoms in the Z-matrix depend
    on the same set of other atoms and, in particular, have the same bond atom.

    This check was implemented in https://github.com/noegroup/bgmol

    """
    dependent_rows = []
    all234 = [(torsion[1], set(torsion[2:])) for torsion in z_matrix]
    for i, other in enumerate(all234):
        if other in all234[:i]:
            dependent_rows.append(i)

    if len(dependent_rows) > 1:
        err_msg = 'The following Z-matrix rows are not independent:\n'
        for i in dependent_rows:
            err_msg += f'\tRow {i}: {z_matrix[i]}\n'
        raise RuntimeError(err_msg)


def is_collinear(points, tol=1e-2):
    """Check that three points are not collinear.

    Parameters
    ----------
    points : torch.Tensor
        Shape ``(batch_size, 3, 3)``. The three points.
    tol : float
        Numerical tolerance for checking collinearity.

    Returns
    -------
    is_collinear : bool
        ``True`` if three points are collinear. ``False`` otherwise.

    """
    # Divide the three points
    p0, p1, p2 = points.transpose(0, 1)
    # tol in the same units of p0, p1, and p2.
    v01 = torch.nn.functional.normalize(p1 - p0, dim=-1)
    v12 = torch.nn.functional.normalize(p2 - p1, dim=-1)
    return torch.any(torch.isclose(torch.abs(batchwise_dot(v01, v12)),
                                   torch.tensor(1.0), atol=tol, rtol=0.))


def _is_hydrogen(atom):
    """Return True if the atom is a hydrogen.

    atom is an MDAnalysis.Atom.

    Raises an exception if the atom has no information on the element.

    """
    err_msg = ("The topology files have no information on the atom elements. "
               "This is required to infer a robust Z-matrix. You can either "
               "provide a topology file that includes this info (e.g., a PDB) "
               "or add this information programmatically by overwriting "
               "MixedMAFMap.create_universe() and set, e.g., "
               "universe.add_TopologyAttr('element', ['O', 'H', 'C', ...]).")

    try:
        element = atom.element
    except MDAnalysis.exceptions.NoDataError as e:
        raise ValueError(err_msg) from e
    else:
        element = element.upper()
    # In some cases there is an element attribute but it's empty.
    if element == '':
        raise ValueError(err_msg)
    return element == 'H'


# =============================================================================
# COORDINATE CONVERSION
# =============================================================================

class _CartesianToMixedFlow(torch.nn.Module):
    """Flow converting Cartesian to mixed (internal+Cartesian) coordinates.

    This class 1) converts Cartesian coordinates to mixed coordinates, 2)
    executes the wrapped flow, and 3) converts the coordinates back from mixed
    to Cartesian.

    While converting to mixed coordinates, the flow roto-translates the Cartesian
    coordinates to a relative frame of reference defined by the origin and axes
    atoms. Optionally, these roto-translational degrees of freedom can be removed
    (e.g., if the mapped system has roto-translational symmetry).

    """

    def __init__(
            self,
            flow: torch.nn.Module,
            cartesian_atom_indices: Sequence[int],
            z_matrix: Sequence[Sequence[int]],
            reference_atom_indices: torch.Tensor,
            remove_ref_rototranslation: Sequence[bool],
    ):
        """Constructor.

        Parameters
        ----------
        flow : torch.nn.Module
            The wrapped normalizing flow that takes as input the mixed
            coordinates.
        cartesian_atom_indices : Sequence[int]
            Shape ``(n_cartesian_atoms,)``. Indices of the atoms represented as
            Cartesian (both mapped and conditioning, and  after the fixed atoms
            have been removed. The indices must be ordered.
        z_matrix : np.ndarray
            Shape ``(n_ic_atoms, 4)``. The Z-matrix for the atoms represented as
            internal coordinates.
        reference_atom_indices : torch.Tensor
            The indices (after the fixed atoms are removed) of the origin, axis,
            and plane atoms (in this order) used to determine the relative
            frame of reference for the Cartesian coordinates.
        remove_ref_rototranslation : Sequence[bool]
            Shape (3,). Whether to remove the rototranslational degrees of
            freedom of the 3 reference atoms from the coordinates passed to the
            wrapped flow.

        """
        from bgflow.nn.flow.crd_transform.ic import RelativeInternalCoordinateTransformation

        super().__init__()

        #: Wrapped flow.
        self.flow = flow
        self.register_buffer('remove_ref_rototranslation', torch.tensor(remove_ref_rototranslation))

        # Turn to tensor.
        z_matrix = ensure_tensor_sequence(z_matrix)
        cartesian_atom_indices = ensure_tensor_sequence(cartesian_atom_indices)
        reference_atom_indices = ensure_tensor_sequence(reference_atom_indices)

        # Move reference frame atoms at the end to simplify book-keeping.
        # Reference frame atoms are always Cartesian so we can use searchsorted.
        cartesian_atom_indices = remove_and_shift_sorted_indices(
            cartesian_atom_indices,
            removed_indices=reference_atom_indices.sort().values,
            remove=True,
            shift=False,
        )
        cartesian_atom_indices = torch.cat([cartesian_atom_indices, reference_atom_indices])

        # The modules performing the coordinate conversion.
        self.rel_ic = RelativeInternalCoordinateTransformation(
            z_matrix=z_matrix,
            fixed_atoms=cartesian_atom_indices,
            normalize_angles=True,
        )

    @property
    def cartesian_atom_indices(self) -> torch.Tensor:
        """The atom indices of the atoms treated in Cartesian coordinates."""
        return self.rel_ic.fixed_atoms

    @property
    def z_matrix(self) -> torch.Tensor:
        """torch.Tensor: The Z-matrix."""
        return self.rel_ic.z_matrix

    @property
    def n_ic_atoms(self) -> int:
        """int: Total number of atoms represented in internal coordinates (excluding reference atoms)."""
        return len(self.z_matrix)

    @property
    def n_dofs_out(self) -> int:
        """int: Number of degrees of freedom after the conversion to mixed coordinates."""
        return self._get_coords_indices_by_type()[-1].tolist()

    def _get_coords_indices_by_type(self):
        """The first index in the mixed coordinates for each type of coordinate."""
        # Cartesian atoms include also the 6 rototranslational DOFs.
        n_cartesians = 3 * len(self.cartesian_atom_indices) - 3
        if self.remove_ref_rototranslation[0]:
            n_cartesians -= 3
        if self.remove_ref_rototranslation[1]:
            n_cartesians -= 2
        if self.remove_ref_rototranslation[2]:
            n_cartesians -= 1

        # Mixed coordinates are obtained by concatenating coord types in this order:
        coords_sizes = [
            self.n_ic_atoms,  # bonds
            self.n_ic_atoms,  # angles
            self.n_ic_atoms,  # torsions
            1,  # d01
            1,  # d02
            1,  # a102
            n_cartesians, # cartesian (including rototranslational)
        ]
        return torch.cumsum(torch.tensor(coords_sizes), dim=0)

    def get_dof_indices_by_type(
            self,
            conditioning_atom_indices: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Return the indices of the different types of DOFs after conversion to mixed coordinates.

        Parameters
        ----------
        conditioning_atom_indices : torch.Tensor or None
            Shape (n_atoms,). The indices of the conditioning atoms (after the
            fixed atoms have been removed).

        Returns
        -------
        indices_by_type : dict[str, torch.Tensor]
            A dict of indices grouped by type of DOF with keys: 'distances',
            'angles', 'torsions', 'cartesians', 'conditioning', and 'reference'
            (non-empty only if ``(~remove_ref_rototranslation).any()`` is
            ``True``). Note that the indices in 'conditioning' and the other
            types will overlap (i.e., both mapped and conditioning DOF indices
            are included in 'distances', 'angles', etc.).

        """
        coords_indices = self._get_coords_indices_by_type()
        indices_by_type = {
            'distances': torch.arange(coords_indices[0]),
            'angles': torch.arange(coords_indices[0], coords_indices[1]),
            'torsions': torch.arange(coords_indices[1], coords_indices[2]),
            'd01': torch.arange(coords_indices[2], coords_indices[3]),
            'd02': torch.arange(coords_indices[3], coords_indices[4]),
            'a102': torch.arange(coords_indices[4], coords_indices[5]),
            'cartesians': torch.arange(coords_indices[5], coords_indices[6]),
        }

        # Distances and angles must include also the reference frame atom IC coords.
        indices_by_type['distances'] = torch.cat([indices_by_type['distances'],
                                                  indices_by_type['d01'],
                                                  indices_by_type['d02']])
        indices_by_type['angles'] = torch.cat([indices_by_type['angles'],
                                               indices_by_type['a102']])

        # Rototranslational degrees of freedom, if not removed, are always 0.0
        # and must be treated specially by the flow. These are at the end of cartesians.
        n_refs_in_cartesians = 0
        if not self.remove_ref_rototranslation[0]:
            n_refs_in_cartesians += 3
        if not self.remove_ref_rototranslation[1]:
            n_refs_in_cartesians += 2
        if not self.remove_ref_rototranslation[2]:
            n_refs_in_cartesians += 1

        if n_refs_in_cartesians > 0:
            indices_by_type['reference'] = indices_by_type['cartesians'][-n_refs_in_cartesians:]
            indices_by_type['cartesians'] = indices_by_type['cartesians'][:-n_refs_in_cartesians]
        else:
            indices_by_type['reference'] = torch.tensor([]).to(indices_by_type['cartesians'])

        # Check if there are no conditioning atoms.
        if conditioning_atom_indices is None:
            indices_by_type['conditioning'] = None
        else:
            conditioning_atom_indices_set = set(conditioning_atom_indices.tolist())

            # Conditioning atoms are always Cartesian. Find their DOF indices
            # in indices_by_type['cartesians'] (excluding reference atoms).
            indices = [i for i, v in enumerate(self.cartesian_atom_indices[:-3].tolist())
                       if v in conditioning_atom_indices_set]
            indices = atom_to_flattened_indices(torch.tensor(indices).to(indices_by_type['cartesians']))
            cond_dof_indices = indices_by_type['cartesians'][indices]

            # Rototranslational DOFs of conditioning reference atoms are always
            # removed so we need to check only their internal coords.
            cond_dof_indices = [cond_dof_indices]
            axis_atom_idx, plane_atom_idx = self.cartesian_atom_indices[-2:].tolist()
            if axis_atom_idx in conditioning_atom_indices_set:
                cond_dof_indices.append(indices_by_type['d01'])
            if plane_atom_idx in conditioning_atom_indices_set:
                cond_dof_indices.extend([indices_by_type['d02'], indices_by_type['a102']])
            indices_by_type['conditioning'] = torch.cat(cond_dof_indices).sort().values

            if len(indices_by_type['conditioning']) == 0:
                indices_by_type['conditioning'] = None

        return indices_by_type

    def forward(self, x):
        return self._pass(x, inverse=False)

    def inverse(self, y):
        return self._pass(y, inverse=True)

    def _pass(self, x, inverse):
        # Convert from Cartesian to mixed coordinates.
        y, cumulative_log_det_J, origin_atom_position, rotation_matrix = self.cartesian_to_mixed(x)

        # Run flow.
        if inverse:
            y, log_det_J = self.flow.inverse(y)
        else:
            y, log_det_J = self.flow(y)
        cumulative_log_det_J = cumulative_log_det_J + log_det_J

        # Convert from mixed to Cartesian coordinates.
        y, log_det_J = self.mixed_to_cartesian(y, origin_atom_position, rotation_matrix)
        cumulative_log_det_J = cumulative_log_det_J + log_det_J

        return y, cumulative_log_det_J

    def cartesian_to_mixed(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        """Convert from Cartesian to mixed coordinates.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch_size, n_atoms*3)`` or ``(batch_size, n_atoms, 3)``.
            The Cartesian coordinates.

        Returns
        -------
        y : torch.Tensor
            Shape ``(batch_size, n_atoms*3 - fixed_dofs)`` where ``fixed_dof``
            can be 0, 3, or 6 depending on the value of ``self.remove_ref_rototranslation``
            The mixed coordinates.
        log_det_J : torch.Tensor
            Shape ``(batch_size,)``. The logarithm of the absolute Jacobian
            determinant of the coordinate transformation.
        origin_atom_position : torch.Tensor
            Shape ``(batch_size, 1, 3)``. The position of the origin of the
            relative frame of reference in the original coordinate system.
        rotation_matrix : torch.Tensor
            Shape ``(batch_size, 1, 4)``. A rotation matrix representing the
            rotation performed to transform the original coordinate system into
            the relative one.

        """
        from bgflow.nn.flow.crd_transform.ic import normalize_torsions

        # Convert to mixed coordinates.
        bonds, angles, torsions, x_cartesian, cumulative_log_det_J = self.rel_ic(x)

        # From (batch, 1) to (batch,).
        cumulative_log_det_J = cumulative_log_det_J.squeeze(-1)

        # From flattened to atom representation.
        x_cartesian = flattened_to_atom(x_cartesian)
        batch_size, n_cartesian_atoms, _ = x_cartesian.shape

        # Center the Cartesian coordinates on the origin atom.
        origin_atom_idx, axis_atom_idx, plane_atom_idx = -3, -2, -1
        origin_atom_position = x_cartesian[:, origin_atom_idx]
        x_cartesian = x_cartesian - origin_atom_position.unsqueeze(1)

        # Find rotation matrix to re-orient the frame of reference.
        rotation_matrix = reference_frame_rotation_matrix(
            axis_atom_positions=x_cartesian[:, axis_atom_idx],
            plane_atom_positions=x_cartesian[:, plane_atom_idx],
            axis=get_axis_from_name('x').to(x_cartesian),
            plane_axis=get_axis_from_name('y').to(x_cartesian),
            # We can project on positive axis since the neural spline
            # never makes the coordinate negative.
            project_on_positive_axis=True,
        )

        # Rotate all Cartesian coordinates.
        x_cartesian = batchwise_rotate(x_cartesian, rotation_matrix)

        # The (positive) x-coordinate of the axis atom is the distance from the origin.
        d01 = x_cartesian[:, axis_atom_idx, 0]

        # Transform the coordinates on the x-y plane to polar (distance, angle) from the origin.
        d02, a102, log_det_J = cartesian_to_polar(
            x_cartesian[:, plane_atom_idx, 0],
            x_cartesian[:, plane_atom_idx, 1],
            return_log_det_J=True
        )
        cumulative_log_det_J = cumulative_log_det_J + log_det_J

        # Normalize the angle. angle is in [-pi, pi] so we use normalize_torsions().
        # Normalize torsion takes and return shape (batch_size, 1), not (batch_size,).
        a102, log_det_J = normalize_torsions(a102.unsqueeze(-1))
        a102 = a102.squeeze(-1)
        cumulative_log_det_J = cumulative_log_det_J + log_det_J.squeeze(-1)

        # From (batch, n_atoms, 3) to (batch, n_atoms*3).
        x_cartesian = atom_to_flattened(x_cartesian)

        # Remove rototranslational DOFs.
        mask = torch.full((x_cartesian.shape[1],), True).to(x_cartesian.device)
        if self.remove_ref_rototranslation[0]:
            mask[-9:-6] = False
        if self.remove_ref_rototranslation[1]:
            mask[-6:-3] = False
        else:
            mask[-6:-5] = False
        if self.remove_ref_rototranslation[2]:
            mask[-3:] = False
        else:
            mask[-3:-1] = False
        x_cartesian = x_cartesian[:, mask]

        # Concatenate all DOFs.
        y = torch.cat([
            bonds,
            angles,
            torsions,
            d01.unsqueeze(1),
            d02.unsqueeze(1),
            a102.unsqueeze(1),
            x_cartesian,
        ], dim=-1)

        return y, cumulative_log_det_J, origin_atom_position, rotation_matrix

    def mixed_to_cartesian(
            self,
            y: torch.Tensor,
            origin_atom_position: torch.Tensor,
            rotation_matrix: torch.Tensor,
    ) -> Tuple[torch.Tensor]:
        """Convert from mixed to Cartesian.

        Parameters
        ----------
        y : torch.Tensor
            Shape ``(batch_size, n_atoms*3 - fixed_dofs)`` where ``fixed_dof``
            can be 0, 3, or 6 depending on the value of ``self.remove_ref_rototranslation``
            The mixed coordinates.
        origin_atom_position : torch.Tensor
            Shape ``(batch_size, 1, 3)``. The position of the origin of the
            relative frame of reference in the original coordinate system.
        rotation_matrix : torch.Tensor
            Shape ``(batch_size, 1, 4)``. A rotation matrix representing the
            rotation performed to transform the original coordinate system into
            the relative one.

        Returns
        -------
        x : torch.Tensor
            Shape ``(batch_size, n_atoms*3)`` or ``(batch_size, n_atoms, 3)``.
            The Cartesian coordinates.
        log_det_J : torch.Tensor
            Shape ``(batch_size,)``. The logarithm of the absolute Jacobian
            determinant of the coordinate transformation.

        """
        from bgflow.nn.flow.crd_transform.ic import unnormalize_torsions
        batch_size = y.shape[0]

        # Separate the different types of coordinates.
        bonds, angles, torsions, d01, d02, a102, y_cartesian = torch.tensor_split(
            y, self._get_coords_indices_by_type()[:-1], dim=-1)

        # Unnormalize the angle.
        a102, cumulative_log_det_J = unnormalize_torsions(a102)

        # From shape (batch_size, 1) to (batch_size,).
        d01 = d01.squeeze(-1)
        d02 = d02.squeeze(-1)
        a102 = a102.squeeze(-1)

        # Create an array that includes the removed rototranslational DOFs.
        n_cartesian_atoms_full = len(self.cartesian_atom_indices)
        y_cartesian_full = torch.empty(batch_size, n_cartesian_atoms_full, 3).to(y_cartesian)

        # Convert internal coords of the axes atoms to Cartesian.
        origin_atom_idx, axis_atom_idx, plane_atom_idx = -3, -2, -1
        y_cartesian_full[:, axis_atom_idx, 0] = d01

        # The plane atom lies on the xy-plane.
        plane_atom_x, plane_atom_y, log_det_J = polar_to_cartesian(
            d02,
            a102,
            return_log_det_J=True,
        )
        y_cartesian_full[:, plane_atom_idx, 0] = plane_atom_x
        y_cartesian_full[:, plane_atom_idx, 1] = plane_atom_y
        cumulative_log_det_J = cumulative_log_det_J + log_det_J

        # Add back the translational DOFs.
        mask = torch.full((n_cartesian_atoms_full, 3), True).to(y_cartesian.device)
        if self.remove_ref_rototranslation[0]:
            y_cartesian_full[:, origin_atom_idx] = 0.0
            mask[origin_atom_idx] = False

        # Add back the rotational DOFs.
        if self.remove_ref_rototranslation[1]:
            y_cartesian_full[:, axis_atom_idx, 1:] = 0.0
            mask[axis_atom_idx] = False
        else:
            mask[axis_atom_idx, 0] = False
        if self.remove_ref_rototranslation[2]:
            y_cartesian_full[:, plane_atom_idx, 2] = 0.0
            mask[plane_atom_idx] = False
        else:
            mask[plane_atom_idx, :2] = False

        # Forward all other coordinates. Applying a 2d mask flattens that dimension.
        y_cartesian_full[:, mask] = y_cartesian
        y_cartesian_full = y_cartesian_full.reshape(batch_size, -1, 3)

        # Rotate and translate the Cartesian back to the global frame of reference.
        y_cartesian_full = batchwise_rotate(y_cartesian_full, rotation_matrix, inverse=True)
        y_cartesian_full = y_cartesian_full + origin_atom_position.unsqueeze(1)

        # Convert internal coords back to Cartesian.
        x, log_det_J = self.rel_ic(bonds, angles, torsions, y_cartesian_full, inverse=True)
        cumulative_log_det_J = cumulative_log_det_J + log_det_J.squeeze(-1)

        return x, cumulative_log_det_J
