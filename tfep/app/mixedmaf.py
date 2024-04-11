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
from tfep.utils.misc import atom_to_flattened_indices, atom_to_flattened, flattened_to_atom


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

    The coordinates of the mapped molecules are transformed into a mixed
    Cartesian/internal coordinate representation based on a Z-matrix before going
    through the MAF.

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

    Optionally, the flow can map the atoms in a relative frame of reference
    which based on the position of an ``origin_atom`` and two ``axes_atoms``
    that determine the origin and the orientation of the axes, respectively.
    When given, these atoms are prioritized for the choice of the first three
    atoms of a molecule's Z-matrix. Furthermore, if ``auto_reference_frame``
    is set to ``True``, the class will determine the frame of reference automatically
    based on the first three atoms of the Z-matrix. In this case, if the origin
    atom is mapped, it will be automatically set to be conditioning.

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
            tfep_logger_dir_path: str = 'tfep_logs',
            auto_reference_frame: bool = False,
            n_maf_layers: int = 6,
            bond_limits: Optional[Tuple[pint.Quantity]] = None,
            max_cartesian_displacement: Optional[pint.Quantity] = None,
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
            ``x`` and ``y``.

            These atoms can be either conditioning or mapped. ``axes_atoms[0]``
            has only 1 degree of freedom (DOF) while ``axes_atoms[1]`` has 2.
            Whether these DOFs are mapped or not depends on whether their atoms
            are indicated as mapped or conditioning, respectively.
        tfep_logger_dir_path : str, optional
            The path where to save TFEP-related information (potential energies,
            sample indices, etc.).
        auto_reference_frame : bool, optional
            If ``True``, the class automatically determines a relative frame of
            reference based on the first three atoms entering the Z-matrix. Both
            ``origin_atom`` and ``axes_atoms`` must be ``None`` for this. Note
            that the origin atom, if mapped, will become effectively conditioning
            because its position will be maintained.

            This is useful, for example, in vacuum, where the potential is invariant
            to rototranslations of the molecule and the system effectively has 3*N-6
            degrees of freedom.
        n_maf_layers : int, optional
            The number of MAF layers.
        bond_limits : Tuple[pint.Quantity] or None
            The minimum and maximum bond length used to set the limits to map
            bonds with the neural spline transformer. Default is [0.5, 3.0] Angstrom.
        max_cartesian_displacement : pint.Quantity or None
            Cartesian coordinates are mapped with neural spline transformers and
            their limits are set to ``min_value-max_cartesian_displacement`` and
            ``max_value-max_cartesian_displacement``, where ``min/max_value`` are
            the minimum and maximum value observed in the entire dataset for that
            particular degree of freedom. Default is 3.0 Angstrom.
        dataloader_kwargs : Dict, optional
            Extra keyword arguments to pass to ``torch.utils.data.DataLoader``.
        **kwargs
            Other keyword arguments to pass to the constructor of :class:`tfep.nn.flows.MAF`.

        See Also
        --------
        `MDAnalysis Universe object <https://docs.mdanalysis.org/2.6.1/documentation_pages/core/universe.html#MDAnalysis.core.universe.Universe>`_

        """
        # Check input.
        if auto_reference_frame and (origin_atom is not None or axes_atoms is not None):
            raise ValueError('With auto_reference_frame=True both origin_atom and axes_atoms must be None.')

        # Handle mutable default values.
        positions_unit = potential_energy_func.positions_unit
        if bond_limits is None:
            bond_limits = [0.5, 3.0] * positions_unit._REGISTRY.angstrom
        if max_cartesian_displacement is None:
            max_cartesian_displacement = 3.0 * positions_unit._REGISTRY.angstrom

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
        self.save_hyperparameters('n_maf_layers')
        self._auto_reference_frame = auto_reference_frame
        self._kwargs = kwargs

        # Convert limits to input units.
        self._bond_limits = bond_limits.to(positions_unit).magnitude
        self._max_cartesian_displacement = max_cartesian_displacement.to(positions_unit).magnitude

    def configure_flow(self) -> torch.nn.Module:
        """Initialize the normalizing flow.

        Returns
        -------
        flow : torch.nn.Module
            The normalizing flow.

        """
        # Determine Z-matrix and Cartesian atoms and (optionally)
        # automatically determine origin and axes atoms.
        cartesian_atom_indices, z_matrix, z_matrix_w_fixed, are_axes_atoms_bonded = self._build_z_matrix()
        if len(z_matrix) == 0:
            raise ValueError('There are no internal coordinates to map. '
                             'Consider using a Cartesian flow.')

        # Initialize _CartesianToMixedFlow that will map the MAF. We will set
        # the wrapped flow after we have initialized the MAF.
        origin_atom_idx, axes_atoms_indices = self.get_reference_atoms_indices(
            remove_fixed=True, separate_origin_axes=True)
        cartesian_to_mixed_flow = _CartesianToMixedFlow(
            flow=None,
            cartesian_atom_indices=cartesian_atom_indices,
            z_matrix=z_matrix,
            origin_atom_idx=origin_atom_idx,
            axes_atoms_indices=axes_atoms_indices,
        )

        # Now we take a pass at the trajectory to check that the Z-matrix is robust
        # (i.e., no collinear atoms determining angles) and compute the min/max values
        # of the DOFs (after going through _CartesianToMixedFlow) to configure the
        # neural splines correctly.
        min_dof_vals, max_dof_vals = self._analyze_dataset(z_matrix_w_fixed, cartesian_to_mixed_flow)

        # Determine the conditioning DOFs after going through _CartesianToMixedFlow.
        # conditioning_atom_indices must have the indices after the fixed atoms are removed.
        conditioning_atom_indices = self.get_conditioning_indices(idx_type='atom', remove_fixed=True)

        maf_conditioning_dof_indices = cartesian_to_mixed_flow.get_maf_conditioning_dof_indices(
            conditioning_atom_indices=conditioning_atom_indices)

        # Determine the periodic degrees of freedom (i.e., angles and torsions)
        # to pass to the MAF layer.
        maf_periodic_dof_indices = cartesian_to_mixed_flow.get_maf_periodic_dof_indices()

        # Create the transformer.
        transformer = self._get_transformer(
            cartesian_to_mixed_flow=cartesian_to_mixed_flow,
            min_dof_vals=min_dof_vals,
            max_dof_vals=max_dof_vals,
            maf_conditioning_dof_indices=maf_conditioning_dof_indices,
            maf_periodic_dof_indices=maf_periodic_dof_indices,
            are_axes_atoms_bonded=are_axes_atoms_bonded,
        )

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
                **self._kwargs,
            ))
        flow = tfep.nn.flows.SequentialFlow(*maf_layers)

        # Wrap the MAF into the _CartesianToMixedFlow.
        cartesian_to_mixed_flow.flow = flow
        return cartesian_to_mixed_flow

    def _build_z_matrix(self):
        """Determine the Z-matrix, the Cartesian atoms, and (optionally) the automatic frame of reference.

        See the class docstring for an overview of how the Z-matrix is determined.

        The Z-matrix is constructed so that the origin and axes atoms are always
        included among the Cartesian atoms.

        If self._auto_reference_frame is True, this method also sets self._origin_atom_idx
        and self._axes_atoms_indices.

        The returned indices refer to those after the fixed atoms are removed
        except for ``z_matrix_w_fixed``.

        Returns
        -------
        cartesian_atom_indices : numpy.ndarray
            Shape (n_cartesian_atoms,). The indices of the atoms represented by
            Cartesian coordinates (i.e., 3 reference atoms for each molecule,
            and all conditioning atoms). The array is sorted in ascending order.
        z_matrix : numpy.ndarray
            Shape (n_ic_atoms, 4). The Z-matrix for the atoms represented by
            internal coordinates. E.g., ``z_matrix[i] == [7, 2, 4, 8]``
            means that the distance, angle, and dihedral for atom ``7`` must be
            computed between atoms ``7-2``, ``7-2-4``, and ``7-2-4-8`` respectively.
        z_matrix_w_fixed : numpy.ndarray
            Same as z_matrix but the indices refer to the atoms before the fixed
            atoms have been removed.
        are_axes_atoms_bonded : List[bool] or None
            Whether the first and second axes atoms are bonded to the origin atom.

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
        ref_atoms = [self.dataset.universe.atoms[i] for i in ref_atom_indices]

        # Only the mapped atoms are represented using internal coordinates
        # so we build a set to check for membership.
        mapped_atom_indices_w_fixed_set = set(mapped_atom_indices_w_fixed.tolist())

        # Initialize returned values.
        cartesian_atom_indices = []
        system_z_matrix = []
        are_axes_atoms_bonded = None

        # Build the Z-matrix for each connected subgraph.
        for graph_nodes in nx.connected_components(system_graph):
            graph = system_graph.subgraph(graph_nodes).copy()

            # Update are_axes_atoms_bonded. The origin atom can belong only to a molecule.
            # Both origin and axes atoms need to be passed for this to be != None.
            if are_axes_atoms_bonded is None and len(ref_atoms) == 3 and ref_atoms[0] in graph:
                are_axes_atoms_bonded = [graph.has_edge(ref_atoms[0], ref_atoms[1]), graph.has_edge(ref_atoms[0], ref_atoms[2])]

            # Check if this molecule is composed only by conditioning atoms.
            graph_atom_indices = [node.ix for node in graph]
            if len(set(graph_atom_indices).intersection(mapped_atom_indices_w_fixed_set)) == 0:
                # Add everything to Cartesian atoms.
                cartesian_atom_indices.extend(graph_atom_indices)

                # Skip to next molecule.
                continue

            # Build the Z-matrix for this molecule.
            graph_z_matrix = self._build_connected_graph_z_matrix(graph, ref_atom_indices)

            # If requested, automatically determine the frame of reference. We need
            # to do it before we shift the indices to account for the removed fixed
            # atoms and before we remove the first three rows of the Z-matrix since
            # in the 4th row the order of origin/axes atoms is scrambled.
            if self._auto_reference_frame and self._origin_atom_idx is None:
                self._determine_reference_frame_atoms(graph_z_matrix, mapped_atom_indices_w_fixed_set)

            # Now separate the Cartesian atoms from the Z-matrix ones. First three
            # atoms are always Cartesian (or determine the automatic reference frame).
            cartesian_atom_indices.extend([row[0] for row in graph_z_matrix[:3]])

            # Only the mapped atoms are mapped as internal coordinates.
            for z_matrix_row in graph_z_matrix[3:]:
                if z_matrix_row[0] in mapped_atom_indices_w_fixed_set:
                    system_z_matrix.append(z_matrix_row)
                else:
                    cartesian_atom_indices.append(z_matrix_row[0])

            # Test independence.
            check_independent(graph_z_matrix)

        # The atom indices and the Z-matrix so far use the atom indices of the
        # systems before the fixed and reference atoms have been removed. Now
        # we need to map the indices to those after they are removed since these
        # are not passed to _CartesianToMixed._rel_ic.
        z_matrix_w_fixed = np.array(system_z_matrix)
        nonfixed_atom_indices_w_fixed = nonfixed_atom_indices_w_fixed.tolist()
        indices_map = {nonfixed_atom_indices_w_fixed[idx]: idx for idx in range(self.n_nonfixed_atoms)}

        # Convert indices.
        cartesian_atom_indices = [indices_map[i] for i in cartesian_atom_indices]
        for row_idx, z_matrix_row in enumerate(system_z_matrix):
            system_z_matrix[row_idx] = [indices_map[i] for i in z_matrix_row]

        # Sort atom indices and convert everything to numpy array (for RelativeInternalCoordinateTransformation).
        cartesian_atom_indices = np.array(cartesian_atom_indices)
        cartesian_atom_indices.sort()
        return cartesian_atom_indices, np.array(system_z_matrix), z_matrix_w_fixed, are_axes_atoms_bonded

    def _create_networkx_graph(self, atom_indices):
        """Return a networkx graph representing the given atoms."""
        # Select only the bonds in which both atoms are in the atom group.
        nonfixed_atoms = MDAnalysis.AtomGroup(atom_indices, self.dataset.universe)
        internal_bonds = [bond for bond in nonfixed_atoms.bonds
                          if (bond.atoms[0] in nonfixed_atoms and bond.atoms[1] in nonfixed_atoms)]

        # Build a networkx graph representing the topology of all the nonfixed atoms.
        system_graph = nx.Graph()
        system_graph.add_nodes_from(nonfixed_atoms)
        system_graph.add_edges_from(internal_bonds)

        return system_graph

    def _build_connected_graph_z_matrix(self, graph, ref_atom_indices):
        """Build the Z-matrix for a connected graph."""
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

    def _determine_reference_frame_atoms(self, z_matrix_w_fixed: List[int], mapped_atom_indices_w_fixed_set: Set[int]):
        """If requested, determine the origin and axes atoms.

        Both z_matrix and mapped_atom_indices must hold the indices of the atoms
        before the fixed atoms are removed.

        If the automatically-determined origin atom is a mapped atom, this converts
        it to a conditioning atom and updates the self._mapped_atom_indices,
        self._conditioning_atom_indices, and the argument mapped_atom_indices_w_fixed_set.

        """
        if not self._auto_reference_frame:
            return
        assert self._origin_atom_idx is None
        assert self._axes_atoms_indices is None

        # Set origin and axes atom indices as the first three atoms of the Z-matrix.
        # Keep Python integer value of origin idx before converting to tensor.
        origin_atom_idx = z_matrix_w_fixed[0][0]
        self._origin_atom_idx = torch.tensor(origin_atom_idx)
        self._axes_atoms_indices = torch.tensor([z_matrix_w_fixed[1][0], z_matrix_w_fixed[2][0]])

        # If the origin atom is mapped, we move it to the conditioning atom for consistency.
        if origin_atom_idx in mapped_atom_indices_w_fixed_set:
            logger.warning(f'Converting atom {origin_atom_idx} from mapped to conditioning '
                           'because it has been selected as the origin for the automatically '
                           'determined relative frame of reference for the flow.')

            # Remove from mapped.
            self._mapped_atom_indices = self._mapped_atom_indices[self._mapped_atom_indices != self._origin_atom_idx]

            # Insert in conditioning maintaining the order.
            if self._conditioning_atom_indices is None:
                self._conditioning_atom_indices = self._origin_atom_idx.unsqueeze(0).clone()
            else:
                insert_idx = torch.searchsorted(self._conditioning_atom_indices, self._origin_atom_idx)
                self._conditioning_atom_indices = torch.concatenate([
                    self._conditioning_atom_indices[:insert_idx],
                    self._origin_atom_idx.unsqueeze(0),
                    self._conditioning_atom_indices[insert_idx:],
                ])

            # Remove also from the set in place.
            mapped_atom_indices_w_fixed_set.remove(origin_atom_idx)

    def _analyze_dataset(
            self,
            z_matrix_w_fixed: np.ndarray,
            cartesian_to_mixed_flow: torch.nn.Module,
    ) -> Tuple[torch.Tensor]:
        """Check the Z-matrix robustness and compute the minimum and maximum value of each DOF in the trajectory.

        This function goes through the dataset and analyzes the structures to
        check that the angles in the Z-matrix are not defined by collinear angles
        and to compute the min/max values of the DOFs.

        The returned min/max values are for the coordinates in the relative
        frame of reference. This is useful to calculate appropriate values for
        the left/rightmost nodes of the neural spline transformer.

        For this, we need to calculate the minimum and maximum dof AFTER it has
        gone through the partial flow removing the fixed atoms and the relative
        frame of reference has been set by _CartesianToMixedFlow since this is
        the input that will be passed to the transformers.

        Parameters
        ----------
        z_matrix_w_fixed : np.ndarray
            The Z-matrix. The atom indices must refer to those before the fixed
            atoms have been removed.
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
        if self._origin_atom_idx is not None and self._axes_atoms_indices is not None:
            ref_atoms = torch.cat([self._origin_atom_idx.unsqueeze(0), self._axes_atoms_indices])
        else:
            ref_atoms = None

        # Create a flow removing the fixed and conditioning atoms.
        identity_flow = lambda x_: (x_, torch.zeros_like(x_[:, 0]))
        partial_flow = self.create_partial_flow(identity_flow, return_partial=True)

        # Read the trajectory in batches.
        dataset = self.create_dataset()
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=1024, shuffle=False, drop_last=False
        )
        for batch_data in data_loader:
            batch_positions = batch_data['positions']

            # Test collinearity for the Z-matrix for these samples.
            batch_atom_positions = flattened_to_atom(batch_positions)
            for row_idx, zmatrix_row in enumerate(z_matrix_w_fixed):
                if (is_collinear(batch_atom_positions[:, zmatrix_row[:3]]) or
                        is_collinear(batch_atom_positions[:, zmatrix_row[1:]])):
                    raise RuntimeError(f'Row {row_idx+1}: {zmatrix_row} have collinear atoms.')

            # Test collinearity reference frame atoms.
            if ref_atoms is not None and is_collinear(batch_atom_positions[:, ref_atoms]):
                raise RuntimeError('Axes atoms are collinear!')

            # Go through the coordinate conversion flow.
            dofs, _ = partial_flow(batch_positions)
            dofs = cartesian_to_mixed_flow._cartesian_to_mixed(dofs)[0]

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

        return min_dofs, max_dofs

    def _get_transformer(
            self,
            cartesian_to_mixed_flow: torch.nn.Module,
            min_dof_vals: torch.Tensor,
            max_dof_vals: torch.Tensor,
            maf_conditioning_dof_indices: Optional[torch.Tensor],
            maf_periodic_dof_indices: torch.Tensor,
            are_axes_atoms_bonded: Optional[Tuple[bool]],
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
        maf_conditioning_dof_indices : torch.Tensor or None
            The indices of the conditioning DOFs after the conversion to mixed
            coordinates.
        maf_periodic_dof_indices : torch.Tensor
            The indices of the periodic DOFs after the conversion to mixed
            coordinates.
        are_axes_atoms_bonded : Tuple[bool] or None
            Whether the first and second axes atoms are bonded to the origin atom..

        """
        # We need to determine the limits only for the mapped (not conditioning)
        # DOFs since that's what the NeuralSplineTransformer will act on. We compute
        # the limits for all DOFs because it's easier and then filter out the
        # conditioning DOFs. We assume everything is Cartesian and then we fix
        # the limits for angles and distances.
        x0 = min_dof_vals - self._max_cartesian_displacement
        xf = max_dof_vals + self._max_cartesian_displacement

        # Set the limits for the angles.
        assert cartesian_to_mixed_flow._rel_ic.normalize_angles
        x0[maf_periodic_dof_indices] = 0.0
        xf[maf_periodic_dof_indices] = 1.0

        # Set the limits for the bonds. The distances DOFs of the axes atoms
        # might not be bonds so we treat them separately.
        dof_indices = cartesian_to_mixed_flow.get_maf_distance_dof_indices(return_axes=False)
        x0[dof_indices] = self._bond_limits[0]
        xf[dof_indices] = self._bond_limits[1]

        # Set the limits for reference atom distance DOF (the angle has been taken
        # care above). dof_indices is an empty list if there are no axes atoms.
        dof_indices = cartesian_to_mixed_flow.get_maf_distance_dof_indices(return_bonds=False)
        for i, dof_idx in enumerate(dof_indices):
            if are_axes_atoms_bonded is not None and are_axes_atoms_bonded[i]:
                x0[dof_idx] = self._bond_limits[0]
                xf[dof_idx] = self._bond_limits[1]
            else:
                # The axis atom is not bonded to the origin but it's still a distance (i.e. > 0).
                x0[dof_idx] = max(0.0, min_dof_vals[dof_idx] - self._max_cartesian_displacement)

        # Now filter all conditioning dofs.
        if maf_conditioning_dof_indices is not None:
            maf_conditioning_dof_indices_set = set(maf_conditioning_dof_indices.tolist())
            mask = [i not in maf_conditioning_dof_indices_set for i in range(self.n_nonfixed_dofs)]
            x0 = x0[mask]
            xf = xf[mask]

            # We need to filter also for the periodic DOF indices.
            mask = [i not in maf_conditioning_dof_indices_set for i in maf_periodic_dof_indices.tolist()]
            maf_periodic_dof_indices = maf_periodic_dof_indices[mask]

            # The indices of circular refer to the indices of x0/xf. We need to
            # shift them to account for the removal of the conditioning DOFs.
            maf_periodic_dof_indices = maf_periodic_dof_indices - torch.searchsorted(maf_conditioning_dof_indices, maf_periodic_dof_indices)

        return tfep.nn.transformers.NeuralSplineTransformer(
            x0=x0.detach(),
            xf=xf.detach(),
            n_bins=5,
            circular=maf_periodic_dof_indices.detach(),
        )


# =============================================================================
# HELPER CLASSES AND FUNCTIONS
# =============================================================================

def check_independent(z_matrix):
    """Check that rows of the Z-matrix do not depend on the same atoms.

    This raises an error if the coordinates of two atoms in the Z-matrix depend
    on the same set of other atoms and, in particular, have the same bond atom.

    This check was implemented in https://github.com/noegroup/bgmol/tree/main.

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


# TODO: MOVE TO tfep.nn.flows?
class _CartesianToMixedFlow(torch.nn.Module):
    """Utility flow to convert from Cartesian to mixed Cartesian/internal coordinates.

    This also sets the relative frame of reference if origin/axes_atoms_indices are passed.

    """

    def __init__(
            self,
            flow: torch.nn.Module,
            cartesian_atom_indices: np.ndarray,
            z_matrix: np.ndarray,
            origin_atom_idx: Optional[torch.Tensor],
            axes_atoms_indices: Optional[torch.Tensor],
    ):
        """Constructor.

        All the indices must be after the fixed atoms have been removed.

        Parameters
        ----------
        flow : torch.nn.Module
            The normalizing flow to wrap that takes as input mixed coordinates.
        cartesian_atom_indices : np.ndarray
            Shape ``(n_cartesian_atoms,)``. Indices of the atoms represented as
            Cartesian (both mapped and conditioning) after the fixed atoms have
            been removed. This must be an ordered array.
        z_matrix : np.ndarray
            Shape ``(n_ic_atoms, 4)``. The Z-matrix for the atoms represented as
            internal coordinates.
        origin_atom_idx : torch.Tensor or None
            The index of the origin atom after the fixed atoms are removed. If
            given, the Cartesian coordinates will be translated to have this
            atom in the origin before being passed to the MAF.
        axes_atoms_indices : torch.Tensor or None
            The indices of the axes atoms determining the orientation of the
            reference frame after the fixed atoms are removed. If passed, the
            Cartesian coordinates will be rotated into this reference frame
            before being passed to the MAF.

        """
        from bgflow.nn.flow.crd_transform.ic import RelativeInternalCoordinateTransformation

        super().__init__()

        #: Wrapped flow.
        self.flow = flow

        # The modules performing the coordinate conversion.
        self._rel_ic = RelativeInternalCoordinateTransformation(
            z_matrix=z_matrix,
            fixed_atoms=cartesian_atom_indices,
            normalize_angles=True,
        )

        # Find the indices of the reference atoms in the cartesian tensor after
        # cartesian_to_mixed() is called.
        reference_atoms_indices_in_cartesian = []

        # Reference atoms (if present) are always in cartesian_atom_indices so we can use searchsorted.
        if origin_atom_idx is not None:
            idx = np.searchsorted(cartesian_atom_indices, [origin_atom_idx.tolist()])
            reference_atoms_indices_in_cartesian.append(idx[0])
        if axes_atoms_indices is not None:
            indices = np.searchsorted(cartesian_atom_indices, axes_atoms_indices.tolist())
            reference_atoms_indices_in_cartesian.extend(indices)

        # Convert to tensor.
        self.register_buffer('_reference_atoms_indices_in_cartesian',
                             torch.tensor(reference_atoms_indices_in_cartesian, dtype=int))

    @property
    def has_origin_atom(self) -> bool:
        """True if there is an origin atom for the relative frame of reference."""
        return len(self._reference_atoms_indices_in_cartesian) in {1, 3}

    @property
    def has_axes_atoms(self) -> bool:
        """True if there are axes atoms for the relative frame of reference."""
        return len(self._reference_atoms_indices_in_cartesian) > 1

    @property
    def cartesian_atom_indices(self) -> Optional[np.ndarray]:
        """The atom indices of the atoms treated in Cartesian coordinates."""
        return self._rel_ic.fixed_atoms

    @property
    def z_matrix(self) -> np.ndarray:
        """The Z-matrix."""
        return self._rel_ic.z_matrix

    @property
    def n_ic_atoms(self) -> int:
        """Number of atoms represented in internal coordinates."""
        return len(self.z_matrix)

    def get_maf_conditioning_dof_indices(self, conditioning_atom_indices: Optional[torch.Tensor]) -> torch.Tensor:
        """Return the indices of the conditioning DOFs after going through _CartesianToMixedFlow.

        Parameters
        ----------
        conditioning_atom_indices : torch.Tensor or None
            Shape (n_atoms,). The indices of the conditioning atoms (including
            the reference atoms) after the fixed atoms have been removed.

        Returns
        -------
        maf_conditioning_dof_indices : torch.Tensor or None
            The DOF indices after going through _CartesianToMixedFlow.

        """
        # If there are no conditioning atoms or if the only conditioning atom is
        # the origin atom, return None. The origin atom has no unconstrained DOFs
        # and it is removed by _CartesianToMixedFlow.
        if ((conditioning_atom_indices is None) or
                (len(conditioning_atom_indices) == 0) or
                (len(conditioning_atom_indices) == 1 and self.has_origin_atom)):
            return None

        # Convert the conditioning atom indices to their respective index in cartesian_atom_indices.
        conditioning_atom_indices_set = set(conditioning_atom_indices.tolist())
        conditioning_atom_indices_in_cartesian = [i for i, v in enumerate(self.cartesian_atom_indices)
                                                  if v in conditioning_atom_indices_set]

        # We need to treat reference atoms specially because _cartesian_to_mixed()
        # removes them from the cartesian atoms and places the unconstrained DOFs
        # of the axes atoms between the internal and the Cartesian coordinates.
        reference_atoms_indices_in_cartesian_set = set(self._reference_atoms_indices_in_cartesian.tolist())
        conditioning_atom_indices_in_cartesian_no_ref = [i for i in conditioning_atom_indices_in_cartesian
                                                         if i not in reference_atoms_indices_in_cartesian_set]
        conditioning_atom_indices_in_cartesian_no_ref = torch.tensor(
            conditioning_atom_indices_in_cartesian_no_ref).to(self._reference_atoms_indices_in_cartesian)

        # Shift indices due to the removed reference atoms. searchsorted requires sorted tensor.
        # We eventually will shift the indices to the right for the axes atoms DOFs later.
        reference_atoms_indices_in_cartesian_sorted = self._reference_atoms_indices_in_cartesian.sort()[0]
        conditioning_atom_indices_in_cartesian_no_ref = conditioning_atom_indices_in_cartesian_no_ref - torch.searchsorted(
            reference_atoms_indices_in_cartesian_sorted, conditioning_atom_indices_in_cartesian_no_ref)

        # Shift indices by the number of internal coordinates before the Cartesian ones.
        maf_conditioning_dof_indices = conditioning_atom_indices_in_cartesian_no_ref + self.n_ic_atoms

        # Convert from atom to DOF indices.
        maf_conditioning_dof_indices = atom_to_flattened_indices(maf_conditioning_dof_indices)

        # Axes atoms can be either mapped or conditioning. The DOFs of the axes
        # atoms are placed between the internal coordinates and the Cartesian atoms.
        if self.has_axes_atoms:
            # Whether the axes atoms are mapped or not, they have three degrees of
            # freedom that shift to the right the conditioning DOF indices.
            maf_conditioning_dof_indices = maf_conditioning_dof_indices + 3

            # We need to know whether the axes atoms are conditioning.
            conditioning_atom_indices_in_cartesian_set = set(conditioning_atom_indices_in_cartesian)
            axes_atoms_indices_in_cartesian = self._reference_atoms_indices_in_cartesian[-2:].tolist()

            # Collect the axes atoms DOFs that are conditioning.
            to_concatenate = []
            if axes_atoms_indices_in_cartesian[0] in conditioning_atom_indices_in_cartesian_set:
                to_concatenate.append(3*self.n_ic_atoms)
            if axes_atoms_indices_in_cartesian[1] in conditioning_atom_indices_in_cartesian_set:
                to_concatenate.extend([3*self.n_ic_atoms+1, 3*self.n_ic_atoms+2])

            # Concatenate reference and other conditioning DOFs.
            if len(to_concatenate) > 0:
                to_concatenate = torch.tensor(to_concatenate).to(maf_conditioning_dof_indices)
                maf_conditioning_dof_indices = torch.cat([to_concatenate, maf_conditioning_dof_indices])

        return maf_conditioning_dof_indices

    def get_maf_periodic_dof_indices(self) -> torch.Tensor:
        """Return the indices of the periodic DOF (angles and torsions) after the conversion from Cartesian to mixed."""
        # Except for the single angle defining the reference frame atoms, all
        # angles and torsions are placed right after the bonds.
        maf_periodic_dof_indices = list(range(self.n_ic_atoms, 3*self.n_ic_atoms))

        # Check if there are axes atoms, whose DOFs are placed between the
        # internal and Cartesian coordinates after the conversion.
        if self.has_axes_atoms:
            # The axes atoms are defined by two distances and 1 angle.
            maf_periodic_dof_indices.append(3*self.n_ic_atoms+2)

        return torch.tensor(maf_periodic_dof_indices)

    def get_maf_distance_dof_indices(self, return_bonds: bool = True, return_axes: bool = True) -> torch.Tensor:
        """Return the indices of the bond DOFs after the conversion from Cartesian to mixed.

        These do not include the two distances defining the coordinates of the axes atoms.

        """
        # The bonds are placed at the beginning.
        if return_bonds:
            maf_distance_dof_indices = list(range(self.n_ic_atoms))
        else:
            maf_distance_dof_indices = []

        # Check if there are axes atoms, whose DOFs are placed between the
        # internal and Cartesian coordinates after the conversion.
        if self.has_axes_atoms and return_axes:
            # The axes atoms are defined by two distances and 1 angle.
            maf_distance_dof_indices.extend([3*self.n_ic_atoms, 3*self.n_ic_atoms+1])

        return torch.tensor(maf_distance_dof_indices)

    def forward(self, x):
        return self._pass(x, inverse=False)

    def inverse(self, y):
        return self._pass(y, inverse=True)

    def _pass(self, x, inverse):
        # Convert from Cartesian to mixed coordinates.
        y, cumulative_log_det_J, origin_atom_position, rotation_matrix = self._cartesian_to_mixed(x)

        # Run flow.
        if inverse:
            y, log_det_J = self.flow.inverse(y)
        else:
            y, log_det_J = self.flow(y)
        cumulative_log_det_J = cumulative_log_det_J + log_det_J

        # Convert from mixed to Cartesian coordinates.
        y, log_det_J = self._mixed_to_cartesian(y, origin_atom_position, rotation_matrix)
        cumulative_log_det_J = cumulative_log_det_J + log_det_J

        return y, cumulative_log_det_J

    def _cartesian_to_mixed(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        """Convert from Cartesian to mixed coordinates."""
        # Convert to mixed coordinates.
        bonds, angles, torsions, x_cartesian, cumulative_log_det_J = self._rel_ic(x)

        # From (batch, 1) to (batch,).
        cumulative_log_det_J = cumulative_log_det_J.squeeze(-1)

        # From flattened to atom representation.
        x_cartesian = flattened_to_atom(x_cartesian)
        batch_size, n_cartesian_atoms, _ = x_cartesian.shape

        # We'll have to remove the reference atoms from the cartesian atoms using a mask.
        if self.has_origin_atom or self.has_axes_atoms:
            kept_atoms_mask = torch.full((n_cartesian_atoms,), True).to(
                self._reference_atoms_indices_in_cartesian.device)

        # Center the Cartesian coordinates on the origin atom.
        if not self.has_origin_atom:
            origin_atom_position = None
        else:
            # Flag origin atom to be removed.
            origin_atom_idx = self._reference_atoms_indices_in_cartesian[0]
            kept_atoms_mask[origin_atom_idx] = False

            # Save the position to restore it later.
            origin_atom_position = x_cartesian[:, origin_atom_idx]
            x_cartesian = x_cartesian - origin_atom_position.unsqueeze(1)

        # Re-orient the frame of reference.
        if not self.has_axes_atoms:
            rotation_matrix = None
            axes_atoms_dof = [torch.empty(0).to(self._reference_atoms_indices_in_cartesian)]
        else:
            axis_atom_idx = self._reference_atoms_indices_in_cartesian[-2]
            plane_atom_idx = self._reference_atoms_indices_in_cartesian[-1]
            axis_atom_pos = x_cartesian[:, axis_atom_idx]
            plane_atom_pos = x_cartesian[:, plane_atom_idx]

            # Find rotation matrix.
            rotation_matrix = reference_frame_rotation_matrix(
                axis_atom_positions=axis_atom_pos,
                plane_atom_positions=plane_atom_pos,
                axis=get_axis_from_name('x').to(x_cartesian),
                plane_axis=get_axis_from_name('y').to(x_cartesian),
                # We can project on positive axis since the neural spline
                # never makes the coordinate negative.
                project_on_positive_axis=True,
            )

            # Rotate all Cartesian coordinates.
            x_cartesian = batchwise_rotate(x_cartesian, rotation_matrix)

            # The (positive) x-coordinate of the axis atom is the distance from the origin.
            dist_axis_atom = x_cartesian[:, axis_atom_idx, 0]

            # Transform the coordinates on the x-y plane to polar (distance, angle) from the origin.
            dist_plane_atom, angle, log_det_J = cartesian_to_polar(
                x_cartesian[:, plane_atom_idx, 0],
                x_cartesian[:, plane_atom_idx, 1],
                return_log_det_J=True
            )
            cumulative_log_det_J = cumulative_log_det_J + log_det_J

            # Normalize the angle. angle is in [-pi, pi] so we use normalize_torsions().
            # Normalize torsion takes and return shape (batch_size, 1), not (batch_size,).
            from bgflow.nn.flow.crd_transform.ic import normalize_torsions
            angle, log_det_J = normalize_torsions(angle.unsqueeze(-1))
            angle = angle.squeeze(-1)
            cumulative_log_det_J = cumulative_log_det_J + log_det_J.squeeze(-1)

            # Group the axes atoms DOFs. From shape (batch,) to (batch, 1).
            axes_atoms_dof = [dist_axis_atom.unsqueeze(1), dist_plane_atom.unsqueeze(1), angle.unsqueeze(1)]

            # Flag axes atoms to be removed.
            kept_atoms_mask[self._reference_atoms_indices_in_cartesian[-2:]] = False

        # Remove reference atoms.
        if self.has_origin_atom or self.has_axes_atoms:
            x_cartesian = x_cartesian[:, kept_atoms_mask]

        # From (batch, n_atoms, 3) to (batch, n_atoms*3).
        x_cartesian = atom_to_flattened(x_cartesian)

        # Concatenate all DOFs.
        y = torch.cat([bonds, angles, torsions] + axes_atoms_dof + [x_cartesian], dim=-1)

        return y, cumulative_log_det_J, origin_atom_position, rotation_matrix

    def _mixed_to_cartesian(
            self,
            y: torch.Tensor,
            origin_atom_position: torch.Tensor,
            rotation_matrix: torch.Tensor,
    ) -> Tuple[torch.Tensor]:
        """Convert from mixed to Cartesian."""
        cumulative_log_det_J = 0

        # Separate the IC and Cartesian DOFs.
        bonds = y[:, :self.n_ic_atoms]
        angles = y[:, self.n_ic_atoms:2*self.n_ic_atoms]
        torsions = y[:, 2*self.n_ic_atoms:3*self.n_ic_atoms]
        y_cartesian = y[:, 3*self.n_ic_atoms:]

        # Separate the axes atoms DOFs.
        if self.has_axes_atoms:
            dist_axis_atom, dist_plane_atom, angle, y_cartesian = y_cartesian.split(
                [1, 1, 1, y_cartesian.shape[-1]-3], dim=-1)

            # Unnormalize the angle.
            from bgflow.nn.flow.crd_transform.ic import unnormalize_torsions
            angle, log_det_J = unnormalize_torsions(angle)
            cumulative_log_det_J = cumulative_log_det_J + log_det_J

            # From shape (batch_size, 1) to (batch_size,).
            dist_axis_atom = dist_axis_atom.squeeze(-1)
            dist_plane_atom = dist_plane_atom.squeeze(-1)
            angle = angle.squeeze(-1)

        # From (batch, n_atoms*3) to (batch, n_atoms, 3).
        y_cartesian = flattened_to_atom(y_cartesian)

        # Find the positions of the reference atoms.
        batch_size, n_cartesian_atoms, _ = y_cartesian.shape
        reference_atom_positions = []
        if self.has_origin_atom:
            # We insert the origin since we'll translate also this later.
            reference_atom_positions.append(torch.zeros_like(origin_atom_position))
        if self.has_axes_atoms:
            # The axis atom lies on the x-axis.
            axis_atom_pos = torch.zeros(batch_size, 3).to(y_cartesian)
            axis_atom_pos[:, 0] = dist_axis_atom

            # The plane atom lies on the xy-plane.
            x, y, log_det_J = polar_to_cartesian(
                dist_plane_atom,
                angle,
                return_log_det_J=True,
            )
            plane_atom_pos = torch.zeros_like(axis_atom_pos)
            plane_atom_pos[:, 0] = x
            plane_atom_pos[:, 1] = y

            # Update.
            reference_atom_positions.extend([axis_atom_pos, plane_atom_pos])
            cumulative_log_det_J = cumulative_log_det_J + log_det_J

        # Insert back into y_cartesian the reference atoms.
        if len(reference_atom_positions) > 0:
            y_cartesian_tmp = torch.empty(batch_size, n_cartesian_atoms+len(reference_atom_positions), 3).to(y_cartesian)

            # Start by setting the reference atom positions.
            for idx, ref_atom_idx in enumerate(self._reference_atoms_indices_in_cartesian):
                y_cartesian_tmp[:, ref_atom_idx] = reference_atom_positions[idx]

            # Now set the other Cartesian coordinates.
            mask = torch.full(y_cartesian_tmp.shape[1:2], fill_value=True).to(
                self._reference_atoms_indices_in_cartesian.device)
            mask[self._reference_atoms_indices_in_cartesian] = False
            y_cartesian_tmp[:, mask] = y_cartesian

            y_cartesian = y_cartesian_tmp

        # Rotate and translate the Cartesian back to the global frame of reference.
        if self.has_axes_atoms:
            y_cartesian = batchwise_rotate(y_cartesian, rotation_matrix, inverse=True)
        if self.has_origin_atom:
            y_cartesian = y_cartesian + origin_atom_position.unsqueeze(1)

        # Convert internal coords back to Cartesian.
        x, log_det_J = self._rel_ic(bonds, angles, torsions, y_cartesian, inverse=True)
        cumulative_log_det_J = cumulative_log_det_J + log_det_J.squeeze(-1)

        return x, cumulative_log_det_J
