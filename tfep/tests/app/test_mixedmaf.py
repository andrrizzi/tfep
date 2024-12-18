#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test classes and functions in the ``tfep.app.mixedmaf`` module.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import os
import random

import MDAnalysis
import numpy as np
import pint
import pytest
import torch

from tfep.app.mixedmaf import MixedMAFMap, _CartesianToMixedFlow
import tfep.nn.transformers
from tfep.utils.geometry import (
    batchwise_rotate,
    get_axis_from_name,
    reference_frame_rotation_matrix,
)

from .. import DATA_DIR_PATH, MockPotential, benzoic_acid_universe, water_universe
from . import check_atom_groups


# bgflow is an optional dependency of the package.
bgflow = pytest.importorskip('bgflow')


# =============================================================================
# GLOBAL VARIABLES
# =============================================================================

CHLOROMETHANE_PDB_FILE_PATH = os.path.join(DATA_DIR_PATH, 'chloro-fluoromethane.pdb')

UNITS = pint.UnitRegistry()


# =============================================================================
# TEST MODULE CONFIGURATION
# =============================================================================

_old_default_dtype = None

def setup_module(module):
    global _old_default_dtype
    _old_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.double)


def teardown_module(module):
    torch.set_default_dtype(_old_default_dtype)


# =============================================================================
# TEST UTILITIES
# =============================================================================

def create_z_matrix(
        frags_n_atoms,
        reference_atom_indices,
        remove_ref_rototranslation,
        consecutive_frag_atoms=True,
        conditioning_atom_indices=None,
):
    """Generate test cases to test _CartesianToMixedFlow.

    Parameters
    ----------
    frags_n_atoms : list[int]
        frags_n_atoms[i] is the number of atoms in the i-th generated fragment.
        Fragments up to 5 atoms are supported.
    reference_atom_indices : list[int]
    remove_ref_rototranslation : list[bool]
    consecutive_frag_atoms : bool
        If True, the atom indices belonging to a fragment have consecutive
        indices. Otherwise, their order is scrambled.
    conditioning_atom_indices : list[int]
        The indices of the conditioning atoms.

    Returns
    -------
    cartesian_atom_indices : torch.Tensor
        Cartesian atom indices to initialize _CartesianToMixedFlow.
    z_matrix : torch.Tensor
        Z-matrix to initialize _CartesianToMixedFlow.
    reference_atom_indices : list[int] or torch.Tensor
        Reference atom indices to initialize _CartesianToMixedFlow. This is
        different from the input argument only if consecutive_frag_atoms is
        False.
    cartesian_coords : torch.Tensor
        Input cartesian coordinates.
    mixed_coords : torch.Tensor
        Expected mixed coordinates.
    conditioning_atom_indices : torch.Tensor
        The input for _CartesianToMixedFlow.get_dof_indices_by_type(). This is
        different from conditioning_atom_indices only if consecutive_frag_atoms
        is False.
    mixed_dof_indices : Dict[str, torch.Tensor]
        The expected result of _CartesianToMixedFlow.get_dof_indices_by_type().

    """
    # mixed coordinates are divided in
    #   bonds, angles, torsions, d01, d02, a102, cartesian
    # Angles and torsion must be normalized to [0, 1].
    frags = [
        # 1 atom.
        {
            'z_matrix': [[0, -1, -1, -1]],
            'cartesian': [[0., 0., 0.]],
            'ic': {
                'bonds': [], 'angles': [], 'torsions': [],
            },
        },
        # 2 atoms.
        {
            'z_matrix': [
                [1, -1, -1, -1],
                [0, 1, -1, -1],
            ],
            'cartesian': [[0., 0., 0.], [0.8, 0.8, 0.8]],
            'ic': {
                'bonds': [], 'angles': [], 'torsions': [],
            },
        },
        # 3 atoms.
        {
            'z_matrix': [
                [0, -1, -1, -1],
                [2, 0, -1, -1],
                [1, 0, 2, -1],
            ],
            'cartesian': [[1., 0., 0.], [0., 0., -1.], [0., 0., 0.]],
            'ic': {
                'bonds': [], 'angles': [], 'torsions': [],
            },
        },
        # 4 atoms.
        {
            'z_matrix': [
                [1, -1, -1, -1],
                [0, 1, -1, -1],
                [2, 1, 0, -1],
                [3, 2, 0, 1],
            ],
            'cartesian': [[0., 0., 0.], [0., -1., 0.], [0., 0., 1.], [-1., 0., 0.]],
            'ic': {
                'bonds': [1.4142136], 'angles': [0.25], 'torsions': [0.25],
            },
        },
        # 5 atoms.
        {
            'z_matrix': [
                [3, -1, -1, -1],
                [1, 3, -1, -1],
                [0, 1, 3, -1],
                [4, 1, 0, 3],
                [2, 3, 4, 0],
            ],
            'cartesian': [[0., 0., 1.], [1., 1., 1.], [1., 0., 0.], [0., 0., 0.], [0., 1., 0.]],
            'ic': {
                'bonds': [1.4142136, 1.], 'angles': [0.3333333, 0.5], 'torsions': [0.59796, 0.25],
            },
        },
    ]

    # Used to find conditioning atoms.
    if conditioning_atom_indices is None:
        conditioning_atom_indices_set = set()
    else:
        conditioning_atom_indices_set = set(conditioning_atom_indices)

    # Returned values.
    z_matrix = []
    cartesian_atom_indices = []
    cartesian_coords = []
    mixed_coords = {'bonds': [], 'angles': [], 'torsions': [], 'cartesians': []}
    conditioning_dof_indices = []

    # Keep track of the position of the three reference atoms.
    ref_cartesian_coords = {}

    # Build input arguments for _CartesianToMixedFlow.
    atom_counts = np.cumsum([0] + frags_n_atoms)
    for frag_idx, frag_n_atoms in enumerate(frags_n_atoms):
        # To avoid overlapping and collinear atoms, we shift the Cartesian
        # coordinates of each fragment randomly.
        frag_cart_coords = torch.tensor(frags[frag_n_atoms-1]['cartesian'])
        frag_cart_coords = frag_cart_coords + frag_idx*torch.rand(3)

        # Concatenate input Cartesian coordinates.
        cartesian_coords.append(frag_cart_coords)

        # Shift atom indices to avoid duplicates among fragments.
        frag_z_matrix = frags[frag_n_atoms-1]['z_matrix']
        frag_z_matrix = (np.array(frag_z_matrix) + atom_counts[frag_idx]).tolist()

        # Update Cartesian coords.
        for row_idx, row in enumerate(frag_z_matrix):
            atom_idx = row[0]
            frag_atom_idx = atom_idx - atom_counts[frag_idx]
            if (row_idx < 3) or atom_idx in conditioning_atom_indices_set:
                # Cartesian
                cartesian_atom_indices.append(atom_idx)
                atom_cart_coords = frag_cart_coords[frag_atom_idx]

                # Reference atoms are pushed at the end in the mixed coordinates.
                if atom_idx in reference_atom_indices:
                    ref_cartesian_coords[atom_idx] = atom_cart_coords
                else:
                    if atom_idx in conditioning_atom_indices_set:
                        first_idx = 3 * len(mixed_coords['cartesians'])
                        conditioning_dof_indices.extend(range(first_idx, first_idx+3))
                    mixed_coords['cartesians'].append(atom_cart_coords.tolist())
            else:
                # Z-matrix
                z_matrix.append(row)
                for k in ['bonds', 'angles', 'torsions']:
                    mixed_coords[k].append(frags[frag_n_atoms-1]['ic'][k][row_idx-3])

    # Convert to tensor.
    cartesian_atom_indices = torch.tensor(cartesian_atom_indices, dtype=int)
    z_matrix = torch.tensor(z_matrix, dtype=int)
    cartesian_coords = torch.cat(cartesian_coords)
    mixed_coords = {k: torch.tensor(v) for k, v in mixed_coords.items()}

    # From dict to list (in the order origin, axis, plane atoms).
    ref_cartesian_coords = [ref_cartesian_coords[i] for i in reference_atom_indices]

    # Rototranslate the cartesian into the relative reference frame.
    v01 = (ref_cartesian_coords[1] - ref_cartesian_coords[0]).unsqueeze(0)
    v02 = (ref_cartesian_coords[2] - ref_cartesian_coords[0]).unsqueeze(0)
    rotation_matrix = reference_frame_rotation_matrix(
        axis_atom_positions=v01,
        plane_atom_positions=v02,
        axis=get_axis_from_name('x'),
        plane_axis=get_axis_from_name('y'),
        project_on_positive_axis=True,
    )
    if len(mixed_coords['cartesians']) > 0:
        mixed_coords['cartesians'] = mixed_coords['cartesians'] - ref_cartesian_coords[0]
        mixed_coords['cartesians'] = batchwise_rotate(mixed_coords['cartesians'].unsqueeze(0), rotation_matrix)[0]
        mixed_coords['cartesians'] = mixed_coords['cartesians'].flatten()

    # Compute reference frame internal coordinates.
    mixed_coords['d01'] = torch.linalg.norm(v01, dim=1)
    mixed_coords['d02'] = torch.linalg.norm(v02, dim=1)

    # Angle is normalized and in polar coordinates w.r.t. relative reference frame.
    v02 = batchwise_rotate(v02.unsqueeze(0), rotation_matrix)[0]
    mixed_coords['a102'] = (torch.atan2(v02[:, 1], v02[:, 0]) + np.pi) / (2*np.pi)

    # Push at the cartesian coordinates (not removed) of the reference frame
    # atoms and add conditioning reference frame atoms DOFs.
    mixed_coords['reference'] = []
    for idx, n_zeros in enumerate([3, 2, 1]):
        if not remove_ref_rototranslation[idx]:
            mixed_coords['reference'] = mixed_coords['reference'] + [0.]*n_zeros
    mixed_coords['reference'] = torch.tensor(mixed_coords['reference'])

    # DOF indices by type.
    types_order = ['bonds', 'angles', 'torsions', 'd01', 'd02', 'a102', 'cartesians', 'reference']
    first_idx = np.cumsum([len(mixed_coords[t]) for t in types_order]).tolist()
    mixed_dof_indices = {
        'd01': first_idx[2:3], 'd02': first_idx[3:4], 'a102': first_idx[4:5],
    }
    mixed_dof_indices.update({
        'distances': list(range(first_idx[0])) + mixed_dof_indices['d01'] + mixed_dof_indices['d02'],
        'angles': list(range(first_idx[0], first_idx[1])) + mixed_dof_indices['a102'],
        'torsions': range(first_idx[1], first_idx[2]),
        'cartesians': range(first_idx[5], first_idx[6]),
        'reference': range(first_idx[6], first_idx[7]),
    })
    mixed_dof_indices = {k: torch.tensor(v, dtype=int) for k, v in mixed_dof_indices.items()}

    # The indices in conditioning_dof_indices refer now to the indices in mixed_coords['cartesians'].
    conditioning_dof_indices = [first_idx[5] + i for i in conditioning_dof_indices]

    # Add conditioning reference frame atom internal coord DOFs indices.
    if reference_atom_indices[2] in conditioning_atom_indices_set:
        conditioning_dof_indices = first_idx[3:5] + conditioning_dof_indices
    if reference_atom_indices[1] in conditioning_atom_indices_set:
        conditioning_dof_indices = first_idx[2:3] + conditioning_dof_indices
    if len(conditioning_dof_indices) > 0:
        mixed_dof_indices['conditioning'] = torch.tensor(conditioning_dof_indices)
    else:
        mixed_dof_indices['conditioning'] = None

    # Concatenate all mixed coordinates.
    mixed_coords = torch.cat([mixed_coords[t] for t in types_order])

    # Scramble the order of the input atom.
    if not consecutive_frag_atoms:
        # Generate a random permutation (without replacement).
        n_atoms = atom_counts[-1].tolist()
        permutation = torch.randperm(n_atoms)

        # Fix the indices in the Z-matrix and cartesian atoms.
        z_matrix = permutation[z_matrix]
        reference_atom_indices = permutation[reference_atom_indices]
        cartesian_atom_indices = permutation[cartesian_atom_indices]

        # Fix the order in the input Cartesian coordinates.
        cartesian_coords = cartesian_coords[permutation.sort().indices]

        # Fix the conditioning Cartesian atom indices.
        if conditioning_atom_indices is not None:
            conditioning_atom_indices = permutation[conditioning_atom_indices].sort().values

    return (
        cartesian_atom_indices,
        z_matrix,
        reference_atom_indices,
        cartesian_coords.flatten().unsqueeze(0),  # batch dimension
        mixed_coords.unsqueeze(0),
        conditioning_atom_indices,
        mixed_dof_indices,
    )


class MyMixedMAFMap(MixedMAFMap):
    """A MixedMAFMap creating a solvated system.

    The system is composed by (in this order) one benzoic acid molecule (15 atoms),
    one molecule of chloromethane (5 atoms), one fluoride ion (1 atom), and two
    water molecules (3 atoms). Their residue names (for easy selection) are BEN,
    CLMET, F, WAT1, and WAT2 respectively. The positions of the atoms may overlap
    so don't run a potential energy evaluation.

    The returned trajectory has only 1 frame and as a consequence the spline
    lower/upper limits for the Cartesian coordinates are identical so here we
    shift them by one Angstrom.

    """

    CARTESIAN_LIMIT_SHIFT = 1.0

    def __init__(self, benzoic_acid_only=False, **kwargs):
        super().__init__(
            potential_energy_func=MockPotential(),
            topology_file_path=CHLOROMETHANE_PDB_FILE_PATH,
            coordinates_file_path=CHLOROMETHANE_PDB_FILE_PATH,
            temperature=298*UNITS.kelvin,
            **kwargs
        )
        self.benzoic_acid_only = benzoic_acid_only

    def create_universe(self):
        # Load the benzoic acid
        benzoic_acid = benzoic_acid_universe()
        if self.benzoic_acid_only:
            return benzoic_acid

        # Load the chloromethane + fluoride system from disk.
        chloromethane = super().create_universe()

        # Load the water.
        water = water_universe(n_waters=2)

        # Combine the two universes.
        combined = MDAnalysis.Merge(benzoic_acid.atoms, chloromethane.atoms, water.atoms)

        # Fix residue names for easy reading.
        combined.del_TopologyAttr('resname')
        combined.add_TopologyAttr('resname', ['BEN', 'CLMET', 'F', 'WAT1', 'WAT2'])
        return combined

    def _get_transformer(self, *args, **kwargs):
        mixed_transformer = super()._get_transformer(*args, **kwargs)
        # Cartesian transformer is the only with a learnable upper bound.
        for transformer in mixed_transformer._transformers:
            if (isinstance(transformer, tfep.nn.transformers.NeuralSplineTransformer) and
                    transformer._learn_lower_bound):
                transformer.x0 -= self.CARTESIAN_LIMIT_SHIFT
                transformer.xf += self.CARTESIAN_LIMIT_SHIFT
        return mixed_transformer


# =============================================================================
# TESTS CartesianToMixedFlow
# =============================================================================

# Because of how the Z-matrix is built the reference atoms appear always as the
# first atoms of the fragments' Z-matrices. We provide test cases consistently.
@pytest.mark.parametrize('frags_n_atoms,reference_atom_indices', [
    ([4], [1, 0, 2]),
    ([5], [3, 1, 0]),
    ([4, 3], [1, 0, 2]),
    ([3, 4], [0, 2, 1]),
    ([3, 4], [0, 2, 4]),
    ([3, 3, 4], [0, 3, 7]),
    ([3, 3, 4], [3, 5, 7]),
    ([1, 3, 3, 4], [0, 1, 4]),
    ([1, 3, 3, 4], [1, 8, 0]),
    ([1, 4, 2, 3, 5], [2, 6, 5]),
    ([1, 4, 2, 3, 5], [2, 1, 6]),
    ([5, 1, 4, 3, 3, 3], [3, 1, 0]),
    ([3, 2, 4, 5, 1, 1], [15, 14, 0]),
])
@pytest.mark.parametrize('remove_ref0', [True, False])
@pytest.mark.parametrize('remove_ref1', [True, False])
@pytest.mark.parametrize('remove_ref2', [True, False])
@pytest.mark.parametrize('consecutive_frag_atoms', [True, False])
def test_cartesian_to_mixed_flow_conversion(
        frags_n_atoms,
        reference_atom_indices,
        remove_ref0,
        remove_ref1,
        remove_ref2,
        consecutive_frag_atoms,
):
    """Test _CartesianToMixedFlow.cartesian_to_mixed()."""
    remove_ref_rototranslation = [remove_ref0, remove_ref1, remove_ref2]

    # Create test case.
    cartesian_atom_indices, z_matrix, reference_atom_indices, cartesian_coords, mixed_coords, _, _ = create_z_matrix(
        frags_n_atoms,
        reference_atom_indices,
        remove_ref_rototranslation=remove_ref_rototranslation,
        consecutive_frag_atoms=consecutive_frag_atoms,
        conditioning_atom_indices=None,
    )

    # Create flow.
    flow = _CartesianToMixedFlow(
        flow=None,
        cartesian_atom_indices=cartesian_atom_indices,
        z_matrix=z_matrix,
        reference_atom_indices=reference_atom_indices,
        remove_ref_rototranslation=remove_ref_rototranslation,
    )

    # Convert coordinates.
    cartesian_coords = cartesian_coords.detach()
    cartesian_coords.requires_grad = True
    converted, log_det_J, global_origin, global_rotation_quat = flow.cartesian_to_mixed(cartesian_coords)
    assert torch.allclose(converted, mixed_coords)

    # Convert back.
    cartesian_inv, log_det_J_inv = flow.mixed_to_cartesian(converted, global_origin, global_rotation_quat)
    assert torch.allclose(cartesian_coords, cartesian_inv)
    assert torch.allclose(log_det_J+log_det_J_inv, torch.zeros_like(log_det_J))


@pytest.mark.parametrize('frags_n_atoms,reference_atom_indices', [
    ([4], [1, 0, 2]),
    ([5], [3, 1, 0]),
    ([4, 3], [1, 0, 2]),
    ([3, 4], [0, 2, 1]),
    ([3, 4], [0, 2, 4]),
    ([3, 3, 4], [0, 3, 7]),
    ([3, 3, 4], [3, 5, 7]),
    ([1, 3, 3, 4], [0, 1, 4]),
    ([1, 3, 3, 4], [1, 8, 0]),
    ([1, 4, 2, 3, 5], [2, 6, 5]),
    ([1, 4, 2, 3, 5], [2, 1, 6]),
    ([5, 1, 4, 3, 3, 3], [3, 1, 0]),
    ([3, 2, 4, 5, 1, 1], [15, 14, 0]),
])
@pytest.mark.parametrize('n_conditioning_atoms', [0, 1, 2, 3])
@pytest.mark.parametrize('remove_ref0', [True, False])
@pytest.mark.parametrize('remove_ref1', [True, False])
@pytest.mark.parametrize('remove_ref2', [True, False])
@pytest.mark.parametrize('consecutive_frag_atoms', [True, False])
def test_cartesian_to_mixed_flow_get_dof_indices_by_type(
        frags_n_atoms,
        reference_atom_indices,
        n_conditioning_atoms,
        remove_ref0,
        remove_ref1,
        remove_ref2,
        consecutive_frag_atoms,
):
    """Test _CartesianToMixedFlow.get_dof_indices_by_type()."""
    remove_ref_rototranslation = [remove_ref0, remove_ref1, remove_ref2]

    # We need to have at least 1 atom represented by the Z-matrix, but some
    # choices of conditioning_atom_indices might lead to all-cartesian coords.
    n_atoms = sum(frags_n_atoms)
    cartesian_atom_indices = [None] * n_atoms
    while len(cartesian_atom_indices) == n_atoms:
        if n_conditioning_atoms == 0:
            conditioning_atom_indices = None
        else:
            conditioning_atom_indices = sorted(random.sample(range(n_atoms), n_conditioning_atoms))

        # Create test case.
        cartesian_atom_indices, z_matrix, reference_atom_indices_in, _, _, conditioning_atom_indices, mixed_dof_indices = create_z_matrix(
            frags_n_atoms,
            reference_atom_indices,
            remove_ref_rototranslation=remove_ref_rototranslation,
            consecutive_frag_atoms=consecutive_frag_atoms,
            conditioning_atom_indices=conditioning_atom_indices,
        )

    # Create flow.
    flow = _CartesianToMixedFlow(
        flow=None,
        cartesian_atom_indices=cartesian_atom_indices,
        z_matrix=z_matrix,
        reference_atom_indices=reference_atom_indices_in,
        remove_ref_rototranslation=remove_ref_rototranslation,
    )

    # get_dof_indices_by_type requires a tensor.
    if conditioning_atom_indices is not None:
        conditioning_atom_indices = torch.tensor(conditioning_atom_indices)

    # Check dof indices.
    dof_indices = flow.get_dof_indices_by_type(conditioning_atom_indices)
    assert len(dof_indices) == len(mixed_dof_indices)
    for k, v in dof_indices.items():
        if v is None:
            assert mixed_dof_indices[k] is None
        else:
            assert torch.allclose(v, mixed_dof_indices[k])


# =============================================================================
# TESTS MixedMAFMap
# =============================================================================

@pytest.mark.parametrize('mapped_atoms,conditioning_atoms,origin_atom,axes_atoms,expected_z_matrix,expected_ref_atom_indices', [
    # Chloromethane is mapped. Everything else is fixed. No origin/axes.
    ('resname CLMET',
     None,
     None, None,
     [[3, 0, 2, 1], [4, 0, 3, 2]],
     [0, 1, 2],
     ),
    # Map separable parts of benzoic acid. Everything else is fixed. No origin/axes.
    ('resname BEN and (name H3 or name C3 or name C4 or name H4 or name HO2 or name O2 or name C or name O1)',
     None,
     None, None,
     [[5, 2, 0, 1], [7, 4, 3, 6]],
     [0, 1, 2],
     ),
    # Map multiple molecules: chloromethane, water, and F. Everything else is fixed. No origin/axes.
    ('resname CLMET or resname WAT1 or resname F',
     None,
     None, None,
     [[3, 0, 2, 1], [4, 0, 3, 2]],
     [0, 1, 2],
     ),
    # Map multiple molecules: benzoic acid and chloromethane. Everything else is fixed. No origin/axes.
    ('resname BEN or resname CLMET',
     None,
     None, None,
     [
         [8, 3, 4, 0],
         [1, 0, 3, 8],
         [2, 0, 1, 3],
         [5, 4, 3, 8],
         [10, 4, 5, 3],
         [7, 8, 3, 5],
         [14, 8, 7, 3],
         [9, 2, 0, 1],
         [6, 7, 5, 8],
         [11, 5, 6, 4],
         [13, 7, 6, 8],
         [12, 6, 7, 5],
         [18, 15, 17, 16],
         [19, 15, 18, 17],
     ],
     [3, 0, 4],
     ),
    # Condition benzoic acid's H atoms on its other atoms.
    ('resname BEN and element H',
     'resname BEN and not element H',
     None, None,
      [
         [10, 4, 5, 3],
         [14, 8, 7, 3],
         [9, 2, 0, 1],
         [11, 5, 6, 4],
         [13, 7, 6, 8],
         [12, 6, 7, 5],
     ],
     [3, 0, 4],
     ),
    # Set the origin atom.
    ('resname BEN and element C',
     None,
     'resname BEN and name C4', None,
      [[2, 3, 4, 5], [6, 5, 4, 2], [1, 6, 2, 5], [0, 1, 6, 2]],
     [4, 3, 5],
     ),
    # Conditioning origin atom.
    ('resname BEN and element C and not name C4',
     'resname BEN and name C4',
     'resname BEN and name C4', None,
      [[2, 3, 4, 5], [6, 5, 4, 2], [1, 6, 2, 5], [0, 1, 6, 2]],
     [4, 3, 5],
     ),
    # Set origin and axes atom.
    ('resname BEN and element C',
     None,
     'resname BEN and name C2', 'resname BEN and (name C3 or name C5)',
     [[1, 2, 3, 5], [0, 1, 2, 5], [6, 1, 5, 0], [4, 5, 3, 6]],
     [2, 3, 5],
     ),
    # Conditioning origin with mapped axes atom.
    ('resname BEN and element C and not name C6',
     'resname BEN and name C6',
     'resname BEN and name C6', 'resname BEN and (name C1 or name C5)',
      [[0, 1, 6, 5], [2, 1, 0, 6], [4, 5, 6, 2], [3, 4, 2, 5]],
     [6, 1, 5],
     ),
    # Conditioning axes atoms with mapped origin.
    ('resname BEN and element C and not (name C3 or name C4)',
     'resname BEN and (name C3 or name C4)',
     'resname BEN and name C2', 'resname BEN and (name C3 or name C4)',
     [[1, 2, 3, 4], [0, 1, 2, 3], [6, 1, 0, 2], [5, 6, 4, 1]],
     [2, 3, 4],
     ),
    # All reference atoms are conditioning.
    ('resname BEN and element C and not (name C1 or name C or name C5)',
     'resname BEN and (name C1 or name C or name C5)',
     'resname BEN and name C1', 'resname BEN and (name C or name C5)',
      [[2, 1, 0, 5], [6, 5, 1, 2], [3, 2, 1, 5], [4, 3, 5, 2]],
     [1, 0, 5],
     ),
    # Origin and axes are on different mapped molecules.
    ('(resname BEN and element C) or resname CLMET',
     None,
     'resname CLMET and name C1', 'resname BEN and (name C1 or name C6)',
      [[2, 1, 0, 6], [3, 2, 1, 0], [5, 6, 1, 3], [4, 5, 3, 6], [10, 7, 9, 8], [11, 7, 10, 9]],
     [7, 1, 6],
     ),
    # Origin and bonded axes atom are different conditioning molecules.
    ('resname CLMET',
     'resname WAT1',
     'resname WAT1 and name O', '(resname CLMET and name H3) or (resname WAT1 and name H2)',
      [[2, 0, 1, 4], [3, 0, 2, 1]],
     [5, 4, 7],
     ),
])
def test_mixed_maf_flow_build_z_matrix(
        mapped_atoms,
        conditioning_atoms,
        origin_atom,
        axes_atoms,
        expected_z_matrix,
        expected_ref_atom_indices,
):
    """MixedMAFMap correctly converts the coordinates into Cartesian+internal DOFs."""
    # Initialize the map.
    tfep_map = MyMixedMAFMap(
        batch_size=2,
        mapped_atoms=mapped_atoms,
        conditioning_atoms=conditioning_atoms,
        origin_atom=origin_atom,
        axes_atoms=axes_atoms,
    )
    tfep_map.setup()

    # Get the CartesianToMixedFlow.
    if tfep_map.n_fixed_atoms > 0:
        cartesian_to_mixed_flow = tfep_map._flow.flow
    else:
        cartesian_to_mixed_flow = tfep_map._flow

    # Shortcuts.
    ic_atom_indices = cartesian_to_mixed_flow.z_matrix[:, 0]
    cartesian_atom_indices = cartesian_to_mixed_flow.cartesian_atom_indices
    n_ic_atoms = len(ic_atom_indices)
    n_cartesian_atoms = len(cartesian_atom_indices)

    # Check that we determine the correct Z-matrix.
    assert torch.allclose(cartesian_to_mixed_flow.z_matrix, torch.tensor(expected_z_matrix))
    assert torch.allclose(cartesian_to_mixed_flow.rel_ic.fixed_atoms[-3:], torch.tensor(expected_ref_atom_indices))

    # The Z-matrix and fixed atoms cover all the mapped + conditioning atoms.
    n_expected_atoms = tfep_map.n_mapped_atoms + tfep_map.n_conditioning_atoms
    assert n_ic_atoms + n_cartesian_atoms == n_expected_atoms
    assert len(set(ic_atom_indices) | set(cartesian_atom_indices)) == n_expected_atoms


@pytest.mark.parametrize('fix_origin,fix_orientation', [
    (False, False),
    (True, False),
    (True, True),
])
@pytest.mark.parametrize('mapped_atoms,conditioning_atoms,expected_mapped,expected_conditioning,expected_fixed,expected_mapped_fixed_removed,expected_conditioning_fixed_removed', [
    # If neither mapped nor conditioning are given, all atoms are mapped.
    (None, None, list(range(15)), None, None, list(range(15)), None),
    # If only mapped is given, the non-mapped are fixed.
    ('index 0:5', None, list(range(6)), None, list(range(6, 15)), list(range(6)), None),
    ([0, 3, 4, 5, 7], None, [0, 3, 4, 5, 7], None, [1, 2, 6]+list(range(8, 15)), [0, 1, 2, 3, 4], None),
    ('index 1:13', None, list(range(1, 14)), None, [0, 14], list(range(13)), None),
    (np.array([3, 4, 5, 8, 12]), None, [ 3, 4, 5, 8, 12], None, [0, 1, 2, 6, 7, 9, 10, 11, 13, 14], [0, 1, 2, 3, 4], None),
    # If only conditioning is given, the non-conditioning are mapped.
    (None, 'index 3:4', [0, 1, 2]+list(range(5, 15)), [3, 4], None, [0, 1, 2]+list(range(5, 15)), [3, 4]),
    (None, torch.tensor([0, 4, 5]), [1, 2, 3]+list(range(6, 15)), [0, 4, 5], None, [1, 2, 3]+list(range(6, 15)), [0, 4, 5]),
    # If both are given, everything else is fixed.
    ('index 3:6', [1], [3, 4, 5, 6], [1], [0, 2]+list(range(7, 15)), [1, 2, 3, 4], [0]),
    (torch.tensor([1, 3, 4, 5, 6]), [2]+list(range(7, 14)), [1, 3, 4, 5, 6], [2]+list(range(7, 14)), [0, 14], [0, 2, 3, 4, 5], [1]+list(range(6, 13))),
    ([0, 3, 4, 8, 14], np.array([1, 5]), [0, 3, 4, 8, 14], [1, 5], [2, 6, 7, 9, 10, 11, 12, 13], [0, 2, 3, 5, 6], [1, 4]),
])
def test_mixed_maf_flow_atom_groups(
        fix_origin,
        fix_orientation,
        mapped_atoms,
        conditioning_atoms,
        expected_mapped,
        expected_conditioning,
        expected_fixed,
        expected_mapped_fixed_removed,
        expected_conditioning_fixed_removed,
):
    """Mapped, conditioning, fixed, and reference frame atoms are selected and handled correctly."""
    check_atom_groups(
        tfep_map_cls=MyMixedMAFMap,
        # Currently, the test picks only conditioning origin atoms and assumes
        # that the translational/rotationa DOFs are kept fixed
        fix_origin=fix_origin,
        fix_orientation=fix_orientation,
        mapped_atoms=mapped_atoms,
        conditioning_atoms=conditioning_atoms,
        expected_mapped=expected_mapped,
        expected_conditioning=expected_conditioning,
        expected_fixed=expected_fixed,
        expected_mapped_fixed_removed=expected_mapped_fixed_removed,
        expected_conditioning_fixed_removed=expected_conditioning_fixed_removed,

        # MixedMAFMap kwargs.
        batch_size=1,
        benzoic_acid_only=True,
        remove_translation=True,
        remove_rotation=True,
    )


@pytest.mark.parametrize('origin_atom,axes_atoms', [
    (False, None),
    (True, None),
    (True, 'resname BEN and (name C3 or name C5)'),
    (True, 'resname BEN and (name C3 or name C6)'),
    (True, 'resname BEN and (name C2 or name C3)'),
    (True, 'resname BEN and (name C2 or name C1)'),
])
@pytest.mark.parametrize('conditioning_atoms', [False, True])
@pytest.mark.parametrize('two_fragments', [False, True])
@pytest.mark.parametrize('remove_translation', [False, True])
@pytest.mark.parametrize('remove_rotation', [False, True])
def test_mixed_maf_flow_get_transformer(
        origin_atom,
        axes_atoms,
        conditioning_atoms,
        two_fragments,
        remove_translation,
        remove_rotation,
):
    """The limits of the neural spline transformer are constructed correctly."""
    # MockPotential default positions unit is angstrom.
    distance_lower_limit_displacement = 0.2

    mapped_atoms = 'resname BEN and element C'
    if two_fragments:
         mapped_atoms = f'(({mapped_atoms}) or resname WAT1)'
    origin_atom = 'resname BEN and name C4' if origin_atom else None
    conditioning_atoms = 'resname BEN and element O' if conditioning_atoms else None

    # Create map.
    tfep_map = MyMixedMAFMap(
        batch_size=2,
        mapped_atoms=mapped_atoms,
        conditioning_atoms=conditioning_atoms,
        origin_atom=origin_atom,
        axes_atoms=axes_atoms,
        remove_translation=remove_translation,
        remove_rotation=remove_rotation,
        distance_lower_limit_displacement=distance_lower_limit_displacement * UNITS.angstrom,
    )
    tfep_map.setup()

    # Get transformer.
    mixed_transformer = tfep_map._flow.flow.flow[0]._transformer
    transformers = mixed_transformer._transformers
    distance_spl, angle_spl, torsion_spl = transformers[:3]

    # There are always 4 bonds/angles/torsions.
    n_ic = 4

    # Check lengths.
    assert len(distance_spl.x0) == n_ic + 2
    assert len(angle_spl.x0) == n_ic + 1
    assert len(torsion_spl.x0) == n_ic

    # Check bond limits. Benzoic acid has no bonds less/greater than 1.35/1.5 A.
    assert torch.all(distance_spl.x0[:n_ic] > torch.tensor(1.35 - distance_lower_limit_displacement))
    assert torch.all(distance_spl.xf[:n_ic] < torch.tensor(1.5))

    # Check angles and torsions limits.
    assert torch.allclose(angle_spl.x0, torch.zeros_like(angle_spl.x0))
    assert torch.allclose(angle_spl.xf, torch.ones_like(angle_spl.x0))
    assert torch.allclose(torsion_spl.x0, torch.zeros_like(torsion_spl.x0))
    assert torch.allclose(torsion_spl.xf, torch.ones_like(torsion_spl.x0))

    # Torsions must be flagged as circular (but not bond angles which are in [0, pi]).
    expected_circular_indices = list(range(2*n_ic, 3*n_ic))
    assert torch.all(mixed_transformer._indices2 == torch.tensor(expected_circular_indices))

    # Check if there are the axes atoms DOFs after the internal coordinates.
    for axes_atom_idx in range(2):
        # The limits depends on the value during the simulation.
        positions = tfep_map.dataset.universe.trajectory[0].positions
        axes_pos = positions[tfep_map._axes_atoms_indices[axes_atom_idx].tolist()]

        # Find distance.
        axes_pos = axes_pos - positions[tfep_map._origin_atom_idx.tolist()]
        axes_dist = np.linalg.norm(axes_pos).tolist()

        idx = n_ic + axes_atom_idx
        assert np.isclose(distance_spl.x0[idx].tolist(), max(0.0, axes_dist-distance_lower_limit_displacement), atol=1e-5)
        assert np.isclose(distance_spl.xf[idx].tolist(), axes_dist, atol=1e-5)

    # Check if there are Cartesian.
    if two_fragments:
        cartesian_spl = transformers[3]
        assert len(cartesian_spl.x0) == 9

        # The other mapped and conditioning are treated as Cartesian. There is only
        # 1 frame in the trajectory so the min and max value for the DOF is the same.
        # MyMixedMAF shifts the lower limit since there is only 1 frame in the trajectory.
        assert torch.allclose(
            cartesian_spl.x0 + tfep_map.CARTESIAN_LIMIT_SHIFT,
            cartesian_spl.xf - tfep_map.CARTESIAN_LIMIT_SHIFT,
        )

    # Check if there are references Cartesian DOFs.
    if not (remove_translation and remove_rotation):
        reference_vpt = transformers[-1]
        assert isinstance(reference_vpt, tfep.nn.transformers.VolumePreservingShiftTransformer)
        expected_len = 6 - 3*remove_translation - 3*remove_rotation
        assert len(getattr(mixed_transformer, '_indices' + str(len(transformers)-1))) == expected_len


@pytest.mark.parametrize('conditioning_atoms', [None, 'BEN', 'WAT1'])
@pytest.mark.parametrize('remove_translation', [False, True])
@pytest.mark.parametrize('remove_rotation', [False, True])
def test_mixed_maf_flow_get_maf_degrees_in(conditioning_atoms, remove_translation, remove_rotation):
    """The input degrees for MAF returned by MixedMAFFlow._get_maf_degrees_in are correct.

    In particular, the reference DOFs (if present) get always assigned the last degree.

    """
    mapped_atoms = 'resname BEN'
    if conditioning_atoms is None:
        mapped_atoms += ' or resname WAT1'
    elif conditioning_atoms == 'BEN':
        mapped_atoms += ' and not (name C or name C1 or name C2) or resname WAT1'
        conditioning_atoms = 'resname BEN and (name C or name C1 or name C2)'
    elif conditioning_atoms == 'WAT1':
        conditioning_atoms = 'resname WAT1'

    # Create the class and get the indices by type the CartesianToMixedFlow.
    tfep_map = MyMixedMAFMap(
        mapped_atoms=mapped_atoms,
        conditioning_atoms=conditioning_atoms,
        remove_translation=remove_translation,
        remove_rotation=remove_rotation,
    )
    tfep_map.setup()
    cartesian_to_mixed_flow = tfep_map._flow.flow
    maf_dof_indices = cartesian_to_mixed_flow.get_dof_indices_by_type(
        tfep_map.get_conditioning_indices(idx_type='atom', remove_fixed=True))

    # Get the degrees in.
    degrees_in = tfep_map._get_maf_degrees_in(
        n_dofs_in=cartesian_to_mixed_flow.n_dofs_out,
        maf_dof_indices=maf_dof_indices,
    )

    for degree in degrees_in:
        # There are reference input DOFs only if they are mapped and
        # remove_translation/rotation is False.
        is_ben_conditioning = (conditioning_atoms is not None) and ('BEN' in conditioning_atoms)
        n_references = len(maf_dof_indices['reference'])
        if is_ben_conditioning or (remove_translation and remove_rotation):
            assert n_references == 0
        elif not (remove_translation or remove_rotation):
            assert n_references == 6
        else:
            assert n_references == 3

        # Orientation and origin are assigned the last degree if present.
        last_degree = torch.max(degree)
        assert torch.all(degree[maf_dof_indices['reference']] == last_degree)
        if n_references > 0:
            assert torch.count_nonzero(degree == last_degree) == n_references


def test_mixed_maf_flow_error_empty_z_matrix():
    """An error is raised if there are no internal coordinates to map."""
    tfep_map = MyMixedMAFMap(batch_size=2, mapped_atoms='resname WAT1')
    with pytest.raises(ValueError, match='no internal coordinates to map'):
        tfep_map.setup()


def test_mixed_maf_flow_error_no_element_info():
    """An error is raised if the topology has no information on atom elements."""
    # Class that removes info on atom elements.
    class _MyMixedMAFMap(MyMixedMAFMap):
        def create_universe(self):
            universe = super().create_universe()
            universe.del_TopologyAttr('element')
            return universe

    tfep_map = _MyMixedMAFMap(batch_size=2, mapped_atoms='resname WAT1')
    with pytest.raises(ValueError, match="no information on the atom elements"):
        tfep_map.setup()


def test_mixed_maf_flow_axes_without_origin():
    """An error is raised if auto_reference_frame is set and origin/axes atoms are given."""
    with pytest.raises(ValueError, match="axes_atoms cannot be passed"):
        MyMixedMAFMap(
            batch_size=2,
            mapped_atoms='resname CLMET',
            origin_atom=None,
            axes_atoms=[3, 6],
        )


def test_mixed_maf_flow_error_collinear_atoms():
    """An error is raised if the Z-matrix results in collinear atoms."""
    tfep_map = MyMixedMAFMap(
        batch_size=2,
        mapped_atoms='resname BEN and element C',
        origin_atom='resname BEN and name C1',
        axes_atoms='resname BEN and (name C or name C4)',
    )
    with pytest.raises(RuntimeError, match='collinear'):
        tfep_map.setup()


def test_mixed_maf_flow_error_fixed_ref_atoms():
    """An error is raised if the reference frame atoms are fixed."""
    tfep_map = MyMixedMAFMap(
        mapped_atoms='resname BEN and element C',
        origin_atom='resname CLMET and name C1',
    )
    with pytest.raises(ValueError, match='atoms must be mapped or conditioning'):
        tfep_map.setup()
