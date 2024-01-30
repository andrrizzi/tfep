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

import MDAnalysis
import numpy as np
import pint
import pytest
import torch

from tfep.utils.misc import flattened_to_atom
from tfep.app.mixedmaf import MixedMAFMap, _CartesianToMixedFlow

from .. import DATA_DIR_PATH, MockPotential, benzoic_acid_universe, water_universe
from . import check_atom_groups


# bgflow is an optional dependency of the package.
try:
    import bgflow
except ImportError:
    pytest.skip('requires bgflow to be installed', allow_module_level=True)


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

class MyMixedMAFMap(MixedMAFMap):
    """A MixedMAFMap creating a solvated system.

    The system is composed by (in this order) one benzoic acid molecule (15 atoms),
    one molecule of chloromethane (5 atoms), one fluoride ion (1 atom), and two
    water molecules (3 atoms). Their residue names (for easy selection) are BEN,
    CLMET, F, WAT1, and WAT2 respectively. The positions of the atoms may overlap
    so don't run a potential energy evaluation.

    The returned trajectory has only 1 frame.

    """

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


# =============================================================================
# TESTS CartesianToMixedFlow
# =============================================================================

@pytest.mark.parametrize('origin,axes,conditioning,expected', [
    (None, None, None, None),
    (None, None, [1, 5, 6], [6, 7, 8, 15, 16, 17, 18, 19, 20]),
    # Onl origin atom.
    (2, None, [2], None),
    (5, None, [1, 5, 6], [6, 7, 8, 15, 16, 17]),
    # Both axes atoms are conditioning.
    (None, [2, 5], [2, 5, 6], [6, 7, 8, 15, 16, 17]),
    # Both axes atoms are mapped.
    (None, [2, 5], [1], [9, 10, 11]),
    # One axes atom is mapped the other is not.
    (None, [2, 5], [4, 5], [7, 8, 12, 13, 14]),
    (None, [2, 5], [1, 2, 6], [6, 9, 10, 11, 15, 16, 17]),
    # Both origin and axes atoms.
    (2, [5, 1], [1, 2, 5, 6], [6, 7, 8, 12, 13, 14]),
    (2, [5, 1], [1, 2, 4], [7, 8, 9, 10, 11]),
    (2, [5, 1], [2, 4, 5], [6, 9, 10, 11]),
    (2, [5, 1], [2, 6], [12, 13, 14]),
])
def test_cartesian_to_mixed_flow_get_maf_conditioning_dof_indices(origin, axes, conditioning, expected):
    """Test method _CartesianToMixedFlow.get_maf_conditioning_dof_indices()."""
    # Convert to tensors.
    origin, axes, conditioning, expected = [None if x is None else torch.tensor(x) for x in (origin, axes, conditioning, expected)]

    flow = _CartesianToMixedFlow(
        flow=None,
        cartesian_atom_indices=np.array([1, 2, 4, 5, 6]),
        z_matrix=np.array([[3, 1, 5, 2], [0, 2, 3, 1]]),
        origin_atom_idx=origin,
        axes_atoms_indices=axes,
    )
    conditioning_indices = flow.get_maf_conditioning_dof_indices(conditioning_atom_indices=conditioning)

    if expected is None:
        assert conditioning_indices is expected
    else:
        assert torch.all(conditioning_indices == expected)


@pytest.mark.parametrize('axes', [None, [1, 0], [3, 5]])
def test_cartesian_to_mixed_flow_get_maf_periodic_dof_indices(axes):
    """Test method _CartesianToMixedFlow.get_maf_periodic_dof_indices()."""
    expected = torch.tensor([2, 3, 4, 5])
    if axes is not None:
        axes = torch.tensor(axes)
        expected = torch.cat([expected, torch.tensor([8])])

    flow = _CartesianToMixedFlow(
        flow=None,
        cartesian_atom_indices=np.array([0, 1, 3, 4, 5]),
        z_matrix=np.array([[2, 4, 0, 1], [6, 4, 2, 1]]),
        origin_atom_idx=None,
        axes_atoms_indices=axes,
    )
    periodic_indices = flow.get_maf_periodic_dof_indices()

    assert torch.all(periodic_indices == expected)


@pytest.mark.parametrize('axes', [None, torch.tensor([1, 0])])
@pytest.mark.parametrize('return_bonds', [False, True])
@pytest.mark.parametrize('return_axes', [False, True])
def test_cartesian_to_mixed_flow_get_maf_distance_dof_indices(axes, return_bonds, return_axes):
    """Test method _CartesianToMixedFlow.get_maf_distance_dof_indices()."""
    flow = _CartesianToMixedFlow(
        flow=None,
        cartesian_atom_indices=np.array([0, 1, 3, 4, 5]),
        z_matrix=np.array([[2, 4, 0, 1], [6, 4, 2, 1]]),
        origin_atom_idx=None,
        axes_atoms_indices=axes,
    )
    distance_indices = flow.get_maf_distance_dof_indices(return_bonds=return_bonds, return_axes=return_axes)

    expected = []
    if return_bonds:
        expected = [0, 1]
    if return_axes and axes is not None:
        expected.extend([6, 7])
    assert torch.all(distance_indices == torch.tensor(expected))


@pytest.mark.parametrize('origin', [None, 4, 7])
@pytest.mark.parametrize('axes', [None, (2, 1), (9, 2), (0, 3)])
def test_cartesian_to_mixed_flow_conversion(origin, axes):
    """Test methods _CartesianToMixedFlow._mixed_to_cartesian() and ._cartesian_to_mixed()."""
    batch_size, n_atoms = 2, 10
    flow = _CartesianToMixedFlow(
        flow=None,
        cartesian_atom_indices=np.array([0, 1, 2, 3, 4, 5, 7, 9]),
        z_matrix=np.array([[6, 2, 1, 4], [8, 1, 4, 2]]),
        origin_atom_idx=None if origin is None else torch.tensor(origin),
        axes_atoms_indices=None if axes is None else torch.tensor(axes),
    )

    # Forward pass.
    x = torch.rand(batch_size, n_atoms*3)
    y, log_det_J, origin_atom_position, rotation_matrix = flow._cartesian_to_mixed(x)

    # Bonds and distances are greater than 0.
    distance_dofs = flow.get_maf_distance_dof_indices()
    assert torch.all(y[:, distance_dofs] >= 0)

    # Periodic indices are normalized angles within 0 and 1.
    periodic_dofs = flow.get_maf_periodic_dof_indices()
    assert torch.all(y[:, periodic_dofs] >= 0)
    assert torch.all(y[:, periodic_dofs] <= 1)

    # With origin and/or axes atoms the total number of DOFs is reduced.
    expected_n_dofs = x.shape[1]
    if flow.has_origin_atom:
        expected_n_dofs -= 3
    if flow.has_axes_atoms:
        expected_n_dofs -= 3
    assert y.shape == (batch_size, expected_n_dofs)

    # Without origin and axes atoms, all Cartesian coordinates are invariant.
    if not (flow.has_origin_atom or flow.has_axes_atoms):
        x_cartesian = flattened_to_atom(x)[:, flow.cartesian_atom_indices]
        y_cartesian = flattened_to_atom(y)[:, flow.n_ic_atoms:]
        assert torch.allclose(x_cartesian, y_cartesian)

    # Test inversion.
    x_inv, log_det_J_inv = flow._mixed_to_cartesian(y, origin_atom_position, rotation_matrix)
    assert torch.allclose(x, x_inv)

    # Total Jacobian determinant of forward+inverse without a flow should be one.
    assert torch.allclose(log_det_J + log_det_J_inv, torch.zeros_like(log_det_J))


# =============================================================================
# TESTS MixedMAFMap
# =============================================================================

@pytest.mark.parametrize('mapped_atoms,conditioning_atoms,origin_atom,axes_atoms,expected_are_bonded,expected_z_matrix', [
    # Chloromethane is mapped. Everything else is fixed. No origin/axes.
    ('resname CLMET',
     None,
     None, None, None,
     [[3, 0, 2, 1], [4, 0, 3, 2]]
     ),
    # Map separable parts of benzoic acid. Everything else is fixed. No origin/axes.
    ('resname BEN and (name H3 or name C3 or name C4 or name H4 or name HO2 or name O2 or name C or name O1)',
     None,
     None, None, None,
     [[5, 2, 0, 1], [7, 4, 3, 6]]
     ),
    # Map multiple molecules: chloromethane, water, and F. Everything else is fixed. No origin/axes.
    ('resname CLMET or resname WAT1 or resname F',
     None,
     None, None, None,
     [[3, 0, 2, 1], [4, 0, 3, 2]]
     ),
    # Map multiple molecules: benzoic acid and chloromethane. Everything else is fixed. No origin/axes.
    ('resname BEN or resname CLMET',
     None,
     None, None, None,
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
     ]),
    # Condition benzoic acid's H atoms on its other atoms.
    ('resname BEN and element H',
     'resname BEN and not element H',
     None, None, None,
      [
         [10, 4, 5, 3],
         [14, 8, 7, 3],
         [9, 2, 0, 1],
         [11, 5, 6, 4],
         [13, 7, 6, 8],
         [12, 6, 7, 5],
     ]),
    # Set the origin atom.
    ('resname BEN and element C and not name C4',
     'resname BEN and name C4',
     'resname BEN and name C4', None, None,
      [[2, 3, 4, 5], [6, 5, 4, 2], [1, 6, 2, 5], [0, 1, 6, 2]]
     ),
    # Set as axes atom the C3 and C5 atoms of benzoic acid.
    ('resname BEN and element C',
     None,
     None, 'resname BEN and (name C3 or name C5)', None,
      [[4, 5, 3, 2], [1, 2, 3, 5], [0, 1, 2, 5], [6, 1, 5, 0]]
     ),
    # Axes atom bonded.
    ('resname BEN and element C and not name C6',
     'resname BEN and name C6',
     'resname BEN and name C6', 'resname BEN and (name C1 or name C5)', [True, True],
      [[0, 1, 6, 5], [2, 1, 0, 6], [4, 5, 6, 2], [3, 4, 2, 5]],
     ),
    # Origin and axes are distant on the same molecule.
    ('resname BEN and element C and not name C1',
     'resname BEN and name C1',
     'resname BEN and name C1', 'resname BEN and (name C or name C4)', [True, False],
      [[2, 1, 0, 4], [6, 1, 2, 0], [3, 2, 4, 1], [5, 6, 4, 1]]
     ),
    # Origin and axes are on different mapped molecules.
    ('(resname BEN and element C) or (resname CLMET and not name C1)',
     'resname CLMET and name C1',
     'resname CLMET and name C1', 'resname BEN and (name C1 or name C6)', [False, False],
      [[2, 1, 0, 6], [3, 2, 1, 0], [5, 6, 1, 3], [4, 5, 3, 6], [10, 7, 9, 8], [11, 7, 10, 9]]
     ),
    # Origin and bonded axes atom are different conditioning molecules.
    ('resname CLMET',
     'resname WAT1',
     'resname WAT1 and name O', '(resname CLMET and name H3) or (resname WAT1 and name H2)', [False, True],
      [[2, 0, 1, 4], [3, 0, 2, 1]]
     ),
])
def test_mixed_maf_flow_build_z_matrix(
        mapped_atoms,
        conditioning_atoms,
        origin_atom,
        axes_atoms,
        expected_are_bonded,
        expected_z_matrix,
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
    assert (cartesian_to_mixed_flow.z_matrix == expected_z_matrix).all()

    # Test are_axes_atoms_bonded.
    _, _, are_axes_atoms_bonded = tfep_map._build_z_matrix()
    assert are_axes_atoms_bonded == expected_are_bonded

    # The Z-matrix and fixed atoms cover all the mapped + conditioning atoms.
    n_expected_atoms = tfep_map.n_mapped_atoms + tfep_map.n_conditioning_atoms
    assert n_ic_atoms + n_cartesian_atoms == n_expected_atoms
    assert len(set(ic_atom_indices) | set(cartesian_atom_indices)) == n_expected_atoms


@pytest.mark.parametrize('fix_origin', [False, True])
@pytest.mark.parametrize('fix_orientation', [False, True])
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
def test_atom_groups(
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
    )


def test_mixed_maf_flow_auto_reference_atoms():
    """Origin and axes atoms are automatically selected if requested."""
    tfep_map = MyMixedMAFMap(
        batch_size=2,
        mapped_atoms='resname CLMET',
        conditioning_atoms='resname F',
        auto_reference_frame=True,
    )
    tfep_map.setup()
    assert tfep_map.get_reference_atoms_indices(remove_fixed=True).tolist() == [0, 1, 2]

    # The origin atom has been converted to a conditioning DOF.
    assert tfep_map._origin_atom_idx in tfep_map._conditioning_atom_indices
    assert tfep_map._origin_atom_idx not in tfep_map._mapped_atom_indices


@pytest.mark.parametrize('origin_atom', [False, True])
@pytest.mark.parametrize('axes_atoms,are_bonded', [
    (None, None),
    ('resname BEN and (name C3 or name C5)', [True, True]),
    ('resname BEN and (name C3 or name C6)', [True, False]),
    ('resname BEN and (name C2 or name C3)', [False, True]),
    ('resname BEN and (name C2 or name C1)', [False, False]),
])
def test_mixed_maf_flow_get_transformer(origin_atom, axes_atoms, are_bonded):
    """The limits of the neural spline transformer are constructed correctly."""
    # MockPotential default positions unit is angstrom.
    bond_limits = np.array([0.2, 5.4])  # in Angstrom
    max_cartesian_displacement = 2.  # in Angstrom

    mapped_atoms = 'resname BEN and element C'
    conditioning_atoms = 'resname BEN and element O'
    if origin_atom:
        mapped_atoms += ' and not name C4'
        origin_atom = 'resname BEN and name C4'
        conditioning_atoms += ' or ' + origin_atom
    else:
        origin_atom = None
        are_bonded = [False, False]

    tfep_map = MyMixedMAFMap(
        batch_size=2,
        mapped_atoms=mapped_atoms,
        conditioning_atoms=conditioning_atoms,
        origin_atom=origin_atom,
        axes_atoms=axes_atoms,
        bond_limits=bond_limits * UNITS.angstrom,
        max_cartesian_displacement=max_cartesian_displacement / 10. * UNITS.nanometer,
    )
    tfep_map.setup()

    # Get transformer.
    transformer = tfep_map._flow.flow.flow[0]._transformer

    # There are always 4 bonds/angles/torsions.
    n_ic = 4

    # Check bond limits.
    assert torch.allclose(transformer.x0[:n_ic], torch.full((n_ic,), bond_limits[0]))
    assert torch.allclose(transformer.xf[:n_ic], torch.full((n_ic,), bond_limits[1]))

    # Check angles and torsions limits.
    assert torch.allclose(transformer.x0[n_ic:3*n_ic], torch.zeros(2*n_ic))
    assert torch.allclose(transformer.xf[n_ic:3*n_ic], torch.ones(2*n_ic))

    # Angles and torsions must be flagged as circular. Wait to test them all
    # in case there are axes atom.
    expected_circular_indices = list(range(n_ic, 3*n_ic))

    # Check if there are the axes atoms DOFs after the internal coordinates.
    if axes_atoms is not None:
        # The limits are different if the axes atoms are bonded to the origin or not.
        for i, is_bonded in enumerate(are_bonded):
            idx = 3 * n_ic + i
            if is_bonded:
                assert np.isclose(transformer.x0[idx].tolist(), bond_limits[0])
                assert np.isclose(transformer.xf[idx].tolist(), bond_limits[1])
            else:
                # The limits depends on the value during the simulation.
                positions = tfep_map.dataset.universe.trajectory[0].positions
                axes_pos = positions[tfep_map._axes_atoms_indices[i].tolist()]

                # Find distance.
                if origin_atom is not None:
                    axes_pos = axes_pos - positions[tfep_map._origin_atom_idx.tolist()]
                axes_dist = np.linalg.norm(axes_pos).tolist()

                assert np.isclose(transformer.x0[idx].tolist(), max(0.0, axes_dist-max_cartesian_displacement), atol=1e-5)
                assert np.isclose(transformer.xf[idx].tolist(), axes_dist+max_cartesian_displacement, atol=1e-5)

        # The third DOF is always an angle.
        idx = 3 * n_ic + 2
        assert np.isclose(transformer.x0[idx].tolist(), 0.0)
        assert np.isclose(transformer.xf[idx].tolist(), 1.0)
        expected_circular_indices.append(idx)

        # First index treated as a Cartesian coordinate.
        start_cartesian_idx = idx + 1
    else:
        start_cartesian_idx = 3 * n_ic + 1

    # Neural splines don't care about conditioning atoms.
    assert len(transformer.x0) == tfep_map.n_mapped_dofs
    assert len(transformer.xf) == tfep_map.n_mapped_dofs

    # The other mapped and conditioning are treated as Cartesian. There is only
    # 1 frame in the trajectory so the min and max value for the DOF is the same.
    expected_diff = 2 * max_cartesian_displacement
    diff = transformer.xf[start_cartesian_idx:] - transformer.x0[start_cartesian_idx:]
    assert torch.all(torch.isclose(diff, torch.tensor(expected_diff)))

    # Test DOFs flagged as circular.
    assert torch.all(transformer._circular == torch.tensor(expected_circular_indices))


def test_error_empty_z_matrix():
    """An error is raised if there are no internal coordinates to map."""
    tfep_map = MyMixedMAFMap(batch_size=2, mapped_atoms='resname WAT1')
    with pytest.raises(ValueError, match='no internal coordinates to map'):
        tfep_map.setup()


def test_error_no_element_info():
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


def test_error_auto_reference_with_origin_or_axes():
    """An error is raised if auto_reference_frame is set and origin/axes atoms are given."""
    with pytest.raises(ValueError, match="origin_atom and axes_atoms must be None"):
        MyMixedMAFMap(
            batch_size=2,
            mapped_atoms='resname CLMET',
            origin_atom=1,
            axes_atoms=[3, 6],
            auto_reference_frame=True
        )
