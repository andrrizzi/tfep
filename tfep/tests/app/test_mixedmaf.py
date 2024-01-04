#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test classes and fuctions in the ``tfep.app.mixedmaf`` module.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import os

import MDAnalysis
import pint
import pytest
import torch

from tfep.utils.misc import flattened_to_atom
from tfep.app.mixedmaf import MixedMAFMap, _CartesianToMixedFlow

from .. import DATA_DIR_PATH, MockPotential, benzoic_acid_universe, water_universe


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

    This also wraps the _CartesianToInternalFlow into a TrackedFlow.

    """

    def __init__(self, **kwargs):
        super().__init__(
            potential_energy_func=MockPotential(),
            topology_file_path=CHLOROMETHANE_PDB_FILE_PATH,
            coordinates_file_path=CHLOROMETHANE_PDB_FILE_PATH,
            temperature=298*UNITS.kelvin,
            initialize_identity=False,
            **kwargs
        )
        self.cartesian_to_mixed_flow = None

    def _create_universe(self):
        # Load the chloromethane + fluoride system from disk.
        chloromethane = super()._create_universe()

        # Load the benzoic acid and the water.
        benzoic_acid = benzoic_acid_universe()
        water = water_universe(n_waters=2)

        # Combine the two universes.
        combined = MDAnalysis.Merge(benzoic_acid.atoms, chloromethane.atoms, water.atoms)

        # Fix residue names for easy reading.
        combined.del_TopologyAttr('resname')
        combined.add_TopologyAttr('resname', ['BEN', 'CLMET', 'F', 'WAT1', 'WAT2'])
        return combined

    def configure_flow(self):
        # Modify and leep track of _CartesianToMixedFlow.
        flow = super().configure_flow()
        flow.__class__ = MyCartesianToMixedFlow
        self.cartesian_to_mixed_flow = flow
        return flow


class MyCartesianToMixedFlow(_CartesianToMixedFlow):
    """Keeps track of the conversion result."""

    def _cartesian_to_mixed(self, *args, **kwargs):
        out = super()._cartesian_to_mixed(*args, **kwargs)
        self.conversion = out[0]
        return out


# =============================================================================
# TESTS
# =============================================================================

@pytest.mark.parametrize('mapped_atoms,expected_z_matrix', [
    # Chloromethane is mapped.
    ('resname CLMET', [[3, 0, 2, 1], [4, 0, 3, 2]]),
    # Map separable parts of benzoic acid.
    ('resname BEN and (name H3 or name C3 or name C4 or name H4 or name HO2 or name O2 or name C or name O1)', [[5, 2, 0, 1], [7, 4, 3, 6]]),
    # Map multiple molecules: chloromethane, water, and F.
    ('resname CLMET or resname WAT1 or resname F', [[3, 0, 2, 1], [4, 0, 3, 2]]),
    # Map multiple molecules: benzoic acid and chloromethane.
    ('resname BEN or resname CLMET', [
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
])
@pytest.mark.parametrize('conditioning_atoms', [None])#, 'resname WAT2', 'all'])
@pytest.mark.parametrize('fix_origin', [False])#, True])
@pytest.mark.parametrize('axes_atoms', [None])#, 'conditioning', 'mapped'])
def test_cartesian_and_internal_division(
        mapped_atoms,
        conditioning_atoms,
        expected_z_matrix,
        fix_origin,
        axes_atoms,
):
    """MixedMAFMap correctly converts the coordinates into Cartesian+internal DOFs."""
    # TODO: FIX ME
    origin_atom = None
    axes_atoms = None

    # Initialize the map.
    tfep_map = MyMixedMAFMap(
        batch_size=2,
        mapped_atoms=mapped_atoms,
        conditioning_atoms=conditioning_atoms,
        origin_atom=origin_atom,
        axes_atoms=axes_atoms,
    )
    tfep_map.setup()

    # Shortcuts.
    rel_ic = tfep_map.cartesian_to_mixed_flow._rel_ic
    ic_atom_indices = rel_ic.z_matrix[:, 0]
    cartesian_atom_indices = rel_ic.fixed_atoms
    n_ic_atoms = len(ic_atom_indices)
    n_cartesian_atoms = len(cartesian_atom_indices)

    # Check that we determine the correct Z-matrix.
    assert (rel_ic.z_matrix == expected_z_matrix).all()

    # The Z-matrix and fixed atoms cover all the mapped + conditioning atoms
    # except for the reference frame atoms.
    n_reference_atoms = 0
    if origin_atom is not None:
        n_reference_atoms += 1
    if axes_atoms is not None:
        n_reference_atoms += 2
    n_expected_atoms = tfep_map.n_mapped_atoms + tfep_map.n_conditioning_atoms - n_reference_atoms
    assert n_ic_atoms + n_cartesian_atoms == n_expected_atoms
    assert len(set(ic_atom_indices) | set(cartesian_atom_indices)) == n_expected_atoms

    # Forward pass.
    x = tfep_map.dataset[0]['positions'].unsqueeze(0)
    tfep_map(x)

    # When we are not changing the frame of reference, the coordinates of the
    # Cartesian atoms should be invariant after the transformation.
    if (origin_atom is None) and (axes_atoms is None):
        # From flattened to atom shape.
        x_atom = flattened_to_atom(x)
        x_transformed_atom = flattened_to_atom(tfep_map.cartesian_to_mixed_flow.conversion)

        # The fixed atoms are not in the converted x.
        if tfep_map.n_fixed_atoms > 0:
            fixed_indices_set = set(tfep_map._fixed_atom_indices.tolist())
            nonfixed_atom_indices = torch.tensor([i for i in range(tfep_map.dataset.n_atoms)
                                                  if i not in fixed_indices_set])
            x_atom = x_atom[:, nonfixed_atom_indices]

        # The atoms represented by Cartesian coordinates are represented at the end of the feature vector.
        assert torch.all(~torch.isclose(x_atom[:, ic_atom_indices], x_transformed_atom[:, :n_ic_atoms]))
        assert torch.allclose(x_atom[:, cartesian_atom_indices], x_transformed_atom[:, n_ic_atoms:])


def test_periodic():
    """Test that periodic DOFs are treated as such."""
    # TODO
    assert False


def test_error_empty_z_matrix():
    """An error is raised if there are no internal coordinates to map."""
    # TODO
    assert False


def test_error_no_element_info():
    """An error is raised if the topology has no information on atom elements."""
    # TODO
    with pytest.raises(ValueError, match="no information on the atom elements"):
        pass
