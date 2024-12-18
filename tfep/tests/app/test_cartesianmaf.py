#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test classes in the ``tfep.app.cartesianmaf.py`` module.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import os

import numpy as np
import pint
import pytest
import torch

from tfep.app import CartesianMAFMap

from .. import MockPotential, DATA_DIR_PATH
from . import check_atom_groups


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
# TESTS
# =============================================================================

@pytest.mark.parametrize('fix_origin', [False, True])
@pytest.mark.parametrize('fix_orientation', [False, True])
@pytest.mark.parametrize('mapped_atoms,conditioning_atoms,expected_mapped,expected_conditioning,expected_fixed,expected_mapped_fixed_removed,expected_conditioning_fixed_removed', [
    # If neither mapped nor conditioning are given, all atoms are mapped.
    (None, None, list(range(6)), None, None, list(range(6)), None),
    # If only mapped is given, the non-mapped are fixed.
    ('index 0:2', None, [0, 1, 2], None, [3, 4, 5], [0, 1, 2], None),
    ([2, 3, 5], None, [2, 3, 5], None, [0, 1, 4], [0, 1, 2], None),
    ('index 1:5', None, [1, 2, 3, 4, 5], None, [0], [0, 1, 2, 3, 4], None),
    (np.array([0, 2, 3, 4, 5]), None, [0, 2, 3, 4, 5], None, [1], [0, 1, 2, 3, 4], None),
    # If only conditioning is given, the non-conditioning are mapped.
    (None, 'index 3:4', [0, 1, 2, 5], [3, 4], None, [0, 1, 2, 5], [3, 4]),
    (None, torch.tensor([0, 4, 5]), [1, 2, 3], [0, 4, 5], None, [1, 2, 3], [0, 4, 5]),
    # If both are given, everything else is fixed.
    ('index 2:4', [1], [2, 3, 4], [1], [0, 5], [1, 2, 3], [0]),
    (torch.tensor([1, 4]), [2, 5], [1, 4], [2, 5], [0, 3], [0, 2], [1, 3]),
    ([0, 2, 4], np.array([3, 5]), [0, 2, 4], [3, 5], [1], [0, 1, 3], [2, 4]),
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
        tfep_map_cls=CartesianMAFMap,
        fix_origin=fix_origin,
        fix_orientation=fix_orientation,
        mapped_atoms=mapped_atoms,
        conditioning_atoms=conditioning_atoms,
        expected_mapped=expected_mapped,
        expected_conditioning=expected_conditioning,
        expected_fixed=expected_fixed,
        expected_mapped_fixed_removed=expected_mapped_fixed_removed,
        expected_conditioning_fixed_removed=expected_conditioning_fixed_removed,

        # CartesianMAFFlow kwargs.
        potential_energy_func=MockPotential(),
        topology_file_path=CHLOROMETHANE_PDB_FILE_PATH,
        coordinates_file_path=CHLOROMETHANE_PDB_FILE_PATH,
        temperature=298*UNITS.kelvin,
        batch_size=2,
    )


def test_error_origin_atom_not_conditioning():
    """An error is raised if the origin atom is not a conditioning atom."""
    tfep_map = CartesianMAFMap(
        mapped_atoms=range(6),
        origin_atom=1,
        potential_energy_func=MockPotential(),
        topology_file_path=CHLOROMETHANE_PDB_FILE_PATH,
        coordinates_file_path=CHLOROMETHANE_PDB_FILE_PATH,
        temperature=298*UNITS.kelvin,
        batch_size=2,
    )
    with pytest.raises(ValueError, match="is not a conditioning atom"):
        tfep_map.setup()
