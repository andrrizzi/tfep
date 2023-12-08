#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test objects and function in the module ``tfep.app.cartesianmaf``.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import os

import pint
import pytest
import torch

from tfep.app.cartesianmaf import CartesianMAFMap
from tfep.utils.math import batchwise_dot
from tfep.utils.misc import flattened_to_atom

from .test_base import MockPotential

# =============================================================================
# GLOBAL VARIABLES
# =============================================================================

SCRIPT_DIR_PATH = os.path.dirname(__file__)
CHLOROMETHANE_PDB_FILE_PATH = os.path.join(SCRIPT_DIR_PATH, '..', 'data', 'chloro-fluoromethane.pdb')

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

@pytest.mark.parametrize('reference_atoms,mapped_atoms,conditioning_atoms', [
    ([1, 3, 5], None, None),
    ([5, 1, 2], None, None),
    ([0, 1, 2], [0, 1, 2], None),
    ([5, 2, 1], [0, 1, 2, 5], None),
    ([1, 3, 5], None, [0, 2]),
    ([5, 1, 3], None, [0, 2]),
    ([0, 1, 2], [0, 1, 2], [3, 5]),
    ([5, 2, 1], [0, 1, 2, 5], [3]),
    ([5, 2, 0], [0, 2, 5], [4]),
])
def test_rototranslational_equivariance(
        reference_atoms,
        mapped_atoms,
        conditioning_atoms,
):
    """Test that global rototranslational degrees of freedom are not modified."""
    tfep_map = CartesianMAFMap(
        potential_energy_func=MockPotential(),
        topology_file_path=CHLOROMETHANE_PDB_FILE_PATH,
        coordinates_file_path=CHLOROMETHANE_PDB_FILE_PATH,
        temperature=300*UNITS.kelvin,
        batch_size=2,
        reference_atoms=reference_atoms,
        mapped_atoms=mapped_atoms,
        conditioning_atoms=conditioning_atoms,
        n_maf_layers=2,
        initialize_identity=False,
    )
    tfep_map.setup()

    # Generate random positions.
    n_features = tfep_map.dataset.n_atoms * 3
    x = torch.randn(tfep_map.hparams.batch_size, n_features)

    # Run map.
    y, log_det_J = tfep_map(x)

    x = flattened_to_atom(x)
    y = flattened_to_atom(y)

    # Make sure the transformation did something.
    assert not torch.allclose(x, y)

    # The center atom should be left untouched.
    assert torch.allclose(x[:, reference_atoms[0]], y[:, reference_atoms[0]])

    # The direction center-axis should be the same.
    dir_01_x = torch.nn.functional.normalize(x[:, reference_atoms[1]] - x[:, reference_atoms[0]])
    dir_01_y = torch.nn.functional.normalize(y[:, reference_atoms[1]] - y[:, reference_atoms[0]])
    assert torch.allclose(dir_01_x, dir_01_y)

    # The mapped plane atom should be orthogonal to the plane-center-axis plane normal.
    dir_02_x = torch.nn.functional.normalize(x[:, reference_atoms[2]] - x[:, reference_atoms[0]])
    plane_x = torch.cross(dir_01_x, dir_02_x)
    dir_02_y = torch.nn.functional.normalize(y[:, reference_atoms[2]] - y[:, reference_atoms[0]])
    assert torch.allclose(batchwise_dot(plane_x, dir_02_y), torch.zeros(len(dir_02_y)))


def test_error_less_than_3_mapped_atoms():
    """An error is raised if the number of mapped atoms is less than 3."""
    tfep_map = CartesianMAFMap(
        potential_energy_func=MockPotential(),
        topology_file_path=CHLOROMETHANE_PDB_FILE_PATH,
        coordinates_file_path=CHLOROMETHANE_PDB_FILE_PATH,
        temperature=300*UNITS.kelvin,
        reference_atoms=[0, 1, 2],
        mapped_atoms='index 0:1',

    )

    with pytest.raises(ValueError, match="must be at least 3"):
        tfep_map.setup()


@pytest.mark.parametrize('reference_atoms', [
    (0, 0, 2),
    (0, 2, 2),
    (0, 2, 0),
])
def test_error_reference_atom_overlap(reference_atoms):
    """An error is raised if the center, axis, and/or plane atoms overlap."""
    tfep_map = CartesianMAFMap(
        potential_energy_func=MockPotential(),
        topology_file_path=CHLOROMETHANE_PDB_FILE_PATH,
        coordinates_file_path=CHLOROMETHANE_PDB_FILE_PATH,
        temperature=300*UNITS.kelvin,
        reference_atoms=reference_atoms,
    )

    with pytest.raises(ValueError, match="must be different"):
        tfep_map.setup()


@pytest.mark.parametrize('reference_atoms', [
    (0, 4, 1),
    (0, 1, 4),
    (4, 1, 0),
    (3, 4, 1),
    (1, 4, 3),
])
def test_error_reference_atom_not_mapped(reference_atoms):
    """An error is raised if the reference atoms are not mapped atoms."""
    tfep_map = CartesianMAFMap(
        potential_energy_func=MockPotential(),
        topology_file_path=CHLOROMETHANE_PDB_FILE_PATH,
        coordinates_file_path=CHLOROMETHANE_PDB_FILE_PATH,
        temperature=300*UNITS.kelvin,
        reference_atoms=reference_atoms,
        mapped_atoms=[0, 1, 2],
    )

    with pytest.raises(ValueError, match="is not a mapped atom"):
        tfep_map.setup()
