#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test objects and function in the module tfep.nn.flow.centroid.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import pytest
import torch

import tfep.nn.flows
from tfep.nn.flows.oriented import OrientedFlow
from tfep.utils.math import batchwise_dot
from tfep.utils.misc import atom_to_flattened, flattened_to_atom

# =============================================================================
# GLOBAL VARIABLES
# =============================================================================

# Random number generator. Makes sure tests are reproducible from run to run.
GENERATOR = torch.Generator()
GENERATOR.manual_seed(0)


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
# UTILITY FUNCTIONS
# =============================================================================

class IdentityFlow:

    def __init__(self, _):
        pass

    def __call__(self, x):
        return x, torch.zeros(len(x))

    def inverse(self, y):
        return self(y)


class TranslateFlow:

    def __init__(self, _):
        pass

    def __call__(self, x, sign=1):
        n_dofs = x.shape[-1]
        translate = torch.arange(n_dofs, dtype=x.dtype)
        return x + sign*translate, torch.zeros(len(x))

    def inverse(self, y):
        return self(y, sign=-1)


class MyMAF:

    def __init__(self, dimension_in):
        self.maf = tfep.nn.flows.MAF(dimension_in, initialize_identity=False)

    def __call__(self, x):
        return self.maf(x)

    def inverse(self, y):
        return self.maf.inverse(y)


# =============================================================================
# TESTS
# =============================================================================

@pytest.mark.parametrize('flow', [IdentityFlow, TranslateFlow, MyMAF])
@pytest.mark.parametrize('axis_point_idx,plane_point_idx', [
    (0, 1), (1, 0), (3, 1)
])
@pytest.mark.parametrize('axis,plane', [
    ('x', 'xy'),
    ('x', 'xz'),
    ('y', 'xy'),
    ('y', 'yz'),
    ('z', 'xz'),
    ('z', 'yz'),
])
@pytest.mark.parametrize('rotate_back', [True, False])
def test_oriented_flow(flow, axis_point_idx, plane_point_idx, axis, plane, rotate_back):
    """Test that the output of OrientedFlow is in the expected reference frame."""
    batch_size = 10
    n_points = 4

    # Build flow.
    flow = OrientedFlow(
        flow((n_points-1) * 3),
        axis_point_idx=axis_point_idx,
        plane_point_idx=plane_point_idx,
        axis=axis,
        plane=plane,
        rotate_back=rotate_back,
    )

    # Create random input and run flow.
    x = torch.randn(batch_size, n_points * 3, generator=GENERATOR)
    y, log_det_J = flow(x)

    # Check that the axis and plane atoms are where they should.
    x = flattened_to_atom(x)
    y = flattened_to_atom(y)
    if rotate_back:
        expected_directions = torch.nn.functional.normalize(x[:, axis_point_idx])
        expected_normal_planes = torch.nn.functional.normalize(torch.cross(expected_directions, x[:, plane_point_idx]))
    else:
        expected_directions = OrientedFlow._AXES[axis].type(x.dtype)
        expected_normal_planes = [OrientedFlow._AXES[a].type(x.dtype)
                                  for a in ['x', 'y', 'z'] if a not in plane][0]

    # The axis atom is on the expected axis.
    normalized_y_axis = torch.nn.functional.normalize(y[:, axis_point_idx])
    sign = torch.sign(batchwise_dot(normalized_y_axis, expected_directions)).unsqueeze(1)
    assert torch.allclose(sign * normalized_y_axis, expected_directions)

    # The plane atom is orthogonal to the plane normal.
    assert torch.allclose(batchwise_dot(expected_normal_planes, y[:, plane_point_idx]),
                          torch.zeros(batch_size))

    # When rotate_back is True, we can also compute the inverse.
    if rotate_back:
        x_inv, log_det_J_inv = flow.inverse(atom_to_flattened(y))
        assert torch.allclose(atom_to_flattened(x), x_inv)
        assert torch.allclose(log_det_J + log_det_J_inv, torch.zeros_like(log_det_J))
    else:
        # Otherwise an error is raised.
        with pytest.raises(ValueError, match="can be computed only if 'rotate_back' is set"):
            flow.inverse(atom_to_flattened(y))


@pytest.mark.parametrize('axis,plane,expected_axis,expected_plane', [
    (None, 0, 1, 0),
    (None, 2, 0, 2),
    (0, None, 0, 1),
    (1, None, 1, 0),
    (None, None, 0, 1),
])
def test_automatic_axis_plane_selection(axis, plane, expected_axis, expected_plane):
    """Test automatic selection of axis/plane points."""
    flow = OrientedFlow(flow=IdentityFlow(9), axis_point_idx=axis, plane_point_idx=plane)
    assert flow._axis_point_idx == expected_axis
    assert flow._plane_point_idx == expected_plane


def test_return_partial():
    """With return_partial=True, only the propagated DOFs are returned."""
    batch_size = 3
    n_points = 4
    flow = OrientedFlow(MyMAF((n_points-1) * 3),)

    # Default is return_partial == False
    x = torch.randn(batch_size, n_points * 3)
    y, log_det_J = flow(x)
    assert y.shape == (batch_size, n_points*3)

    # The 3 constrained DOFs are not returned.
    flow.return_partial = True
    y, log_det_J = flow(x)
    assert y.shape == (batch_size, n_points*3-3)


def test_error_rotate_and_partial():
    """An error is raised if both rotate_back and partial_result is set."""
    with pytest.raises(ValueError, match="supported only if 'rotate_back=False'"):
        OrientedFlow(IdentityFlow(9), return_partial=True, rotate_back=True)


def test_error_equal_axis_plane_atoms():
    """An error is raised if axis and plane atoms are the same."""
    with pytest.raises(ValueError, match="must be different"):
        OrientedFlow(IdentityFlow(9), axis_point_idx=1, plane_point_idx=1)


def test_error_axis_not_in_plane():
    """An error is raised if the reference axis does not belong to the reference plane."""
    with pytest.raises(ValueError, match="must be constrained on an axis on the same plane"):
        OrientedFlow(IdentityFlow(9), axis='x', plane='yz')
