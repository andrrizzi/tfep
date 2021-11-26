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
from tfep.utils.math import normalize_vector, batchwise_dot
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
def test_centered_centroid_flow(flow, axis_point_idx, plane_point_idx, axis, plane, rotate_back):
    """Test with identity and MAF flow that centroid is in the expected position."""
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
        round_off_imprecisions=False
    )

    # Create random input and run flow.
    x = torch.randn(batch_size, n_points * 3, generator=GENERATOR)
    y, log_det_J = flow(x)

    # Check that the axis and plane atoms are where they should.
    x = flattened_to_atom(x)
    y = flattened_to_atom(y)
    if rotate_back:
        expected_directions = normalize_vector(x[:, axis_point_idx])
        expected_normal_planes = normalize_vector(torch.cross(expected_directions, x[:, plane_point_idx]))
    else:
        expected_directions = OrientedFlow._AXES[axis].type(x.dtype)
        expected_normal_planes = [OrientedFlow._AXES[a].type(x.dtype)
                                  for a in ['x', 'y', 'z'] if a not in plane][0]

    # The axis atom is on the expected axis.
    assert torch.allclose(normalize_vector(y[:, axis_point_idx]), expected_directions)

    # The plane atom is orthogonal to the plane normal.
    assert torch.allclose(
        batchwise_dot(expected_normal_planes, y[:, plane_point_idx]),
        torch.zeros(batch_size)
    )

    # When translate_back is True, we can also compute the inverse.
    if rotate_back:
        x_inv, log_det_J_inv = flow.inverse(atom_to_flattened(y))
        assert torch.allclose(atom_to_flattened(x), x_inv)
        assert torch.allclose(log_det_J + log_det_J_inv, torch.zeros_like(log_det_J))
