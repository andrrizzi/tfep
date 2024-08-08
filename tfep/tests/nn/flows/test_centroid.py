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

import numpy as np
import pytest
import torch

import tfep.nn.flows
from tfep.nn.flows.centroid import CenteredCentroidFlow
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
        self.maf = tfep.nn.flows.MAF(degrees_in=list(range(dimension_in)), initialize_identity=False)

    def __call__(self, x):
        return self.maf(x)

    def inverse(self, y):
        return self.maf.inverse(y)


# =============================================================================
# TESTS
# =============================================================================

def test_inconsistent_configuration():
    """Test that subset_point_indices and weights must have the same length."""
    space_dimension = 3
    with pytest.raises(ValueError, match='length'):
        CenteredCentroidFlow(
            None,  # flow, not needed.
            space_dimension,
            subset_point_indices=[2, 3, 4],
            weights=[4, 6]
        )


@pytest.mark.parametrize('exclude_fixed_point', [False, True])
@pytest.mark.parametrize('subset_point_indices,weights', [
    (None, None),
    ([0, 1], None),
    ([1, 2], torch.tensor([0.25, 0.75])),
])
@pytest.mark.parametrize('fixed_point_idx', [0, 1])
def test_compute_centroid(exclude_fixed_point, subset_point_indices, weights, fixed_point_idx):
    """Test CenteredCentroidFlow._compute_centroid()."""
    coords = torch.tensor([
        [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
        [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
    ], dtype=torch.float64)

    # Initialize flow.
    flow = CenteredCentroidFlow(
        None,  # Flow. Not needed here.
        space_dimension=coords.shape[-1],
        subset_point_indices=subset_point_indices,
        weights=weights,
        fixed_point_idx=fixed_point_idx,
    )

    # Compute the centroid.
    centroid = flow._compute_centroid(coords, exclude_fixed_point)
    if exclude_fixed_point:
        centroid, weight = centroid

    # Compute the expected centroid with numpy.
    coords = coords.detach().cpu().numpy()
    if subset_point_indices is not None:
        coords = coords[:, subset_point_indices]

    if weights is not None:
        weights = weights.detach().cpu().numpy()

    expected_centroid = np.average(coords, axis=1, weights=weights)
    if exclude_fixed_point:
        w = 1/coords.shape[1] if weights is None else weights[fixed_point_idx]
        assert w == weight
        expected_centroid = expected_centroid - coords[:, fixed_point_idx]*w

    assert np.allclose(centroid, expected_centroid)


@pytest.mark.parametrize('flow', [IdentityFlow, TranslateFlow, MyMAF])
@pytest.mark.parametrize('space_dimension', [1, 2, 3])
@pytest.mark.parametrize('subset_point_indices,weights', [
    (None, None),
    ([0, 1], None),
    ([0, 2], torch.tensor([0.25, 0.75])),
])
@pytest.mark.parametrize('origin', [None, True])
@pytest.mark.parametrize('translate_back', [True, False])
def test_centered_centroid_flow(
        flow,
        space_dimension,
        subset_point_indices,
        weights,
        origin,
        translate_back
):
    """Test with identity and MAF flow that centroid is in the expected position."""
    batch_size = 2
    n_points = 3

    # The dimension of origin depends on space_dimension.
    if origin is True:
        origin = torch.ones(space_dimension)

    # Create input. E.g., x is [[[0, 0, 0], [1, 1, 1], [2, 2, 2]],
    #                           [[1, 1, 1], [2, 2, 2], [3, 3, 3]].
    x = []
    for batch_idx in range(batch_size):
        x.append(np.arange(batch_idx, n_points+batch_idx, dtype=np.float64).repeat(
            space_dimension).reshape((n_points, space_dimension)))
    x = np.array(x)

    # Compute expected centroid.
    if subset_point_indices is None:
        expected_centroid = np.average(x, axis=1, weights=weights)
    else:
        expected_centroid = np.average(x[:, subset_point_indices], axis=1, weights=weights)

    # Build flow.
    flow = CenteredCentroidFlow(
        flow((n_points-1) * space_dimension),
        space_dimension,
        subset_point_indices=subset_point_indices,
        weights=weights,
        origin=origin,
        translate_back=translate_back
    )

    # Run flow.
    x = torch.tensor(atom_to_flattened(x))
    y, log_det_J = flow(x)

    # Compute the new centroid position.
    y_atom_shape = flattened_to_atom(y.detach().cpu().numpy(), space_dimension)
    if subset_point_indices is None:
        new_centroid = np.average(y_atom_shape, axis=1, weights=weights)
    else:
        new_centroid = np.average(y_atom_shape[:, subset_point_indices], axis=1, weights=weights)

    # Check that centroid is correct.
    if translate_back:
        assert np.allclose(new_centroid, expected_centroid)
    else:
        if origin is None:
            origin = np.zeros(space_dimension)
        assert np.allclose(new_centroid, origin)

    # When translate_back is True, we can also compute the inverse.
    if translate_back:
        x_inv, log_det_J_inv = flow.inverse(y)
        assert torch.allclose(x, x_inv)
        assert torch.allclose(log_det_J + log_det_J_inv, torch.zeros_like(log_det_J))
