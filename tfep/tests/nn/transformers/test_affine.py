#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test objects and function in tfep.nn.transformers.affine.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import pytest
import torch
import torch.autograd

from tfep.utils.math import batch_autograd_log_abs_det_J
from tfep.nn.transformers.affine import (
    affine_transformer,
    affine_transformer_inverse,
    volume_preserving_shift_transformer,
    volume_preserving_shift_transformer_inverse,
)
from ..utils import create_random_input


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
# UTILS
# =============================================================================

def get_random_input(batch_size, n_features, scale):
    """Create random input and parameters."""
    n_parameters = 2 if scale else 1
    x, coefficients = create_random_input(batch_size, n_features, seed=0,
                                          n_parameters=n_parameters)
    shift = coefficients[:, 0]

    if scale:
        log_scale = coefficients[:, 1]
        return x, [shift, log_scale]

    return x, [shift]


# =============================================================================
# TESTS
# =============================================================================

@pytest.mark.parametrize('batch_size', [2, 5])
@pytest.mark.parametrize('n_features', [2, 5, 8])
@pytest.mark.parametrize('volume_preserving', [False, True])
def test_affine_transformer_round_trip(batch_size, n_features, volume_preserving):
    """Make sure the forward + inverse conposition of affine transformers is equal to the identity."""
    x, parameters = get_random_input(batch_size, n_features, scale=not volume_preserving)

    # Select function.
    if volume_preserving:
        func = volume_preserving_shift_transformer
        inv = volume_preserving_shift_transformer_inverse
    else:
        func = affine_transformer
        inv = affine_transformer_inverse

    # Check that a round trip gives the identity function.
    y, log_det_J_y = func(x, *parameters)
    x_inv, log_det_J_x_inv = inv(y, *parameters)

    assert torch.allclose(x, x_inv)
    assert torch.allclose(log_det_J_y + log_det_J_x_inv, torch.zeros_like(log_det_J_y))


@pytest.mark.parametrize('batch_size', [2, 5])
@pytest.mark.parametrize('n_features', [2, 5, 8])
@pytest.mark.parametrize('func', [
    affine_transformer, affine_transformer_inverse,
    volume_preserving_shift_transformer, volume_preserving_shift_transformer_inverse,
])
def test_affine_transformer_log_det_J(batch_size, n_features, func):
    """Check that the log_det_J of the affine transformer is correct."""
    volume_preserving = 'volume_preserving' in func.__name__
    x, parameters = get_random_input(batch_size, n_features, scale=not volume_preserving)

    # Check the log(abs(det(J))).
    y, log_det_J = func(x, *parameters)
    log_det_J_ref = batch_autograd_log_abs_det_J(x, y)
    assert torch.allclose(log_det_J, log_det_J_ref)

    # Check the volume preserving scaling.
    if volume_preserving:
        assert torch.allclose(log_det_J, torch.zeros_like(log_det_J))


@pytest.mark.parametrize('batch_size', [1, 5])
@pytest.mark.parametrize('periodic_indices', [[0, 2, 3], [2, 4]])
@pytest.mark.parametrize('periodic_limits', [[0., 1.], [-2., 3.]])
def test_volume_preserving_transformer_periodic(batch_size, periodic_indices, periodic_limits):
    """Test that periodic degrees of freedom remain within the boundaries."""
    n_features = 5
    periodic_indices = torch.tensor(periodic_indices)
    periodic_limits = torch.tensor(periodic_limits)

    # Create input and shifts.
    x, parameters = get_random_input(batch_size, n_features, scale=False)
    shift = parameters[0].detach()
    shift.requires_grad = False

    # Make sure we shift the periodic features above the limits.
    for i, dof_idx in enumerate(periodic_indices):
        if i % 2 == 0:
            shift[:, dof_idx] += 10.0
            assert torch.all(shift[:, dof_idx] > periodic_limits[1])
        else:
            shift[:, dof_idx] -= 10.0
            assert torch.all(shift[:, dof_idx] < periodic_limits[0])

    # Shift.
    y, log_det_J_y = volume_preserving_shift_transformer(x, shift, periodic_indices, periodic_limits)
    x_inv, log_det_J_x_inv = volume_preserving_shift_transformer_inverse(y, shift, periodic_indices, periodic_limits)

    # Check that all the periodic DOFs are within the limits.
    assert torch.all(y[:, periodic_indices] >= periodic_limits[0])
    assert torch.all(y[:, periodic_indices] <= periodic_limits[1])
    assert torch.all(x_inv[:, periodic_indices] >= periodic_limits[0])
    assert torch.all(x_inv[:, periodic_indices] <= periodic_limits[1])
