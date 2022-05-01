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
)
from ..utils import create_random_input


# =============================================================================
# TESTS
# =============================================================================

@pytest.mark.parametrize('batch_size', [2, 5])
@pytest.mark.parametrize('n_features', [2, 5, 8])
def test_affine_transformer_round_trip(batch_size, n_features):
    """Make sure the forward + inverse conposition of affine transformers is equal to the identity."""
    x, coefficients = create_random_input(batch_size, n_features, dtype=torch.double,
                                          n_parameters=2, seed=0)
    shift, log_scale = coefficients[:, 0], coefficients[:, 1]

    # Check that a round trip gives the identity function.
    y, log_det_J_y = affine_transformer(x, shift, log_scale)
    x_inv, log_det_J_x_inv = affine_transformer_inverse(y, shift, log_scale)

    assert torch.allclose(x, x_inv)
    assert torch.allclose(log_det_J_y + log_det_J_x_inv, torch.zeros_like(log_det_J_y))


@pytest.mark.parametrize('batch_size', [2, 5])
@pytest.mark.parametrize('n_features', [2, 5, 8])
@pytest.mark.parametrize('func', [affine_transformer, affine_transformer_inverse])
def test_affine_transformer_log_det_J(batch_size, n_features, func):
    """Check that the log_det_J of the affine transformer is correct."""
    x, coefficients = create_random_input(batch_size, n_features, dtype=torch.double,
                                          n_parameters=2, seed=0)
    shift, log_scale = coefficients[:, 0], coefficients[:, 1]

    # Check the log(abs(det(J))).
    y, log_det_J = func(x, shift, log_scale)
    log_det_J_ref = batch_autograd_log_abs_det_J(x, y)
    assert torch.allclose(log_det_J, log_det_J_ref)
