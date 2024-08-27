#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test objects and function in tfep.nn.transformers.spline.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import numpy as np
import pytest
import torch
import torch.autograd

from tfep.nn.transformers.spline import (
    NeuralSplineTransformer,
    neural_spline_transformer, neural_spline_transformer_inverse,
)
from tfep.utils.math import batch_autograd_log_abs_det_J
from .. import create_random_input


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
# REFERENCE IMPLEMENTATIONS
# =============================================================================

def reference_neural_spline(x, x0, y0, widths, heights, slopes):
    """Reference implementation of neural_spline_transformer for testing."""
    x = x.detach().cpu().numpy()
    x0 = x0.detach().cpu().numpy()
    y0 = y0.detach().cpu().numpy()
    widths = widths.detach().cpu().numpy()
    heights = heights.detach().cpu().numpy()
    slopes = slopes.detach().cpu().numpy()

    batch_size, n_bins, n_features = widths.shape
    n_knots = n_bins + 1

    # Set the slope of the last knot if not given.
    if slopes.shape[1] < n_knots:
        slopes = np.concatenate([slopes, slopes[:, 0:1]], axis=1)

    knots_x = np.empty((batch_size, n_bins+1, n_features), dtype=x.dtype)
    knots_x[:, 0] = x0
    knots_x[:, 1:] = x0 + np.cumsum(widths, axis=1)
    knots_y = np.empty((batch_size, n_bins+1, n_features), dtype=x.dtype)
    knots_y[:, 0] = y0
    knots_y[:, 1:] = y0 + np.cumsum(heights, axis=1)

    y = np.empty_like(x)
    log_det_J = np.zeros(batch_size, dtype=x.dtype)

    for batch_idx in range(batch_size):
        for feat_idx in range(n_features):
            bin_idx = np.digitize(x[batch_idx, feat_idx], knots_x[batch_idx, :, feat_idx], right=False) - 1

            xk = knots_x[batch_idx, bin_idx, feat_idx]
            xk1 = knots_x[batch_idx, bin_idx+1, feat_idx]
            yk = knots_y[batch_idx, bin_idx, feat_idx]
            yk1 = knots_y[batch_idx, bin_idx+1, feat_idx]

            deltak = slopes[batch_idx, bin_idx, feat_idx]
            deltak1 = slopes[batch_idx, bin_idx+1, feat_idx]

            sk = (yk1 - yk) / (xk1 - xk)
            epsilon = (x[batch_idx, feat_idx] - xk) / (xk1 - xk)

            numerator = (yk1 - yk) * (sk * epsilon**2 + deltak * epsilon * (1 - epsilon))
            denominator = sk + (deltak1 + deltak - 2*sk) * epsilon * (1 - epsilon)
            y[batch_idx, feat_idx] = yk + numerator / denominator

            numerator = sk**2 * (deltak1 * epsilon**2 + 2*sk*epsilon*(1 - epsilon) + deltak*(1 - epsilon)**2)
            denominator = (sk + (deltak1 + deltak + - 2*sk) * epsilon * (1 - epsilon))**2
            log_det_J[batch_idx] += np.log(numerator / denominator)

    return y, log_det_J


# =============================================================================
# TESTS
# =============================================================================

@pytest.mark.parametrize('batch_size', [1, 4])
@pytest.mark.parametrize('n_features', [2, 5, 8])
@pytest.mark.parametrize('x0', [-2, -1])
@pytest.mark.parametrize('y0', [1, 2])
@pytest.mark.parametrize('n_bins', [2, 3, 5])
def test_neural_spline_transformer_reference(batch_size, n_features, x0, y0, n_bins):
    """Compare PyTorch and reference implementation of neural spline transformer."""
    # Determine the first and final knots of the spline. We
    # arbitrarily set the domain of the first dimension to 0.0
    # to test different dimensions for different features.
    x0 = torch.full((n_features,), x0, dtype=torch.double, requires_grad=False)
    xf = -x0
    xf[0] = 0
    y0 = torch.full((n_features,), y0, dtype=torch.double, requires_grad=False)
    yf = y0 + xf - x0

    # Create widths, heights, and slopes of the bins.
    x, parameters = create_random_input(
        batch_size,
        n_features,
        n_parameters=(3*n_bins+1) * n_features,
        seed=0,
        x_func=torch.rand,
    )
    parameters = parameters.reshape(batch_size, -1, n_features)

    widths = torch.nn.functional.softmax(parameters[:, :n_bins], dim=1) * (xf - x0)
    heights = torch.nn.functional.softmax(parameters[:, n_bins:2*n_bins], dim=1) * (yf - y0)
    slopes = torch.nn.functional.softplus(parameters[:, 2*n_bins:])

    # x is now between 0 and 1 but it must be between x0 and xf. We detach
    # to make the new x a leaf variable and reset requires_grad.
    x = x.detach() * (xf - x0) + x0
    x.requires_grad = True

    ref_y, ref_log_det_J = reference_neural_spline(x, x0, y0, widths, heights, slopes)
    torch_y, torch_log_det_J = neural_spline_transformer(x, x0, y0, widths, heights, slopes)

    assert np.allclose(ref_y, torch_y.detach().cpu().numpy())
    assert np.allclose(ref_log_det_J, torch_log_det_J.detach().cpu().numpy())

    # Check y0, yf boundaries are satisfied
    assert torch.all(y0 < torch_y)
    assert torch.all(torch_y < yf)

    # Compute the reference log_det_J also with autograd and numpy.
    ref_log_det_J2 = batch_autograd_log_abs_det_J(x, torch_y)
    assert torch.allclose(ref_log_det_J2, torch_log_det_J)

    # Check that inverting returns the original input.
    y = torch_y.detach()
    y.requires_grad = True
    x_inv, log_det_J_inv = neural_spline_transformer_inverse(y, x0, y0, widths, heights, slopes)
    assert torch.allclose(x, x_inv)
    assert torch.allclose(torch_log_det_J+log_det_J_inv, torch.zeros_like(torch_log_det_J))

    # Check also the inverse log_det_J.
    ref_log_det_J_inv = batch_autograd_log_abs_det_J(y, x_inv)
    assert torch.allclose(ref_log_det_J_inv, log_det_J_inv)


@pytest.mark.parametrize('circular', [
    True,
    torch.tensor([0]),
    torch.tensor([1]),
    torch.tensor([0, 1]),
    torch.tensor([1, 2]),
])
def test_circular_spline_transformer_periodic(circular):
    """Test that circular spline transformer conditions for periodicity are verified."""
    batch_size = 5
    n_features = 3
    n_bins = 3

    if circular is True:
        circular_indices = torch.arange(n_features)
    else:
        circular_indices = circular

    # Input lower/upper boundaries.
    x0 = torch.tensor([0.0, -1, 2])
    xf = torch.tensor([2.0, -0.5, 5])

    # Create input. The first and last batch are the lower and upper boundaries
    # respectively. The last batches are random.
    epsilon = 1e-8
    x = torch.cat([
        x0.unsqueeze(0) + epsilon,
        xf.unsqueeze(0) - epsilon,
        torch.rand((batch_size-2, n_features)) * (xf - x0) + x0
    ])

    # Create random parameters.
    parameters = torch.randn((batch_size, (3*n_bins+1) * n_features))

    # Create and run the transformer.
    transformer = NeuralSplineTransformer(x0=x0, xf=xf, n_bins=n_bins, circular=circular)
    y, log_det_J = transformer(x, parameters)

    # The slopes of the first and last knots must be the same.
    _, _, slopes, shifts = transformer._get_parameters(parameters)
    assert torch.allclose(slopes[:, 0, circular_indices], slopes[:, -1, circular_indices])

    # The random input are still within boundaries
    assert torch.all(x0 < y)
    assert torch.all(y < xf)

    # The inverse returns the original input.
    x_inv, log_det_J_inv = transformer.inverse(y, parameters)
    assert torch.allclose(x, x_inv)
    assert torch.allclose(log_det_J+log_det_J_inv, torch.zeros_like(log_det_J))

    # If the shifts are 0.0, the boundaries must be mapped to themselves.
    parameters.reshape(batch_size, -1, n_features)[:, 3*n_bins, circular_indices] = 0.0
    y, log_det_J = transformer(x, parameters.reshape(batch_size, -1))
    assert torch.allclose(x[:2], y[:2], atol=10*epsilon)


@pytest.mark.parametrize('circular', [
    False,
    True,
    torch.tensor([0]),
    torch.tensor([1]),
    torch.tensor([0, 2]),
    torch.tensor([1, 2]),
])
def test_identity_neural_spline(circular):
    """Test that get_identity_parameters returns the correct parameters for the identity function."""
    batch_size = 5
    n_features = 3
    n_bins = 3

    # Create random input.
    x0 = torch.randn(n_features)
    xf = x0 + torch.abs(torch.randn(n_features))
    x = torch.rand((batch_size, n_features)) * (xf - x0) + x0

    # Obtain identity parameters.
    transformer = NeuralSplineTransformer(x0=x0, xf=xf, n_bins=n_bins, circular=circular)
    parameters = transformer.get_identity_parameters(n_features)
    # We need to clone to actually allocate the memory or sliced
    # assignment operations on parameters won't work.
    parameters = parameters.unsqueeze(0).expand(batch_size, -1).clone()

    # Check that the parameters give the identity functions.
    y, log_det_J = transformer(x, parameters)
    assert torch.allclose(x, y)
    assert torch.allclose(log_det_J, torch.zeros_like(log_det_J))

    # The inverse is the identity function as well.
    x_inv, log_det_J_inv = transformer.inverse(y, parameters)
    assert torch.allclose(x, x_inv)
    assert torch.allclose(log_det_J_inv, torch.zeros_like(log_det_J))
