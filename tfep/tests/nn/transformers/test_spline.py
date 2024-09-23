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

from tfep.nn.transformers.spline import NeuralSplineTransformer
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

def reference_neural_spline(x, x0, y0, widths, heights, slopes, shifts=None):
    """A slow but simple implementation of neural_spline_transformer for testing."""
    x = x.detach().cpu().numpy()
    x0 = x0.detach().cpu().numpy()
    y0 = y0.detach().cpu().numpy()
    widths = widths.detach().cpu().numpy()
    heights = heights.detach().cpu().numpy()
    slopes = slopes.detach().cpu().numpy()
    if shifts is not None:
        shifts = shifts.detach().cpu().numpy()

    batch_size, n_bins, n_features = widths.shape

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
            x_i = x[batch_idx, feat_idx]
            bin_idx = np.digitize(x_i, knots_x[batch_idx, :, feat_idx], right=False) - 1

            # Shift.
            if shifts is not None:
                x_i = x_i - x0[feat_idx] + shifts[batch_idx, feat_idx]
                x_i = x_i % (knots_x[batch_idx, -1] - x0[feat_idx]) + x0[feat_idx]

            # If the value falls outside the limits, we transform linearly.
            if (bin_idx < 0) or (bin_idx == n_bins):
                i = 0 if bin_idx < 0 else -1
                dx = x[batch_idx, feat_idx] - knots_x[batch_idx, i, feat_idx]
                y[batch_idx, feat_idx] = knots_y[batch_idx, i, feat_idx] + slopes[batch_idx, i, feat_idx] * dx
                log_det_J[batch_idx] += np.log(slopes[batch_idx, i, feat_idx])
            else:
                # Neural spline.
                xk = knots_x[batch_idx, bin_idx, feat_idx]
                xk1 = knots_x[batch_idx, bin_idx+1, feat_idx]
                yk = knots_y[batch_idx, bin_idx, feat_idx]
                yk1 = knots_y[batch_idx, bin_idx+1, feat_idx]

                deltak = slopes[batch_idx, bin_idx, feat_idx]
                deltak1 = slopes[batch_idx, bin_idx+1, feat_idx]

                sk = (yk1 - yk) / (xk1 - xk)
                epsilon = (x_i - xk) / (xk1 - xk)

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

@pytest.mark.parametrize('batch_size', [1, 5])
@pytest.mark.parametrize('n_features', [1, 4])
@pytest.mark.parametrize('n_bins', [2, 3])
@pytest.mark.parametrize('circular', [False, True])
@pytest.mark.parametrize('identity_boundary_slopes', [False, True])
def test_neural_spline_get_parameters(batch_size, n_features, n_bins, circular, identity_boundary_slopes):
    """The parameters are split correctly and the boundary slopes are set correctly."""
    x0 = torch.full((n_features,), -1., dtype=torch.double, requires_grad=False)
    xf = -x0

    # Create the transformer.
    transformer = NeuralSplineTransformer(
        x0=x0,
        xf=xf,
        n_bins=n_bins,
        circular=circular,
        identity_boundary_slopes=identity_boundary_slopes,
    )

    # Verify the expected number of parameters.
    n_parameters = (3*n_bins + 1) * n_features
    if identity_boundary_slopes:
        # In circular splines, the boundary slopes are equal but there is a
        # shift parameter.
        if circular:
            n_parameters -= n_features
        else:
            n_parameters -= 2 * n_features
    assert len(transformer.get_identity_parameters(n_features)) == n_parameters

    # Create random parameters.
    x, parameters = create_random_input(
        batch_size,
        n_features,
        n_parameters=n_parameters,
        x_func=torch.randn,
    )

    # Process the parameters.
    widths, heights, slopes, shifts = transformer._get_parameters(parameters)

    # The lengths of the parameters are correct.
    assert widths.shape == (batch_size, n_bins, n_features)
    assert heights.shape == (batch_size, n_bins, n_features)
    assert slopes.shape == (batch_size, n_bins+1, n_features)

    # For circular splines, the shift parameter is present.
    if circular:
        assert shifts.shape == (batch_size, n_features)
    else:
        assert shifts is None

    # The boundary slopes are correct.
    if identity_boundary_slopes:
        assert torch.allclose(slopes[:, 0], torch.ones_like(slopes[:, 0]))
        assert torch.allclose(slopes[:, -1], torch.ones_like(slopes[:, 0]))
    if circular:
        assert torch.allclose(slopes[:, 0], slopes[:, -1])


@pytest.mark.parametrize('batch_size', [1, 3])
@pytest.mark.parametrize('n_features', [1, 4])
@pytest.mark.parametrize('x0', [-2, -1])
@pytest.mark.parametrize('y0', [1, 2])
@pytest.mark.parametrize('n_bins', [2, 5])
@pytest.mark.parametrize('circular', [False, True])
@pytest.mark.parametrize('identity_boundary_slopes', [False, True])
def test_neural_spline_transformer_reference(batch_size, n_features, x0, y0, n_bins, circular, identity_boundary_slopes):
    """Compare PyTorch and reference implementation of neural spline transformer."""
    # Determine the first and final knots of the spline. We arbitrarily set the
    # domain of the first dimension to 0.0 to test different dimensions for
    # different features.
    x0 = torch.full((n_features,), x0, dtype=torch.double, requires_grad=False)
    xf = -x0
    xf[0] = 0
    y0 = torch.full((n_features,), y0, dtype=torch.double, requires_grad=False)
    yf = y0 + xf - x0

    # Deactivate minimum bin size/slope which are not implemented in the reference.
    transformer = NeuralSplineTransformer(x0, xf, n_bins, y0, yf,
                                          min_bin_size=1e-12, min_slope=1e-12)

    # Create random input. Periodic inputs are expected within the domain.
    x, parameters = create_random_input(
        batch_size,
        n_features,
        n_parameters=transformer.n_parameters_per_input*n_features,
        seed=0,
        x_func=torch.rand if circular else torch.randn,
    )
    if circular:
        # x is now between 0 and 1. Scale and shift to match the spline domain.
        x = (x * (xf - x0) + x0)
        assert torch.all(x >= x0) and torch.all(x <= xf)
    else:
        # x is now distributed around 0. We center it and scale it so that the
        # most of the values are between x0 and xf. We detach to make the new x
        # a leaf variable and reset requires_grad.
        x = (x * (xf - x0) + (x0 + xf) / 2)

    # Set requires_grad to check the log_det_J against autograd.
    x = x.detach()
    x.requires_grad = True

    # The parameters for the reference function.
    widths, heights, slopes, shifts = transformer._get_parameters(parameters)

    # Test the result against the reference implementation.
    torch_y, torch_log_det_J = transformer(x, parameters)
    ref_y, ref_log_det_J = reference_neural_spline(x, x0, y0, widths, heights, slopes)
    assert np.allclose(ref_y, torch_y.detach().cpu().numpy())
    assert np.allclose(ref_log_det_J, torch_log_det_J.detach().cpu().numpy())

    # Check y0, yf boundaries are satisfied for inputs within the spline domain.
    assert torch.all((x0 < x) == (y0 < torch_y))
    assert torch.all((x < xf) == (torch_y < yf))

    # Compute the reference log_det_J also with autograd and numpy.
    ref_log_det_J2 = batch_autograd_log_abs_det_J(x, torch_y)
    assert torch.allclose(ref_log_det_J2, torch_log_det_J)

    # Check that inverting returns the original input.
    y = torch_y.detach()
    y.requires_grad = True
    x_inv, log_det_J_inv = transformer.inverse(y, parameters)
    assert torch.allclose(x, x_inv)
    assert torch.allclose(torch_log_det_J+log_det_J_inv, torch.zeros_like(torch_log_det_J))

    # Check also the inverse log_det_J.
    ref_log_det_J_inv = batch_autograd_log_abs_det_J(y, x_inv)
    assert torch.allclose(ref_log_det_J_inv, log_det_J_inv)


def test_circular_spline_transformer_periodic():
    """Test that circular spline transformer conditions for periodicity are verified."""
    batch_size = 5
    n_features = 3
    n_bins = 3

    # Input lower/upper boundaries.
    x0 = torch.tensor([0.0, -1, 2])
    xf = torch.tensor([2.0, -0.5, 5])

    # Create input. The first and second batches are the lower and upper
    # boundaries, respectively. The last batches are random.
    epsilon = 1e-8
    x = torch.cat([
        x0.unsqueeze(0) + epsilon,
        xf.unsqueeze(0) - epsilon,
        torch.rand((batch_size-2, n_features)) * (xf - x0) + x0
    ])

    # Create random parameters.
    parameters = torch.randn((batch_size, (3*n_bins+1) * n_features))

    # Create and run the transformer.
    transformer = NeuralSplineTransformer(x0=x0, xf=xf, n_bins=n_bins, circular=True)
    y, log_det_J = transformer(x, parameters)

    # The slopes of the first and last knots must be the same.
    _, _, slopes, shifts = transformer._get_parameters(parameters)
    assert torch.allclose(slopes[:, 0], slopes[:, -1])

    # The random input are still within boundaries
    assert torch.all(x0 < y)
    assert torch.all(y < xf)

    # The inverse returns the original input.
    x_inv, log_det_J_inv = transformer.inverse(y, parameters)
    assert torch.allclose(x, x_inv)
    assert torch.allclose(log_det_J+log_det_J_inv, torch.zeros_like(log_det_J))

    # If the shifts are 0.0, the boundaries must be mapped to themselves.
    parameters.reshape(batch_size, -1, n_features)[:, 3*n_bins] = 0.0
    y, log_det_J = transformer(x, parameters.reshape(batch_size, -1))
    assert torch.allclose(x[:2], y[:2], atol=10*epsilon)


@pytest.mark.parametrize('batch_size', [1, 3])
@pytest.mark.parametrize('n_features', [1, 4])
@pytest.mark.parametrize('n_bins', [2, 5])
@pytest.mark.parametrize('circular', [False, True])
@pytest.mark.parametrize('identity_boundary_slopes', [False, True])
def test_identity_neural_spline(batch_size, n_features, n_bins, circular, identity_boundary_slopes):
    """Test that get_identity_parameters returns the correct parameters for the identity function."""
    x0 = torch.randn(n_features)
    xf = x0 + torch.abs(torch.rand(n_features)) + 1.

    # Create random input. For circular splines, the input is expected to be
    # within the spline domain.
    if circular:
        x = torch.rand((batch_size, n_features)) * (xf - x0) + x0
    else:
        x = torch.randn((batch_size, n_features)) * (xf - x0) + (x0 + xf) / 2

    # Obtain identity parameters.
    transformer = NeuralSplineTransformer(
        x0=x0,
        xf=xf,
        n_bins=n_bins,
        circular=circular,
        identity_boundary_slopes=identity_boundary_slopes,
    )
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


def test_min_bin_and_slopes():
    """NeuralSplineTransformer sets a minimum bin size and slope."""
    batch_size = 5
    n_bins = 3
    n_features = 3
    x0 = -2.
    xf = 2.
    min_bin_size = 1e-2
    min_slope = 1e-2

    # Create the spline.
    transformer = NeuralSplineTransformer(
        x0=torch.tensor([x0]*n_features),
        xf=torch.tensor([xf]*n_features),
        n_bins=n_bins,
        min_bin_size=min_bin_size,
        min_slope=min_slope,
    )

    # Create random parameters.
    parameters = torch.randn((batch_size, (3*n_bins+1), n_features))
    parameters[:, 0] = parameters[:, n_bins] = parameters[:, 2*n_bins] = -1e3

    # Verify that there are bins/slopes smaller than the threshold.
    widths = torch.nn.functional.softmax(parameters[:, :n_bins], dim=1) * (xf - x0)
    heights = torch.nn.functional.softmax(parameters[:, n_bins:2*n_bins], dim=1) * (xf - x0)
    offset = torch.log(torch.tensor(np.e) - 1.)
    slopes = torch.nn.functional.softplus(parameters[:, 2*n_bins:] + offset)
    assert torch.any(widths < min_bin_size)
    assert torch.any(heights < min_bin_size)
    assert torch.any(slopes < min_slope)

    # The bins/slopes smaller than the threshold has been increased.
    parameters = parameters.reshape(batch_size, -1)
    widths, heights, slopes, shifts = transformer._get_parameters(parameters)
    assert torch.all(widths >= min_bin_size)
    assert torch.all(heights >= min_bin_size)
    assert torch.all(slopes >= min_slope)
