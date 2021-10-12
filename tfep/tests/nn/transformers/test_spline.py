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

from tfep.nn.transformers.spline import neural_spline_transformer
from ..utils import create_random_input, reference_log_det_J


# =============================================================================
# REFERENCE IMPLEMENTATIONS
# =============================================================================

def reference_neural_spline(x, x0, y0, widths, heights, slopes):
    """Reference implementation of neural_spline_transformer for testing."""
    x = x.detach().numpy()
    x0 = x0.detach().numpy()
    y0 = y0.detach().numpy()
    widths = widths.detach().numpy()
    heights = heights.detach().numpy()
    slopes = slopes.detach().numpy()

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
            bin_idx = np.digitize(x[batch_idx, feat_idx], knots_x[batch_idx, :, feat_idx], right=False) - 1

            xk = knots_x[batch_idx, bin_idx, feat_idx]
            xk1 = knots_x[batch_idx, bin_idx+1, feat_idx]
            yk = knots_y[batch_idx, bin_idx, feat_idx]
            yk1 = knots_y[batch_idx, bin_idx+1, feat_idx]
            if bin_idx == 0:
                deltak = 1
            else:
                deltak = slopes[batch_idx, bin_idx-1, feat_idx]
            if bin_idx == n_bins-1:
                deltak1 = 1
            else:
                deltak1 = slopes[batch_idx, bin_idx, feat_idx]

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

@pytest.mark.parametrize('batch_size', [2, 5])
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
    n_parameters = 3*n_bins - 1
    x, parameters = create_random_input(batch_size, n_features,
                                        n_parameters=n_parameters, seed=0,
                                        x_func=torch.rand)

    widths = torch.nn.functional.softmax(parameters[:, :n_bins], dim=1) * (xf - x0)
    heights = torch.nn.functional.softmax(parameters[:, n_bins:2*n_bins], dim=1) * (yf - y0)
    slopes = torch.nn.functional.softplus(parameters[:, 2*n_bins:])

    # x is now between 0 and 1 but it must be between x0 and xf. We detach
    # to make the new x a leaf variable and reset requires_grad.
    x = x.detach() * (xf - x0) + x0
    x.requires_grad = True

    ref_y, ref_log_det_J = reference_neural_spline(x, x0, y0, widths, heights, slopes)
    torch_y, torch_log_det_J = neural_spline_transformer(x, x0, y0, widths, heights, slopes)

    assert np.allclose(ref_y, torch_y.detach().numpy())
    assert np.allclose(ref_log_det_J, torch_log_det_J.detach().numpy())

    # Check y0, yf boundaries are satisfied
    assert torch.all(y0 < torch_y)
    assert torch.all(torch_y < yf)

    # Compute the reference log_det_J also with autograd and numpy.
    ref_log_det_J2 = reference_log_det_J(x, torch_y)
    assert np.allclose(ref_log_det_J2, torch_log_det_J.detach().numpy())
