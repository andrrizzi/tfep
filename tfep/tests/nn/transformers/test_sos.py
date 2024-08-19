#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test objects and function in tfep.nn.transformers.sos.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import numpy as np
import pytest
import torch
import torch.autograd

from tfep.nn.transformers.affine import AffineTransformer
from tfep.nn.transformers.sos import sos_polynomial_transformer, SOSPolynomialTransformer
from tfep.utils.math import batch_autograd_log_abs_det_J
from ..utils import create_random_input


# =============================================================================
# REFERENCE IMPLEMENTATIONS
# =============================================================================

def reference_sos_polynomial_transformer(x, coefficients):
    """Reference implementation of SOSPolynomialTransformer for testing."""
    x = x.detach().cpu().numpy()
    coefficients = coefficients.detach().cpu().numpy()
    batch_size, n_coefficients, n_features = coefficients.shape
    n_polynomials = (n_coefficients - 1) // 2

    # This is the returned value.
    y = np.empty(shape=x.shape)
    det_J = np.ones(batch_size)

    for batch_idx in range(batch_size):
        for i in range(n_features):
            x_i = x[batch_idx, i]
            coefficients_i = coefficients[batch_idx, :, i]

            # Compute all squared polynomials.
            squared_polynomials = []
            for k in range(n_polynomials):
                a_k0 = coefficients_i[1 + k*2]
                a_k1 = coefficients_i[2 + k*2]
                poly = np.poly1d([a_k1, a_k0])
                squared_polynomials.append(np.polymul(poly, poly))

            # Sum the squared polynomials.
            sum_of_squares_poly = squared_polynomials[0]
            for poly in squared_polynomials[1:]:
                sum_of_squares_poly = np.polyadd(sum_of_squares_poly, poly)

            # The integrand is the derivative w.r.t. the input.
            det_J[batch_idx] *= np.polyval(sum_of_squares_poly, x_i)

            # Integrate and sum constant term.
            a_0 = coefficients_i[0]
            sum_of_squares_poly = np.polyint(sum_of_squares_poly, k=a_0)
            y[batch_idx, i] = np.polyval(sum_of_squares_poly, x_i)

    return y, np.log(np.abs(det_J))


# =============================================================================
# TESTS
# =============================================================================

@pytest.mark.parametrize('batch_size', [2, 5])
@pytest.mark.parametrize('n_features', [2, 5, 8])
@pytest.mark.parametrize('n_polynomials', [2, 3, 5])
def test_sos_polynomial_transformer_reference(batch_size, n_features, n_polynomials):
    """Compare PyTorch and reference implementation of sum-of-squares transformer."""
    x, coefficients = create_random_input(batch_size, n_features,
                                          n_parameters=1+2*n_polynomials, seed=0)

    ref_y, ref_log_det_J = reference_sos_polynomial_transformer(x, coefficients)
    torch_y, torch_log_det_J = sos_polynomial_transformer(x, coefficients)

    assert np.allclose(ref_y, torch_y.detach().cpu().numpy())
    assert np.allclose(ref_log_det_J, torch_log_det_J.detach().cpu().numpy())

    # Compute the reference log_det_J also with autograd and numpy.
    ref_log_det_J2 = batch_autograd_log_abs_det_J(x, torch_y)
    assert torch.allclose(ref_log_det_J2, torch_log_det_J)


@pytest.mark.parametrize('batch_size', [2, 5])
@pytest.mark.parametrize('n_features', [2, 5, 8])
@pytest.mark.parametrize('n_polynomials', [2, 3, 5])
def test_sos_polynomial_transformer_gradcheck(batch_size, n_features, n_polynomials):
    """Run autograd.gradcheck on the SOS polynomial transformer."""
    x, coefficients = create_random_input(batch_size, n_features, dtype=torch.double,
                                          n_parameters=1+2*n_polynomials, seed=0)

    # With a None mask, the module should fall back to the native implementation.
    result = torch.autograd.gradcheck(
        func=sos_polynomial_transformer,
        inputs=[x, coefficients]
    )
    assert result


@pytest.mark.parametrize('n_polynomials', [2, 3])
def test_sos_affine_transformer_equivalence(n_polynomials):
    """The SOS polynomial is a generalization of the affine transformer."""
    batch_size = 2
    dimension = 5

    # Make sure the input arguments are deterministic.
    generator = torch.Generator()
    generator.manual_seed(0)

    # Create random input.
    x, affine_parameters = create_random_input(batch_size, dimension, n_parameters=2, dtype=torch.double)

    # Create coefficients for an SOS polynomial that translate into an affine transformer.
    sos_coefficients = torch.zeros(
        size=(batch_size, 1 + n_polynomials*2, dimension), dtype=torch.double)
    sos_coefficients[:, 0] = affine_parameters[:, 0].clone()

    # Divide the scale coefficient equally among all polynomials.
    # The affine transformer takes the log scale as input parameter.
    scale = torch.sqrt(torch.exp(affine_parameters[:, 1]) / n_polynomials)
    for poly_idx in range(n_polynomials):
        sos_coefficients[:, 1 + poly_idx*2] = scale.clone()

    # Check that they are equivalent.
    affine_y, affine_log_det_J = AffineTransformer()(x, affine_parameters)
    sos_transformer = SOSPolynomialTransformer(n_polynomials=n_polynomials)
    sos_y, sos_log_det_J = sos_transformer(x, sos_coefficients.reshape(batch_size, -1))

    assert torch.allclose(affine_y, sos_y)
    assert torch.allclose(affine_log_det_J, sos_log_det_J)
