#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test objects and function in tfep.nn.transformers.moebius.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import numpy as np
import pytest
import torch
import torch.autograd

from tfep.nn.transformers.moebius import MoebiusTransformer, SymmetrizedMoebiusTransformer
from ..utils import create_random_input

from tfep.utils.math import batch_autograd_log_abs_det_J


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

def create_moebius_random_input(batch_size, n_features, dimension, unit_sphere):
    """Create random input and parameter for the Moebius transformation."""
    x, w = create_random_input(batch_size, n_features, n_parameters=1, seed=0, par_func=torch.randn)

    # Map the points on the unit sphere.
    if unit_sphere:
        x = x.reshape(batch_size, -1, dimension)
        x = torch.nn.functional.normalize(x, dim=-1)
        x = x.reshape(batch_size, -1)

    return x, w


# =============================================================================
# REFERENCE IMPLEMENTATIONS
# =============================================================================

def reference_moebius_transformer(x, w, dimension):
    """Reference implementation of MoebiusTransformer for testing.

    See tfep.nn.transformers.moebius_transformer for the documentation.

    """
    x = x.detach().numpy()
    w = w.detach().numpy()
    batch_size, n_features = x.shape
    n_vectors = n_features // dimension

    # Initialize the output array.
    y = np.empty_like(x)

    # We return also the norm of the input and output.
    x_norm = np.empty(shape=(batch_size, n_vectors))
    y_norm = np.empty(shape=(batch_size, n_vectors))

    # The start of the next block.
    for vector_idx in range(n_vectors):
        start = vector_idx * dimension
        end = start + dimension

        # The input and parameters for the block.
        x_vec = x[:, start:end]
        w_vec = w[:, start:end]

        # Move the x vector on the unit sphere. Keep the number
        # of dimensions so that broadcasting works.
        x_vec_norm = np.linalg.norm(x_vec, axis=1, keepdims=True)
        x_vec_normalized = x_vec / x_vec_norm

        # Map the parameter vectors to the unit sphere.
        w_norm = np.linalg.norm(w_vec, axis=1, keepdims=True)
        w_vec_sphere = 0.99 / (1 + w_norm) * w_vec

        # Compute Moebius transform.
        diff_vec = x_vec_normalized - w_vec_sphere
        w_sphere_norm2 = np.sum(w_vec_sphere**2, axis=1, keepdims=True)
        diff_norm2 = np.sum(diff_vec**2, axis=1, keepdims=True)
        y_vec = (1 - w_sphere_norm2) / diff_norm2 * diff_vec - w_vec_sphere

        # Recover output vector norm.
        y_vec = x_vec_norm * y_vec

        # Update returned values.
        y[:, start:end] = y_vec
        x_norm[:, vector_idx] = x_vec_norm[:, 0]
        y_norm[:, vector_idx] = np.linalg.norm(y_vec, axis=1)

    return y, x_norm, y_norm


def reference_symmetrized_moebius_transformer(x, w, dimension):
    """Reference implementation of SymmetrizedMoebiusTransformer for testing.

    See tfep.nn.transformers.symmetrized_moebius_transformer for the documentation.

    """
    batch_size, n_features = x.shape
    n_vectors = n_features // dimension

    # Compute the symmetrized transformation.
    f_w, x_norm, f_w_norm = reference_moebius_transformer(x, w, dimension)
    f_iw, _, _ = reference_moebius_transformer(x, -w, dimension)
    f_symmetrized = f_w + f_iw

    f_symmetrized_norm = np.empty(shape=(batch_size, n_vectors))

    # Rescale the
    for vector_idx in range(n_vectors):
        start = vector_idx * dimension
        end = start + dimension
        f_symmetrized_vec = f_symmetrized[:, start:end]

        # Rescale.
        f_symmetrized_vec_norm = np.linalg.norm(f_symmetrized_vec, axis=1, keepdims=True)
        x_vec_norm = x_norm[:, vector_idx:vector_idx+1]
        f_symmetrized_vec = f_symmetrized_vec * x_vec_norm / f_symmetrized_vec_norm

        # Update returned values.
        f_symmetrized[:, start:end] = f_symmetrized_vec
        f_symmetrized_norm[:, vector_idx] = np.linalg.norm(f_symmetrized_vec, axis=1)

    # Rescale symmetrized vector to have the same norm of the input tensor.
    return f_symmetrized, x_norm, f_symmetrized_norm


# =============================================================================
# TESTS
# =============================================================================

@pytest.mark.parametrize('batch_size', [1, 3, 100])
@pytest.mark.parametrize('n_features,dimension', [
    (4, 2),
    (3, 3),
    (6, 3),
    (6, 2),
    (4, 4),
    (8, 4),
])
@pytest.mark.parametrize('unit_sphere', [True, False])
@pytest.mark.parametrize('transformer_cls', [
    MoebiusTransformer,
    SymmetrizedMoebiusTransformer
])
def test_moebius_transformer_reference(batch_size, n_features, dimension, unit_sphere, transformer_cls):
    """Compare PyTorch and reference implementation of MoebiusTransformer."""
    x, w = create_moebius_random_input(batch_size, n_features, dimension, unit_sphere)

    # Compute transformation with pytorch.
    if transformer_cls == MoebiusTransformer:
        transformer = transformer_cls(dimension=dimension, unit_sphere=unit_sphere)
    else:
        transformer = transformer_cls(dimension=dimension)
    torch_y, torch_log_det_J = transformer(x, w)

    # Compare PyTorch and reference implementation.
    if isinstance(transformer, MoebiusTransformer):
        ref_y, ref_x_norm, ref_y_norm = reference_moebius_transformer(x, w[:, 0], dimension)
    else:
        ref_y, ref_x_norm, ref_y_norm = reference_symmetrized_moebius_transformer(x, w[:, 0], dimension)
    assert np.allclose(ref_y, torch_y.detach().numpy())

    # Make sure the transform doesn't alter the distance from the center of the sphere.
    assert np.allclose(ref_x_norm, ref_y_norm)

    # Compute the reference log_det_J also with autograd.
    ref_log_det_J = batch_autograd_log_abs_det_J(x, torch_y)
    assert torch.allclose(ref_log_det_J, torch_log_det_J)


@pytest.mark.parametrize('batch_size', [1, 3, 100])
@pytest.mark.parametrize('n_features,dimension', [
    (4, 2),
    (3, 3),
    (6, 3),
    (6, 2),
    (4, 4),
    (8, 4),
])
@pytest.mark.parametrize('unit_sphere', [True, False])
@pytest.mark.parametrize('transformer_cls', [
    MoebiusTransformer,
    SymmetrizedMoebiusTransformer
])
def test_moebius_transformer_identity(batch_size, n_features, dimension, unit_sphere, transformer_cls):
    """The MoebiusTransform can implement the identity function correctly."""
    x, _ = create_moebius_random_input(batch_size, n_features, dimension, unit_sphere)

    # The identity should be encoded with parameters w = 0.
    w = torch.zeros_like(x).unsqueeze(1)

    # Compare PyTorch and reference.
    if transformer_cls == MoebiusTransformer:
        transformer = transformer_cls(dimension=dimension, unit_sphere=unit_sphere)
    else:
        transformer = transformer_cls(dimension=dimension)

    y, log_det_J = transformer(x, w)
    assert torch.allclose(x, y)
    assert torch.allclose(log_det_J, torch.zeros_like(log_det_J))


@pytest.mark.parametrize('batch_size', [1, 3, 100])
@pytest.mark.parametrize('n_features,dimension', [
    (4, 2),
    (3, 3),
    (6, 3),
    (6, 2),
    (4, 4),
    (8, 4),
])
@pytest.mark.parametrize('unit_sphere', [True, False])
@pytest.mark.parametrize('transformer_cls', [
    MoebiusTransformer,
    SymmetrizedMoebiusTransformer
])
def test_moebius_transformer_inverse(batch_size, n_features, dimension, unit_sphere, transformer_cls):
    """Compare PyTorch and reference implementation of MoebiusTransformer."""
    x, w = create_moebius_random_input(batch_size, n_features, dimension, unit_sphere)

    # Composing forward and inverse must yield the identity function.
    if transformer_cls == MoebiusTransformer:
        transformer = transformer_cls(dimension=dimension, unit_sphere=unit_sphere)
    else:
        transformer = transformer_cls(dimension=dimension)

    y, log_det_J = transformer(x, w)
    x_inv, log_det_J_inv = transformer.inverse(y, w)
    assert torch.allclose(x, x_inv)
    assert torch.allclose(log_det_J + log_det_J_inv, torch.zeros_like(log_det_J))


@pytest.mark.parametrize('batch_size', [1, 3, 100])
@pytest.mark.parametrize('n_features,dimension', [
    (4, 2),
    (3, 3),
    (6, 3),
    (6, 2),
    (4, 4),
    (8, 4),
])
@pytest.mark.parametrize('unit_sphere', [True, False])
def test_symmetrized_moebius_transformer_flip_equivariance(batch_size, n_features, dimension, unit_sphere):
    """The SymmetrizedMoebiusTransformer is flip equivariant w.r.t. its input."""
    x, w = create_moebius_random_input(batch_size, n_features, dimension, unit_sphere)

    # Compare the transformation with the input and its negative.
    transformer = SymmetrizedMoebiusTransformer(dimension=dimension)
    y, log_det_J = transformer(x, w)
    y_neg, log_det_J_neg = transformer(-x, w)
    assert torch.allclose(y, -y_neg)
