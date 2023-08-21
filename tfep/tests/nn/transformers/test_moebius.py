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

from tfep.nn.transformers.moebius import MoebiusTransformer
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
# REFERENCE IMPLEMENTATIONS
# =============================================================================

def reference_moebius_transformer(x, w, dimension):
    """Reference implementation of MoebiusTransformer for testing.

    See tfep.nn.transformers.moebius_transformer for the docum

    References
    ----------
    [1] Rezende DJ, Papamakarios G, Racanière S, Albergo MS, Kanwar G, Shanahan PE,
        Cranmer K. Normalizing Flows on Tori and Spheres. arXiv preprint
        arXiv:2002.02428. 2020 Feb 6.
    [2] Köhler J, Invernizzi M, De Haan P, Noé F. Rigid body flows for sampling
        molecular crystal structures. arXiv preprint arXiv:2301.11355. 2023 Jan 26.

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


# =============================================================================
# TESTS
# =============================================================================

@pytest.mark.parametrize('batch_size', [1, 3, 100])
@pytest.mark.parametrize('n_features,dimension', [
    (3, 3),
    (6, 3),
    (6, 2),
    (4, 4),
    (8, 4),
])
@pytest.mark.parametrize('unit_sphere', [True, False])
def test_moebius_transformer_reference(batch_size, n_features, dimension, unit_sphere):
    """Compare PyTorch and reference implementation of MoebiusTransformer."""
    x, w = create_random_input(batch_size, n_features, n_parameters=1, seed=0, par_func=torch.randn)

    # Map the points on the unit sphere.
    if unit_sphere:
        x = x.reshape(batch_size, -1, dimension)
        x = x / torch.linalg.norm(x, dim=-1, keepdim=True)
        x = x.reshape(batch_size, -1)

    # Compare PyTorch and reference.
    ref_y, ref_x_norm, ref_y_norm = reference_moebius_transformer(x, w[:, 0], dimension)
    transformer = MoebiusTransformer(dimension=dimension, unit_sphere=unit_sphere)
    torch_y, torch_log_det_J = transformer(x, w)
    assert np.allclose(ref_y, torch_y.detach().numpy())

    # Make sure the transform doesn't alter the distance from the center of the sphere.
    assert np.allclose(ref_x_norm, ref_y_norm)

    # Compute the reference log_det_J also with autograd.
    ref_log_det_J = batch_autograd_log_abs_det_J(x, torch_y)
    assert torch.allclose(ref_log_det_J, torch_log_det_J)


@pytest.mark.parametrize('batch_size', [1, 3, 100])
@pytest.mark.parametrize('n_features,dimension', [
    (3, 3),
    (6, 3),
    (6, 2),
    (4, 4),
    (8, 4),
])
@pytest.mark.parametrize('unit_sphere', [True, False])
def test_moebius_transformer_identity(batch_size, n_features, dimension, unit_sphere):
    """The MoebiusTransform can implement the identity function correctly."""
    x = create_random_input(batch_size, n_features, seed=0, par_func=torch.randn)

    # Map the points on the unit sphere.
    if unit_sphere:
        x = x.reshape(batch_size, -1, dimension)
        x = x / torch.linalg.norm(x, dim=-1, keepdim=True)
        x = x.reshape(batch_size, -1)

    # The identity should be encoded with parameters w = 0.
    w = torch.zeros_like(x).unsqueeze(1)

    # Compare PyTorch and reference.
    transformer = MoebiusTransformer(dimension=dimension, unit_sphere=unit_sphere)
    y, log_det_J = transformer(x, w)
    assert torch.allclose(x, y)
    assert torch.allclose(log_det_J, torch.zeros_like(log_det_J))
