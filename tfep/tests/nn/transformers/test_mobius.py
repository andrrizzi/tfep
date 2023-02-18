#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test objects and function in tfep.nn.transformers.mobius.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import numpy as np
import pytest
import torch
import torch.autograd

from tfep.nn.transformers.mobius import mobius_transformer, unit_cube_to_inscribed_sphere
from ..utils import create_random_input

from tfep.nn.utils import generate_block_sizes
from tfep.utils.math import batch_autograd_log_abs_det_J


# =============================================================================
# REFERENCE IMPLEMENTATIONS
# =============================================================================

def reference_mobius_transformer(x, w, blocks):
    """Reference implementation of MobiusTransformer for testing."""
    x = x.detach().numpy()
    w = w.detach().numpy()
    batch_size, n_features = x.shape

    # Blocks can be an int, in which case x is to be divided in blocks of equal size.
    if isinstance(blocks, int):
        assert n_features % blocks == 0
        blocks = [blocks] * int(n_features / blocks)

    # Initialize the output array.
    y = np.empty_like(x)
    log_det_J = np.zeros(batch_size, dtype=x.dtype)

    # We return also the norm of the input and output.
    x_norm = np.empty(shape=(batch_size, len(blocks)))
    y_norm = np.empty(shape=(batch_size, len(blocks)))

    # The start of the next block.
    block_pointer = 0

    for block_idx, block_size in enumerate(blocks):
        # The input and parameters for the block.
        x_block = x[:, block_pointer:block_pointer+block_size]
        w_block = w[:, block_pointer:block_pointer+block_size]

        # Move the x vector on the unit sphere. Keep the number
        # of dimensions so that broadcasting works.
        x_norm_block = np.linalg.norm(x_block, axis=1, keepdims=True)
        x_normalized_block = x_block / x_norm_block

        # We'll need these terms for the Jacobian as well.
        xw_block = x_normalized_block + w_block
        w_norm = np.linalg.norm(w_block, axis=1, keepdims=True)
        xw_norm = np.linalg.norm(xw_block, axis=1, keepdims=True)
        diff_w_norm = 1 - w_norm**2
        xw_norm_squared = xw_norm**2

        # Compute the output for the block.
        y_normalized_block = diff_w_norm / xw_norm_squared * xw_block + w_block
        y_block = x_norm_block * y_normalized_block

        y[:, block_pointer:block_pointer+block_size] = y_block
        x_norm[:, block_idx] = x_norm_block[:, 0]
        y_norm[:, block_idx] = np.linalg.norm(y_block, axis=1)

        # Compute dxnormalized_i/dx_j.
        dxnormalized_dx = np.empty((batch_size, block_size, block_size))
        for batch_idx in range(batch_size):
            for i in range(block_size):
                for j in range(block_size):
                    dxnormalized_dx[batch_idx, i, j] = - x_block[batch_idx, i] * x_block[batch_idx, j] / x_norm_block[batch_idx, 0]**3
                    if i == j:
                        dxnormalized_dx[batch_idx, i, j] += 1 / x_norm_block[batch_idx, 0]

        # Compute the block Jacobian dy_i/dx_j.
        jacobian = np.empty((batch_size, block_size, block_size), dtype=x.dtype)

        for batch_idx in range(batch_size):
            for i in range(block_size):
                for j in range(block_size):
                    # The first term is d||x||/dx_j * y_normalized, with (d||x||/dx_j)_i = x_j/||x||.
                    jacobian[batch_idx, i, j] = y_normalized_block[batch_idx, i] * x_normalized_block[batch_idx, j]

                    # This is the constant factor in front of the second term.
                    factor = x_norm_block[batch_idx, 0] * diff_w_norm[batch_idx, 0] / xw_norm_squared[batch_idx, 0]

                    # First and second additive terms in the numerator.
                    first_term = dxnormalized_dx[batch_idx, i, j]
                    second_term = 2 / xw_norm_squared[batch_idx, 0] * xw_block[batch_idx, i] * np.dot(xw_block[batch_idx], dxnormalized_dx[batch_idx, :, j])

                    jacobian[batch_idx, i, j] += factor * (first_term - second_term)

        # Compute the log determinant.
        for batch_idx in range(batch_size):
            log_det_J[batch_idx] += np.log(np.abs(np.linalg.det(jacobian[batch_idx])))

        # Point to next block.
        block_pointer += block_size

    return y, log_det_J, x_norm, y_norm


# =============================================================================
# TESTS
# =============================================================================

@pytest.mark.parametrize('n_features,blocks', [
    (3, 3),
    (6, 3),
    (6, 2),
    (5, [3, 2]),
    (7, [3, 2, 2]),
    (8, [3, 2, 2, 1])
])
def test_unit_cube_to_inscribed_sphere(n_features, blocks):
    """Test the mapping from unit cube to its inscribed sphere."""
    # Create a bunch of points within the hypercube with half-side = radius.
    radius = 1
    batch_size = 256
    generator = torch.Generator()
    generator.manual_seed(0)
    w = radius - 2 * radius * torch.rand(batch_size, n_features, generator=generator, dtype=torch.double)

    # In the last two batches we set two cube vertices.
    w[-1] = radius * torch.ones_like(w[-1])
    w[-2] = -radius * torch.ones_like(w[-2])

    # In the third to last batch we try to map the origin.
    w[-3] = torch.zeros_like(w[-3])

    # After the mapping, all points should be within the unit sphere.
    w_mapped = unit_cube_to_inscribed_sphere(w, blocks, shorten_last_block=True)

    blocks = generate_block_sizes(n_features, blocks, shorten_last_block=True)
    block_pointer = 0
    for block_size in blocks:
        norms = []
        for x in [w, w_mapped]:
            x_block = x[:, block_pointer:block_pointer+block_size]
            norms.append((x_block**2).sum(dim=1).sqrt())

        # The test is more meaningful if some of the initial vectors
        # started outside the hypersphere. Exclude the vertices since
        # those are always outside the sphere.
        if block_size > 1:
            assert (norms[0][:-2] > radius).any()
        assert (norms[1] <= radius).all()

        # The cube vertices should be mapped exactly on the sphere surface.
        assert torch.allclose(norms[1][-2:], radius * torch.ones_like(norms[1][-2:]))

        # And the zero should be mapped to zero.
        zero_block = w_mapped[-3, block_pointer:block_pointer+block_size]
        assert torch.all(zero_block == torch.zeros_like(zero_block))

        block_pointer += block_size


@pytest.mark.parametrize('batch_size', [2, 5])
@pytest.mark.parametrize('n_features,blocks', [
    (3, 3),
    (6, 3),
    (6, 2),
    (5, [3, 2]),
    (7, [3, 2, 2]),
    (8, [3, 2, 2, 1])
])
def test_mobius_transformer_reference(batch_size, n_features, blocks):
    """Compare PyTorch and reference implementation of sum-of-squares transformer."""
    x, w = create_random_input(batch_size, n_features,
                               n_parameters=1, seed=0, par_func=torch.rand)
    w = 1 - 2 * w[:, 0]

    # Compare PyTorch and reference.
    ref_y, ref_log_det_J, ref_x_norm, ref_y_norm = reference_mobius_transformer(x, w, blocks)
    torch_y, torch_log_det_J = mobius_transformer(x, w, blocks)

    assert np.allclose(ref_y, torch_y.detach().numpy())
    assert np.allclose(ref_log_det_J, torch_log_det_J.detach().numpy())

    # Make sure the transform doesn't alter the distance from the center of the sphere.
    assert np.allclose(ref_x_norm, ref_y_norm)

    # Compute the reference log_det_J also with autograd and numpy.
    ref_log_det_J2 = batch_autograd_log_abs_det_J(x, torch_y)
    assert torch.allclose(ref_log_det_J2, torch_log_det_J)


@pytest.mark.parametrize('batch_size', [2, 5])
@pytest.mark.parametrize('n_features,blocks', [
    (3, 3),
    (6, 3),
    (6, 2),
    (5, [3, 2]),
    (7, [3, 2, 2]),
    (8, [3, 2, 2, 1])
])
def test_mobius_transformer_gradcheck(batch_size, n_features, blocks):
    """Run autograd.gradcheck on the Mobius transformer."""
    x, w = create_random_input(batch_size, n_features, dtype=torch.double,
                               n_parameters=1, seed=0, par_func=torch.rand)
    w = 1 - 2 * w[:, 0]

    # With a None mask, the module should fall back to the native implementation.
    result = torch.autograd.gradcheck(
        func=mobius_transformer,
        inputs=[x, w, blocks]
    )
    assert result
