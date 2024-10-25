#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test MAF layer in ``tfep.nn.embeddings.mafembed``.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import pytest
import torch

from tfep.nn.embeddings.mafembed import PeriodicEmbedding, FlipInvariantEmbedding
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
# UTILITY FUNCTIONS
# =============================================================================

def create_input(batch_size, n_features_in, limits=None, periodic_indices=None, seed=0):
    """Create random input with correct boundaries.

    If limits=(x0, xf) are passed, the input is constrained to be within x0 and
    xf. x0 and xf can be both floats or Tensors of size (n_features,).

    If periodic_indices is passed, only these DOFs are restrained between x0
    and xf.

    """
    # The non-periodic features do not need to be restrained between 0 and 1.
    x = create_random_input(batch_size, n_features_in, x_func=torch.randn, seed=seed)

    if limits is not None:
        if periodic_indices is None:
            # All DOFs are periodic. Ignore previous random input.
            periodic_indices = torch.tensor(list(range(n_features_in)))

        x_periodic = create_random_input(batch_size, len(periodic_indices), x_func=torch.rand, seed=seed)
        x_periodic = x_periodic * (limits[1] - limits[0]) + limits[0]
        x = x.clone()
        x[:, periodic_indices] = x_periodic

    return x


# =============================================================================
# TEST PERIODIC EMBEDDING
# =============================================================================

@pytest.mark.parametrize('n_periodic', [2, 3, 5])
@pytest.mark.parametrize('limits', [
    (0., 1.),
    (-1., 1.),
    (0., 2*torch.pi),
    (-torch.pi, torch.pi),
    (-5., -2.5),
])
def test_periodic_embedding(n_periodic, limits):
    """Test that PeriodicEmbedding lifts the correct degrees of freedom."""
    batch_size = 3
    n_features_in = 5
    limits = torch.tensor(limits)

    # Select a few random indices for sampling.
    periodic_indices = torch.sort(torch.randperm(n_features_in)[:n_periodic]).values
    lifter = PeriodicEmbedding(n_features_in=n_features_in, periodic_indices=periodic_indices, limits=limits)

    # Create random input with the correct periodicity.
    x = create_input(batch_size, n_features_in, limits=limits, periodic_indices=periodic_indices)

    # Lift periodic DOFs.
    x_out = lifter(x)

    # The lifted input must have n_periodic more elements.
    assert x.shape[1] + n_periodic == x_out.shape[1]

    # The lifter leaves unaltered the non-periodic DOFs and duplicate the periodic ones.
    shift_idx = 0
    for i in range(n_features_in):
        if i in periodic_indices:
            shift_idx += 1
        else:
            assert torch.all(x[:, i] == x_out[:, i+shift_idx])
    assert shift_idx == n_periodic

    # The limits are mapped to the same values (cos=1, sin=0).
    x[0, periodic_indices[0]] = limits[0]
    x[0, periodic_indices[1]] = limits[1]
    x_out = lifter(x)

    expected = torch.tensor([1.0, 0.0])
    assert torch.allclose(x_out[0, periodic_indices[0]:periodic_indices[0]+2], expected)
    # The "+ 1" here is because of the shift idx due to periodic_indices[0].
    assert torch.allclose(x_out[0, periodic_indices[1]+1:periodic_indices[1]+3], expected)


# =============================================================================
# TEST FLIP-INVARIANT EMBEDDING
# =============================================================================

@pytest.mark.parametrize('n_features_in,embedding_dimension,embedded_indices,degrees_in,expected_degrees_out', [
    (4, 4, range(4), [0, 0, 0, 0], [0, 0, 0, 0]),
    (8, 4, None, [0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1]),
    (8, 4, range(8), [1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0]),
    (4, 3, range(4), [0, 0, 0, 0], [0, 0, 0]),
    (8, 3, range(8), [0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1]),
    (8, 3, range(8), [1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0]),
    (9, 2, [2, 3, 4, 5], [3, 1, 2, 2, 2, 2, 0, 5, 4], [3, 1, 2, 2, 0, 5, 4]),
    (10, 2, [1, 2, 3, 4, 6, 7, 8, 9], [1, 3, 3, 3, 3, 2, 0, 0, 0, 0], [1, 3, 3, 2, 0, 0]),
    (10, 2, [0, 1, 2, 3, 5, 6, 7, 8], [2, 2, 2, 2, 3, 0, 0, 0, 0, 1], [2, 2, 3, 0, 0, 1]),
])
def test_flip_invariant_embedding_get_degrees_out(
        n_features_in,
        embedding_dimension,
        embedded_indices,
        degrees_in,
        expected_degrees_out,
):
    """Test FlipInvariantEmbedding.get_degrees_out."""
    # Create embedding layer.
    embedding = FlipInvariantEmbedding(
        n_features_in=n_features_in,
        embedding_dimension=embedding_dimension,
        embedded_indices=embedded_indices,
    )

    # Check degrees out.
    degrees_out = embedding.get_degrees_out(torch.tensor(degrees_in))
    assert torch.all(degrees_out == torch.tensor(expected_degrees_out))


@pytest.mark.parametrize('batch_size', [1, 3])
@pytest.mark.parametrize('n_features_in,embedding_dimension,embedded_indices_in', [
    (4, 4, range(4)),
    (12, 4, range(12)),
    (4, 3, range(4)),
    (12, 3, range(12)),
    (9, 3, [2, 3, 4, 5]),
    (10, 2, [1, 2, 3, 4, 6, 7, 8, 9]),
    (11, 2, [1, 2, 3, 4, 6, 7, 8, 9]),
])
def test_flip_invariant_embedding_invariance(
        batch_size,
        n_features_in,
        embedding_dimension,
        embedded_indices_in,
):
    """Test FlipInvariantEmbedding is flip invariant."""
    embedded_indices_in = torch.tensor(embedded_indices_in)

    # Create embedding layer.
    embedding = FlipInvariantEmbedding(
        n_features_in=n_features_in,
        embedding_dimension=embedding_dimension,
        embedded_indices=embedded_indices_in,
    )

    # Create random input features and embed it.
    x = torch.randn(batch_size, n_features_in)
    out = embedding(x)

    # The output has the correct dimension.
    n_vectors = len(embedded_indices_in) // embedding.vector_dimension
    n_features_diff = n_vectors * (embedding_dimension - embedding.vector_dimension)
    assert out.shape == (batch_size, n_features_in + n_features_diff)

    # Now flip the sign of all features.
    out_flipped = embedding(-x)

    # Find the output indices through get_degrees_out rather than private variables.
    degrees_in = torch.zeros(n_features_in)
    degrees_in[embedded_indices_in] = 1
    degrees_out = embedding.get_degrees_out(degrees_in)
    embedded_mask = degrees_out == 1
    nonembedded_mask = degrees_out == 0

    # The embedded vectors are flip invariant.
    assert torch.all(out[:, embedded_mask] == out_flipped[:, embedded_mask])

    # The non-embedded features have flipped.
    assert torch.all(out[:, nonembedded_mask] == -out_flipped[:, nonembedded_mask])


def test_flip_invariant_embedding_error_degrees_in():
    """An error is raised if there are embedded vectors whose components are assigned different input degrees"""
    embedding = FlipInvariantEmbedding(
        n_features_in=4,
        embedding_dimension=3,
    )
    with pytest.raises(ValueError, match='same degree must be assigned'):
        embedding.get_degrees_out(torch.tensor([0, 0, 0, 1]))
