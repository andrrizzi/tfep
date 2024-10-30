#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test MAF layer in tfep.nn.flows.maf.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import pytest
import torch

from tfep.nn.transformers import (
    AffineTransformer, SOSPolynomialTransformer,
    NeuralSplineTransformer, MoebiusTransformer,
    MixedTransformer,
)
from tfep.nn.conditioners.made import generate_degrees
from tfep.nn.embeddings.mafembed import (
    PeriodicEmbedding,
    FlipInvariantEmbedding,
    MixedEmbedding,
)
from tfep.nn.flows.maf import MAF
from tfep.utils.misc import remove_and_shift_sorted_indices

from .. import check_autoregressive_property
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

def generate_test_case(
        conditioning_indices,
        periodic_indices,
        flip_invariant_indices,
        transformer,
        degrees_in_order,
        weight_norm,
        hidden_layers,
        initialize_identity,
):
    """Create a new test case."""
    batch_size = 2
    limits = torch.tensor([-1., 1.])
    embed_dim = 1
    vector_dim = 2

    # We increase the number of features by the number of conditioning indices
    # to make sure that the MoebiusTransformer receives a number divisible by
    # its vector dimension=2.
    n_conditioning_indices = 0 if conditioning_indices is None else len(conditioning_indices)
    n_features = 6 + n_conditioning_indices

    # Create embedding.
    if (periodic_indices is None) and (flip_invariant_indices is None):
        embedding = None
    elif (periodic_indices is not None) and (flip_invariant_indices is not None):
        embedding = MixedEmbedding(
            n_features_in=n_features,
            embedding_layers=[
                PeriodicEmbedding(
                    n_features_in=len(periodic_indices),
                    limits=limits,
                ),
                FlipInvariantEmbedding(
                    n_features_in=len(flip_invariant_indices),
                    embedding_dimension=embed_dim,
                    vector_dimension=vector_dim,
                )
            ],
            embedded_indices=[periodic_indices, flip_invariant_indices]
        )
    elif periodic_indices is not None:
        embedding = PeriodicEmbedding(
            n_features_in=n_features,
            periodic_indices=periodic_indices,
            limits=limits,
        )
    else:
        embedding = FlipInvariantEmbedding(
            n_features_in=n_features,
            embedding_dimension=embed_dim,
            embedded_indices=flip_invariant_indices,
            vector_dimension=vector_dim,
        )

    # Generate input degrees.
    if isinstance(transformer, MoebiusTransformer):
        repeats = transformer.dimension
    elif flip_invariant_indices is not None:
        repeats = torch.ones(n_features, dtype=int)
        repeats[flip_invariant_indices] = 2

        # Repeats does not have conditioning indices and must have only 1
        # entry for each flip-invariant vector.
        removed_indices = torch.tensor(flip_invariant_indices[1::vector_dim])
        if conditioning_indices is not None:
            removed_indices = torch.cat([torch.tensor(conditioning_indices), removed_indices]).sort().values
        repeats = repeats[remove_and_shift_sorted_indices(torch.arange(n_features), removed_indices, shift=False)]
    else:
        repeats = 1
    degrees_in = generate_degrees(
        n_features=n_features,
        order=degrees_in_order,
        conditioning_indices=conditioning_indices,
        repeats=repeats,
    )

    # Create MAF.
    maf = MAF(
        degrees_in=degrees_in,
        transformer=transformer,
        hidden_layers=hidden_layers,
        embedding=embedding,
        weight_norm=weight_norm,
        initialize_identity=initialize_identity,
    )

    # Create random input. We set the limit of periodic indices so that the
    # MoebiusTransformer cannot move things outside them.
    x = create_random_input(batch_size, n_features, x_func=torch.randn, seed=0)
    if periodic_indices is not None:
        limits = limits / torch.linalg.norm(torch.ones(vector_dim))
        x_periodic = create_random_input(batch_size, len(periodic_indices), x_func=torch.rand, seed=0)
        x_periodic = x_periodic * (limits[1] - limits[0]) + limits[0]
        x = x.clone()
        x[:, periodic_indices] = x_periodic

    return x, maf, degrees_in


# =============================================================================
# TESTS
# =============================================================================

@pytest.mark.parametrize('conditioning_indices,periodic_indices,flip_invariant_indices', [
    (None, None, None),
    ([0, 1], None, None),
    ([2, 7], None, None),
    (None, [2], None),
    (None, [0, 5], None),
    (None, None, [2, 3]),
    (None, None, [0, 1, 4, 5]),
    ([1], [0], [3, 4]),
    ([4], [2, 3], [0, 1, 5, 6]),
    ([0, 7], [3], [5, 6]),
    ([2, 4], [4], [6, 7]),
    ([0, 3, 4], [6], [3, 4]),
])
@pytest.mark.parametrize('transformer', [
    AffineTransformer(),
    SOSPolynomialTransformer(2),
    SOSPolynomialTransformer(3),
    NeuralSplineTransformer(
        x0=torch.tensor(-1., dtype=torch.double),
        xf=torch.tensor(1., dtype=torch.double),
        n_bins=3
    ),
    MoebiusTransformer(dimension=2),
    MixedTransformer(
        transformers=[AffineTransformer(), SOSPolynomialTransformer(3)],
        indices=[[0, 2, 5], [1, 3, 4]],
    ),
])
@pytest.mark.parametrize('degrees_in_order', ['ascending', 'descending'])
@pytest.mark.parametrize('weight_norm', [False, True])
@pytest.mark.parametrize('hidden_layers', [1, 4])
def test_maf_identity_initialization(
        conditioning_indices,
        periodic_indices,
        flip_invariant_indices,
        transformer,
        degrees_in_order,
        weight_norm,
        hidden_layers,
):
    """Test that the identity initialization of MAF works.

    This tests that the flow layers can be initialized to perform the
    identity function.

    """
    x, maf, degrees_in = generate_test_case(
        conditioning_indices=conditioning_indices,
        periodic_indices=periodic_indices,
        flip_invariant_indices=flip_invariant_indices,
        transformer=transformer,
        degrees_in_order=degrees_in_order,
        weight_norm=weight_norm,
        hidden_layers=hidden_layers,
        initialize_identity=True,
    )

    # Test identity.
    y, log_det_J = maf.forward(x)
    assert torch.allclose(x, y)
    assert torch.allclose(log_det_J, torch.zeros_like(log_det_J), atol=1e-6)


@pytest.mark.parametrize('conditioning_indices,periodic_indices,flip_invariant_indices', [
    (None, None, None),
    ([0, 1], None, None),
    ([2, 7], None, None),
    (None, [2], None),
    (None, [0, 5], None),
    (None, None, [2, 3]),
    (None, None, [0, 1, 4, 5]),
    ([1], [0], [3, 4]),
    ([4], [2, 3], [0, 1, 5, 6]),
    ([0, 7], [3], [5, 6]),
    ([2, 4], [4], [6, 7]),
    ([0, 3, 4], [6], [3, 4]),
])
@pytest.mark.parametrize('degrees_in_order', ['ascending', 'descending', 'random'])
@pytest.mark.parametrize('transformer', [
    AffineTransformer(),
    MoebiusTransformer(dimension=2),
    MixedTransformer(
        transformers=[
            AffineTransformer(),
            NeuralSplineTransformer(
                x0=torch.tensor(-1., dtype=torch.double),
                xf=torch.tensor(1., dtype=torch.double),
                n_bins=3,
            ),
        ],
        indices=[[1, 4], [0, 2, 3, 5]],
    )
])
@pytest.mark.parametrize('weight_norm', [False, True])
def test_maf_autoregressive_round_trip(
        conditioning_indices,
        periodic_indices,
        flip_invariant_indices,
        degrees_in_order,
        weight_norm,
        transformer,
):
    """Test the autoregressive property and that the MAF.inverse(MAF.forward(x)) equals the identity."""
    x, maf, degrees_in = generate_test_case(
        conditioning_indices=conditioning_indices,
        periodic_indices=periodic_indices,
        flip_invariant_indices=flip_invariant_indices,
        transformer=transformer,
        degrees_in_order=degrees_in_order,
        weight_norm=weight_norm,
        hidden_layers=2,
        initialize_identity=False,
    )

    # The conditioning features are always left unchanged.
    y, log_det_J = maf.forward(x)
    if conditioning_indices is not None:
        assert torch.allclose(x[:, conditioning_indices], y[:, conditioning_indices])

    # Inverting the transformation produces the input vector.
    x_inv, log_det_J_inv = maf.inverse(y)
    assert torch.allclose(x, x_inv)
    assert torch.allclose(log_det_J + log_det_J_inv, torch.zeros_like(log_det_J), atol=1e-04)

    # Test the autoregressive property.
    check_autoregressive_property(
        model=maf,
        x=x[0],
        degrees_in=degrees_in,
        # The transformed features depend on themselves through the transformer.
        # We need to check that they don't affect the features with greater degree.
        degrees_out=degrees_in+1,
    )
