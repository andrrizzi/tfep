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
    NeuralSplineTransformer, MoebiusTransformer
)
from tfep.nn.conditioners.made import generate_degrees
from tfep.nn.embeddings.mafembed import PeriodicEmbedding
from tfep.nn.flows.maf import MAF

from .. import check_autoregressive_property
from ..utils import create_random_input


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

def create_input(batch_size, dimension_in, limits=None, periodic_indices=None, seed=0):
    """Create random input with correct boundaries.

    If limits=(x0, xf) are passed, the input is constrained to be within x0 and
    xf. x0 and xf can be both floats or Tensors of size (n_features,).

    If periodic_indices is passed, only these DOFs are restrained between x0
    and xf.

    """
    # The non-periodic features do not need to be restrained between 0 and 1.
    x = create_random_input(batch_size, dimension_in, x_func=torch.randn, seed=seed)

    if limits is not None:
        if periodic_indices is None:
            # All DOFs are periodic. Ignore previous random input.
            periodic_indices = torch.tensor(list(range(dimension_in)))

        x_periodic = create_random_input(batch_size, len(periodic_indices), x_func=torch.rand, seed=seed)
        x_periodic = x_periodic * (limits[1] - limits[0]) + limits[0]
        x = x.clone()
        x[:, periodic_indices] = x_periodic

    return x


# =============================================================================
# TESTS
# =============================================================================

@pytest.mark.parametrize('hidden_layers', [1, 4])
@pytest.mark.parametrize('conditioning_indices', [
    [],
    [0, 1],
    [0, 3],
    [1, 3],
    [0, 4],
    [2, 4],
    [3, 4]
])
@pytest.mark.parametrize('periodic_indices', [
    None,
    [0],
    [1],
    [1, 2],
    [0, 2],
])
@pytest.mark.parametrize('degrees_in_order', ['ascending', 'descending'])
@pytest.mark.parametrize('weight_norm', [False, True])
@pytest.mark.parametrize('transformer', [
    AffineTransformer(),
    SOSPolynomialTransformer(2),
    SOSPolynomialTransformer(3),
    NeuralSplineTransformer(x0=torch.tensor(-2., dtype=torch.double), xf=torch.tensor(2., dtype=torch.double), n_bins=3),
    MoebiusTransformer(dimension=3)
])
def test_identity_initialization_MAF(hidden_layers, conditioning_indices, periodic_indices,
                                     degrees_in_order, weight_norm, transformer):
    """Test that the identity initialization of MAF works.

    This tests that the flow layers can be initialized to perform the
    identity function.

    """
    n_features = 5
    batch_size = 2
    # Must be equal to the NeuralSplineTransformer limits.
    limits = [-2., 2.]

    # Periodic indices with MoebiusTransformer doens't make sense.
    if periodic_indices is None:
        embedding = None
    elif isinstance(transformer, MoebiusTransformer):
        pytest.skip('Vector inputs for Moebius transformers are not compatible with periodic.')
    else:
        embedding = PeriodicEmbedding(
            n_features_in=n_features,
            periodic_indices=periodic_indices,
            limits=limits,
        )

    # With the MoebiusTransformer, the output must be vectors of the same size.
    if isinstance(transformer, MoebiusTransformer):
        extra_dim = transformer.dimension - (n_features - len(conditioning_indices)) % transformer.dimension
        n_features = n_features + extra_dim
        repeats = transformer.dimension
    else:
        repeats = 1

    # Create MAF.
    maf = MAF(
        degrees_in=generate_degrees(
            n_features=n_features,
            order=degrees_in_order,
            conditioning_indices=conditioning_indices,
            repeats=repeats,
        ),
        transformer=transformer,
        hidden_layers=hidden_layers,
        embedding=embedding,
        weight_norm=weight_norm,
        initialize_identity=True,
    )

    # Create random input.
    if isinstance(transformer, NeuralSplineTransformer):
        x = create_input(batch_size, n_features, limits=(transformer.x0, transformer.xf))
    else:
        x = create_input(batch_size, n_features)

    y, log_det_J = maf.forward(x)

    assert torch.allclose(x, y)
    assert torch.allclose(log_det_J, torch.zeros(batch_size), atol=1e-6)


@pytest.mark.parametrize('conditioning_indices', [
    [],
    [0, 1],
    [0, 3],
    [1, 3],
    [0, 4],
    [2, 4],
    [3, 4]
])
@pytest.mark.parametrize('periodic_indices', [
    None,
    [0],
    [1],
    [1, 2],
    [0, 2],
])
@pytest.mark.parametrize('degrees_in_order', ['ascending', 'descending', 'random'])
@pytest.mark.parametrize('transformer', [
    AffineTransformer(),
    MoebiusTransformer(dimension=3)
])
@pytest.mark.parametrize('weight_norm', [False, True])
def test_maf_autoregressive_round_trip(conditioning_indices, periodic_indices, degrees_in_order, weight_norm, transformer):
    """Test the autoregressive property and that the MAF.inverse(MAF.forward(x)) equals the identity."""
    n_features = 5
    batch_size = 2
    limits = (0., 2.)

    # Periodic indices with MoebiusTransformer doens't make sense.
    if periodic_indices is None:
        embedding = None
    elif isinstance(transformer, MoebiusTransformer):
        pytest.skip('Vector inputs for Moebius transformers are not compatible with periodic.')
    else:
        embedding = PeriodicEmbedding(
            n_features_in=n_features,
            periodic_indices=periodic_indices,
            limits=limits,
        )

    # With the MoebiusTransformer, the output must be vectors of the same size.
    if isinstance(transformer, MoebiusTransformer):
        extra_dim = transformer.dimension - (n_features - len(conditioning_indices)) % transformer.dimension
        n_features = n_features + extra_dim
        repeats = transformer.dimension
    else:
        repeats = 1

    # Input degrees.
    degrees_in = generate_degrees(
        n_features=n_features,
        order=degrees_in_order,
        conditioning_indices=conditioning_indices,
        repeats=repeats,
    )

    # We don't initialize as the identity function to make the test meaningful.
    maf = MAF(
        degrees_in=degrees_in,
        transformer=transformer,
        hidden_layers=2,
        embedding=embedding,
        weight_norm=weight_norm,
        initialize_identity=False,
    )

    # Create random input.
    x = create_input(batch_size, n_features, limits=limits, periodic_indices=periodic_indices)

    # The conditioning features are always left unchanged.
    y, log_det_J = maf.forward(x)
    assert torch.allclose(x[:, conditioning_indices], y[:, conditioning_indices])

    # Inverting the transformation produces the input vector.
    x_inv, log_det_J_inv = maf.inverse(y)
    assert torch.allclose(x, x_inv)
    assert torch.allclose(log_det_J + log_det_J_inv, torch.zeros(batch_size), atol=1e-04)

    # Test the autoregressive property.
    check_autoregressive_property(
        model=maf,
        x=x[0],
        degrees_in=degrees_in,
        # The transformed features depend on themselves through the transformer.
        # We need to check that they don't affect the features with greater degree.
        degrees_out=degrees_in+1,
    )
