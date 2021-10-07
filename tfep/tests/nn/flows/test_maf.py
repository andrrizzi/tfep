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

import numpy as np
import pytest
import torch

from tfep.nn.utils import generate_block_sizes
from tfep.nn.transformers.affine import AffineTransformer
from tfep.nn.transformers.sos import SOSPolynomialTransformer
from tfep.nn.transformers.spline import NeuralSplineTransformer
from tfep.nn.transformers.mobius import MobiusTransformer
from tfep.nn.flows.maf import MAF
from ..utils import create_random_input


# =============================================================================
# TESTS
# =============================================================================

@pytest.mark.parametrize('dimensions_hidden', [1, 4])
@pytest.mark.parametrize('dimension_conditioning', [0, 2])
@pytest.mark.parametrize('degrees_in', ['input', 'reversed'])
@pytest.mark.parametrize('weight_norm', [False, True])
@pytest.mark.parametrize('split_conditioner', [True, False])
@pytest.mark.parametrize('transformer', [
    AffineTransformer(),
    SOSPolynomialTransformer(2),
    SOSPolynomialTransformer(3),
    NeuralSplineTransformer(x0=torch.tensor(-2), xf=torch.tensor(2), n_bins=3),
    MobiusTransformer(blocks=3, shorten_last_block=True)
])
def test_identity_initialization_MAF(dimensions_hidden, dimension_conditioning, degrees_in,
                                     weight_norm, split_conditioner, transformer):
    """Test that the identity initialization of MAF works.

    This tests that the flow layers can be initialized to perform the
    identity function.

    """
    dimension = 5
    batch_size = 2

    # We don't initialize as the identity function to make the test meaningful.
    maf = MAF(
        dimension,
        dimensions_hidden,
        dimension_conditioning=dimension_conditioning,
        degrees_in=degrees_in,
        weight_norm=weight_norm,
        split_conditioner=split_conditioner,
        transformer=transformer,
        initialize_identity=True
    )

    # Create random input.
    if isinstance(transformer, NeuralSplineTransformer):
        x = create_random_input(batch_size, dimension, x_func=torch.rand)
        x = x * (transformer.xf - transformer.x0) + transformer.x0
    else:
        x = create_random_input(batch_size, dimension)

    y, log_det_J = maf.forward(x)

    assert torch.allclose(x, y)
    assert torch.allclose(log_det_J, torch.zeros(batch_size), atol=1e-6)


@pytest.mark.parametrize('dimension_conditioning', [0, 2])
@pytest.mark.parametrize('degrees_in', ['input', 'reversed', 'random'])
@pytest.mark.parametrize('split_conditioner', [True, False])
@pytest.mark.parametrize('transformer', [
    AffineTransformer(),
    MobiusTransformer(blocks=3, shorten_last_block=True)
])
@pytest.mark.parametrize('weight_norm', [False, True])
def test_round_trip_MAF(dimension_conditioning, degrees_in, weight_norm, split_conditioner, transformer):
    """Test that the MAF.inverse(MAF.forward(x)) equals the identity."""
    dimension = 5
    dimensions_hidden = 2
    batch_size = 2

    # Temporarily set default precision to double to improve comparisons.
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.double)

    # With the Mobius transformer, we need block dependencies.
    if isinstance(transformer, MobiusTransformer):
        blocks = generate_block_sizes(dimension-dimension_conditioning, transformer.blocks,
                                      transformer.shorten_last_block)
        shorten_last_block = transformer.shorten_last_block
        n_blocks = len(blocks)
    else:
        blocks = 1
        shorten_last_block = False
        n_blocks = dimension - dimension_conditioning

    # Make sure the permutation is reproducible.
    if degrees_in == 'random':
        random_state = np.random.RandomState(0)
        degrees_in = random_state.permutation(range(n_blocks))

    # We don't initialize as the identity function to make the test meaningful.
    maf = MAF(
        dimension, dimensions_hidden,
        dimension_conditioning=dimension_conditioning,
        degrees_in=degrees_in,
        weight_norm=weight_norm,
        blocks=blocks,
        shorten_last_block=shorten_last_block,
        split_conditioner=split_conditioner,
        transformer=transformer,
        initialize_identity=False
    )

    # Create random input.
    x = create_random_input(batch_size, dimension)

    # The conditioning features are always left unchanged.
    y, log_det_J = maf.forward(x)
    assert torch.allclose(x[:, :dimension_conditioning], y[:, :dimension_conditioning])

    # Inverting the transformation produces the input vector.
    x_inv, log_det_J_inv = maf.inverse(y)
    assert torch.allclose(x, x_inv)
    assert torch.allclose(log_det_J + log_det_J_inv, torch.zeros(batch_size), atol=1e-04)

    # Restore default dtype.
    torch.set_default_dtype(old_dtype)
