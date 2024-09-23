#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test objects and function in tfep.nn.transformers.mixed.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import pytest
import torch

from tfep.nn.transformers.affine import AffineTransformer
from tfep.nn.transformers.mixed import MixedTransformer
from tfep.nn.transformers.moebius import MoebiusTransformer
from tfep.nn.transformers.spline import NeuralSplineTransformer

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
# SHARED VARIABLES FOR PARAMETRIZED TESTS
# =============================================================================

BATCH_SIZES_TESTED = [1, 5]

# Disible by 2 or 3 (i.e., the number of transformers).
N_FEATURES = 6

# Neural splines assume that the 6 features are divided equally among transformers.
TRANSFORMERS_TESTED = [
    [
        AffineTransformer(),
        NeuralSplineTransformer(
            x0=torch.zeros(3, dtype=torch.double),
            xf=torch.ones(3, dtype=torch.double),
            n_bins=3,
        ),
    ], [
        MoebiusTransformer(dimension=1),
        AffineTransformer(),
        NeuralSplineTransformer(
            x0=torch.zeros(2, dtype=torch.double),
            xf=torch.ones(2, dtype=torch.double),
            n_bins=4,
        ),
    ], [
        NeuralSplineTransformer(
            x0=torch.zeros(3, dtype=torch.double),
            xf=torch.ones(3, dtype=torch.double),
            n_bins=2,
        ),
        NeuralSplineTransformer(
            x0=torch.zeros(3, dtype=torch.double),
            xf=torch.ones(3, dtype=torch.double),
            n_bins=3,
            circular=True,
        ),
    ]
]


# =============================================================================
# UTILS
# =============================================================================

def create_random_input(batch_size, transformers):
    """Random input for MixedTransformer tests."""
    # Create random indices for each transformer.
    indices = torch.randperm(N_FEATURES).reshape(len(transformers), -1)

    # Create the mixed transformer.
    mixed = MixedTransformer(transformers=transformers, indices=indices)

    # Create random input and parameters for each transformer.
    # We use rand because the neural splines have limits between 0 and 1.
    x = torch.rand(batch_size, N_FEATURES)
    n_pars = mixed.get_identity_parameters(N_FEATURES).shape
    parameters = torch.randn(batch_size, *n_pars)

    return mixed, x, parameters


# =============================================================================
# TESTS
# =============================================================================

@pytest.mark.parametrize('batch_size', BATCH_SIZES_TESTED)
@pytest.mark.parametrize('transformers', TRANSFORMERS_TESTED)
def test_mixed_transformer(batch_size, transformers):
    """MixedTransformer applies the transformers to the correct DOFs."""
    n_feats_per_transformer = N_FEATURES // len(transformers)

    # Create random indices for each transformer.
    indices = torch.randperm(N_FEATURES).reshape(
        len(transformers), n_feats_per_transformer)

    # Create the mixed transformer.
    mixed = MixedTransformer(transformers=transformers, indices=indices)

    # Create random input and parameters for each transformer.
    x = torch.rand(batch_size, N_FEATURES, requires_grad=True)
    parameters = [transformer.get_identity_parameters(n_feats_per_transformer)
                  for transformer in transformers]
    parameters = [torch.randn(batch_size, *par.shape) for par in parameters]

    # Concatenate them in the parameters for the mixed transformer.
    mixed_parameters = torch.cat(parameters, dim=1)

    # Run the mixedtransformer.
    y_mixed, log_det_J_mixed = mixed(x, mixed_parameters)

    # Compare result to running the transformers separately.
    for transformer, idx, par in zip(transformers, indices, parameters):
        y, log_det_J = transformer(x[:, idx], par)
        assert torch.allclose(y_mixed[:, idx], y)

    # Compare log_det_J against autograd result.
    ref_log_det_J = batch_autograd_log_abs_det_J(x, y_mixed)
    assert torch.allclose(log_det_J_mixed, ref_log_det_J)


@pytest.mark.parametrize('batch_size', BATCH_SIZES_TESTED)
@pytest.mark.parametrize('transformers', TRANSFORMERS_TESTED)
def test_mixed_transformer_round_trip(batch_size, transformers):
    """A round trip inverse(forward()) equals the identity function."""
    mixed, x, parameters = create_random_input(batch_size, transformers)
    y, log_det_J = mixed(x, parameters)
    x_inv, log_det_J_inv = mixed.inverse(y, parameters)
    assert torch.allclose(x, x_inv)
    assert torch.allclose(log_det_J + log_det_J_inv, torch.zeros_like(log_det_J))


@pytest.mark.parametrize('batch_size', BATCH_SIZES_TESTED)
@pytest.mark.parametrize('transformers', TRANSFORMERS_TESTED)
def test_mixed_transformer_get_identity_parameters(batch_size, transformers):
    """get_identity_parameters returns the parameters of the identity function."""
    torch.manual_seed(0)
    mixed, x, parameters = create_random_input(batch_size, transformers)
    parameters = mixed.get_identity_parameters(N_FEATURES).unsqueeze(0).expand(batch_size, -1)
    y, log_det_J = mixed(x, parameters)
    assert torch.allclose(x, y)
    assert torch.allclose(log_det_J, torch.zeros_like(log_det_J), atol=1e-7)


@pytest.mark.parametrize('batch_size', BATCH_SIZES_TESTED)
@pytest.mark.parametrize('transformers', TRANSFORMERS_TESTED)
def test_mixed_transformer_get_degrees_out(batch_size, transformers):
    """get_degrees_out returns the correct degrees."""
    mixed, x, parameters = create_random_input(batch_size, transformers)

    # Create random degrees for the input.
    degrees_in = torch.randperm(N_FEATURES)

    # Get the degrees for the output for the mixed and separate transformers.
    degrees_out = mixed.get_degrees_out(degrees_in)
    transformers_degrees_out = [
        transformer.get_degrees_out(degrees_in[mixed._indices[idx]])
        for idx, transformer in enumerate(transformers)
    ]

    # The MixedTransformer expects parameters ordered by transformer.
    current_idx = 0
    for degrees in transformers_degrees_out:
        assert torch.allclose(degrees, degrees_out[current_idx:current_idx+len(degrees)])
        current_idx += len(degrees)


def test_mixed_transformer_error_one_transformer():
    """An error is raised if only a single transformer is passed."""
    with pytest.raises(ValueError, match='transformers must be greater than 1'):
        MixedTransformer(transformers=[AffineTransformer()], indices=[range(6)])


def test_mixed_transformer_error_different_lengths():
    """An error is raised if the lengths of the transformer and indices are different."""
    with pytest.raises(ValueError, match='indices must equal that in transformers'):
        MixedTransformer(transformers=[AffineTransformer()]*3, indices=[range(2)]*2)
