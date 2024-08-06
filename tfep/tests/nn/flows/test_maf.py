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
from tfep.nn.flows.maf import MAF, _LiftPeriodic
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

@pytest.mark.parametrize('n_periodic', [2, 3])
@pytest.mark.parametrize('limits', [
    (0., 1.),
    (-1., 1.),
    (0., 2*torch.pi),
    (-torch.pi, torch.pi),
    (-5., -2.5),
])
def test_lift_periodic(n_periodic, limits):
    """Test that _LiftPeriodic lifts the correct degrees of freedom."""
    batch_size = 3
    dimension_in = 5
    limits = torch.tensor(limits)

    # Select a few random indices for sampling.
    periodic_indices = torch.sort(torch.randperm(dimension_in)[:n_periodic]).values
    lifter = _LiftPeriodic(dimension_in=dimension_in, periodic_indices=periodic_indices, limits=limits)

    # Create random input with the correct periodicity.
    x = create_input(batch_size, dimension_in, limits=limits, periodic_indices=periodic_indices)

    # Lift periodic DOFs.
    x_lifted = lifter(x)

    # The lifted input must have n_periodic more elements.
    assert x.shape[1] + n_periodic == x_lifted.shape[1]

    # The lifter leaves unaltered the non-periodic DOFs and duplicate the periodic ones.
    shift_idx = 0
    for i in range(dimension_in):
        if i in periodic_indices:
            shift_idx += 1
        else:
            assert torch.all(x[:, i] == x_lifted[:, i+shift_idx])
    assert shift_idx == n_periodic

    # The limits are mapped to the same values (cos=1, sin=0).
    x[0, periodic_indices[0]] = limits[0]
    x[0, periodic_indices[1]] = limits[1]
    x_lifted = lifter(x)

    expected = torch.tensor([1.0, 0.0])
    assert torch.allclose(x_lifted[0, periodic_indices[0]:periodic_indices[0]+2], expected)
    # The "+ 1" here is because of the shift idx due to periodic_indices[0].
    assert torch.allclose(x_lifted[0, periodic_indices[1]+1:periodic_indices[1]+3], expected)


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
    if periodic_indices is not None and isinstance(transformer, MoebiusTransformer):
        pytest.skip('Vector inputs for Moebius transformers are not compatible with periodic.')

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
        periodic_indices=periodic_indices,
        periodic_limits=limits,
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
def test_maf_round_trip(conditioning_indices, periodic_indices, degrees_in_order, weight_norm, transformer):
    """Test the autoregressive property and that the MAF.inverse(MAF.forward(x)) equals the identity."""
    n_features = 5
    batch_size = 2
    limits = (0., 2.)

    # Periodic indices with MoebiusTransformer doens't make sense.
    if periodic_indices is not None and isinstance(transformer, MoebiusTransformer):
        pytest.skip('Vector inputs for Moebius transformers are not compatible with periodic.')

    # With the MoebiusTransformer, the output must be vectors of the same size.
    if isinstance(transformer, MoebiusTransformer):
        extra_dim = transformer.dimension - (n_features - len(conditioning_indices)) % transformer.dimension
        n_features = n_features + extra_dim
        repeats = transformer.dimension
    else:
        repeats = 1

    # We don't initialize as the identity function to make the test meaningful.
    maf = MAF(
        degrees_in=generate_degrees(
            n_features=n_features,
            order=degrees_in_order,
            conditioning_indices=conditioning_indices,
            repeats=repeats,
        ),
        transformer=transformer,
        hidden_layers=2,
        periodic_indices=periodic_indices,
        periodic_limits=limits,
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
