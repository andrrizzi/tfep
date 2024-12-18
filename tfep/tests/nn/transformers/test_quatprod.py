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

import pytest
import torch

from tfep.nn.transformers.quatprod import QuaternionProductTransformer


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
# TESTS
# =============================================================================

@pytest.mark.parametrize('batch_size', [1, 3])
@pytest.mark.parametrize('n_quaternions', [1, 3])
def test_quat_product_transformer_identity(batch_size, n_quaternions):
    """QuaternionProductTransformer.get_identity_parameters returns the parameters for the identity."""
    # roma is an optional dependency at the moment
    roma = pytest.importorskip('roma')

    transformer = QuaternionProductTransformer()
    w = transformer.get_identity_parameters(n_quaternions*4).expand(batch_size, -1)
    q = roma.random_unitquat(batch_size*n_quaternions).reshape(batch_size, -1)
    q2, log_det_J = transformer(q, w)
    assert torch.allclose(q, q2)
    assert torch.allclose(log_det_J, torch.zeros_like(log_det_J))


@pytest.mark.parametrize('batch_size', [1, 3])
@pytest.mark.parametrize('n_quaternions', [1, 3])
def test_quat_product_transformer_round_trip(batch_size, n_quaternions):
    """Check that a forward-inverse round-trip with QuaternionProductTransformer is the identity."""
    # roma is an optional dependency at the moment
    roma = pytest.importorskip('roma')

    transformer = QuaternionProductTransformer()
    w = torch.randn(batch_size, n_quaternions*4)
    q = roma.random_unitquat(batch_size*n_quaternions).reshape(batch_size, -1).to(w)
    q2, log_det_J = transformer(q, w)
    q_inv, log_det_J_inv = transformer.inverse(q2, w)

    assert torch.allclose(q, q_inv)
    assert torch.allclose(log_det_J, torch.zeros_like(log_det_J))
    assert torch.allclose(log_det_J, log_det_J_inv)


@pytest.mark.parametrize('batch_size', [1, 3])
@pytest.mark.parametrize('n_quaternions', [1, 3])
def test_quat_product_transformer_flip_equivariance(batch_size, n_quaternions):
    """The QuaternionProductTransformer is flip equivariant w.r.t. its input."""
    # roma is an optional dependency at the moment
    roma = pytest.importorskip('roma')

    transformer = QuaternionProductTransformer()
    w = torch.randn(batch_size, n_quaternions*4)
    q = roma.random_unitquat(batch_size*n_quaternions).reshape(batch_size, -1).to(w)

    # Compare the transformation with the input and its negative.
    q2, log_det_J = transformer(q, w)
    q2_neg, log_det_J_neg = transformer(-q, w)
    assert torch.allclose(q2, -q2_neg)
