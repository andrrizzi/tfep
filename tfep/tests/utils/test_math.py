#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test objects and function in tfep.utils.math.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import numpy as np
import pytest
import torch

from tfep.utils.math import cov, batch_autograd_jacobian


# =============================================================================
# GLOBAL VARIABLES
# =============================================================================

# Makes random input deterministic.
_GENERATOR = torch.Generator()
_GENERATOR.manual_seed(0)


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

@pytest.mark.parametrize('ddof', [0, 1])
@pytest.mark.parametrize('dim_sample', [0, 1])
def test_cov(ddof, dim_sample):
    """Test the covariance matrix against the numpy implementation."""
    random_state = np.random.RandomState(0)
    x = random_state.randn(10, 15)

    if dim_sample == 0:
        cov_np = np.cov(x.T, ddof=ddof)
    else:
        cov_np = np.cov(x, ddof=ddof)

    cov_torch = cov(torch.tensor(x), dim_sample=dim_sample, ddof=ddof, inplace=True).numpy()

    assert np.allclose(cov_np, cov_torch)

@pytest.mark.parametrize('in_shape,out_shape', [
    [(), ()],
    [(3,), (4,)],
    [(2, 4), (2, 5)],
])
def test_batch_autograd_jacobian(in_shape, out_shape):
    """Test the batch_autograd_jacobian function."""
    batch_size = 2

    # Create random input.
    x = torch.randn(batch_size, *in_shape, requires_grad=True)

    # Transformation with a linear layer.
    in_features = 1 if len(in_shape) == 0 else in_shape[-1]
    out_features = 1 if len(out_shape) == 0 else out_shape[-1]
    linear = torch.nn.Linear(in_features, out_features)

    def _helper(_x):
        if len(in_shape) == 0:
            _x = _x.unsqueeze(-1)
        _y = linear(_x)
        if len(out_shape) == 0:
            _y = _y.squeeze(-1)
        return _y

    # Compute the jacobian with the batch_autograd_function.
    y = _helper(x)
    jac = batch_autograd_jacobian(x, y)

    # Reference calculation.
    jac_autograd_tmp = torch.autograd.functional.jacobian(_helper, x, vectorize=True)
    jac_autograd = torch.empty(batch_size, *out_shape, *in_shape)
    for batch_idx in range(batch_size):
        jac_autograd[batch_idx] = jac_autograd_tmp[batch_idx].select(
            dim=len(out_shape), index=torch.tensor(batch_idx))

    assert jac.shape == (batch_size, *out_shape, *in_shape)
    assert torch.allclose(jac, jac_autograd)
