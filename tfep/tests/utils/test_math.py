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

from tfep.utils.math import cov


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
