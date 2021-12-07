#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test objects and function in the module tfep.nn.flow.pca.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import numpy as np
import pytest
import torch

from tfep.nn.flows.pca import PCAWhitenedFlow
import tfep.utils.math

# =============================================================================
# GLOBAL VARIABLES
# =============================================================================

# Random number generator. Makes sure tests are reproducible from run to run.
GENERATOR = torch.Generator()
GENERATOR.manual_seed(0)


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

class IdentityFlow:

    def __init__(self, _):
        pass

    def __call__(self, x):
        return x, torch.zeros(len(x))

    def inverse(self, y):
        return self(y)


class MyMAF:

    def __init__(self, dimension_in):
        self.maf = tfep.nn.flows.MAF(dimension_in, initialize_identity=False)

    def __call__(self, x):
        return self.maf(x)

    def inverse(self, y):
        return self.maf.inverse(y)


# =============================================================================
# TESTS
# =============================================================================

@pytest.mark.parametrize('flow', [IdentityFlow, MyMAF])
@pytest.mark.parametrize('n_features', [3, 8])
@pytest.mark.parametrize('blacken', [True, False])
def test_pca_whitened_flow(flow, n_features, blacken):
    """Test with identity and MAF flow that PCAWhitenedFlow removes off-diagonal covariance."""
    # The batch size must be large enough for the eigenvalues of the covariance
    # matrix to be always positive.
    batch_size = 10

    # Create a positive-definite and symmetric matrix used for the covariance
    # and sample the normally-distributed data used to estimate the PCA matrix.
    mean = np.random.rand(n_features) * 2
    cov = np.random.rand(n_features, n_features)
    cov = np.dot(cov, cov.transpose())
    x = np.random.multivariate_normal(mean, cov, size=batch_size)
    x = torch.from_numpy(x)

    # Build flow and run flow.
    pca_flow = PCAWhitenedFlow(flow=flow(n_features), x=x, blacken=blacken)
    y, log_det_J = pca_flow(x)

    # If PCAWhitenedFlow wraps the identity flow, we can directly check the output.
    if flow is IdentityFlow:
        if blacken:
            # If we blacken, we should obtain the same output as before.
            assert torch.allclose(x, y)
            assert torch.allclose(log_det_J, torch.zeros(batch_size))
        else:
            # Otherwise the covariance estimate of the data should be diagonal
            # and the mean zero.
            cov_y = tfep.utils.math.cov(y)
            assert torch.allclose(cov_y, torch.diag(torch.diag(cov_y)))
            assert not torch.allclose(log_det_J, torch.zeros(batch_size))

    # Check the inverse.
    x_inv, log_det_J_inv = pca_flow.inverse(y)
    assert torch.allclose(x, x_inv)
    assert torch.allclose(log_det_J + log_det_J_inv, torch.zeros_like(log_det_J))
