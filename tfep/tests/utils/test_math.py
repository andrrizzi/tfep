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

import MDAnalysis.lib.mdamath
import numpy as np
import pytest
import torch

from tfep.utils.math import cov, angle


# =============================================================================
# GLOBAL VARIABLES
# =============================================================================

# Makes random input deterministic.
_GENERATOR = torch.Generator()
_GENERATOR.manual_seed(0)


# =============================================================================
# REFERENCE FUNCTIONS FOR TESTING
# =============================================================================

def reference_angle(v1, v2):
    v1np, v2np = v1.detach().numpy(), v2.detach().numpy()
    angles = [MDAnalysis.lib.mdamath.angle(v, v2np) for v in v1np]
    return torch.tensor(angles, dtype=v1.dtype)


# =============================================================================
# TESTS
# =============================================================================

@pytest.mark.parametrize('ddof', [0, 1])
@pytest.mark.parametrize('dim_n', [0, 1])
def test_cov(ddof, dim_n):
    """Test the covariance matrix against the numpy implementation."""
    random_state = np.random.RandomState(0)
    x = random_state.randn(10, 15)

    if dim_n == 0:
        cov_np = np.cov(x.T, ddof=ddof)
    else:
        cov_np = np.cov(x, ddof=ddof)

    cov_torch = cov(torch.tensor(x), dim_n=dim_n, ddof=ddof, inplace=True).numpy()

    assert np.allclose(cov_np, cov_torch)


def test_angle_axes():
    """Test the angle() function to measure angles between axes."""
    v1 = torch.eye(3)
    angles = angle(v1, torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(angles, torch.tensor([0.0, np.pi/2, np.pi/2]))
    angles = angle(v1, torch.tensor([0.0, 1.0, 0.0]))
    assert torch.allclose(angles, torch.tensor([np.pi/2, 0.0, np.pi/2]))
    angles = angle(v1, torch.tensor([0.0, 0.0, 1.0]))
    assert torch.allclose(angles, torch.tensor([np.pi/2, np.pi/2, 0.0]))


@pytest.mark.parametrize('batch_size', [1, 3])
@pytest.mark.parametrize('dimension', [2, 3, 4])
def test_angle_against_reference(batch_size, dimension):
    """Test the angle() function on random tensors against a reference implementation."""
    # Build a random input.
    v1 = torch.randn((batch_size, dimension), generator=_GENERATOR)
    v2 = torch.randn(dimension, generator=_GENERATOR)

    # Compare reference and PyTorch implementation.
    angles = angle(v1, v2)
    ref_angles = reference_angle(v1, v2)
    assert torch.allclose(angles, ref_angles)
