#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test objects and function in tfep.nn.encoders.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import numpy as np
import pytest
import torch

from tfep.nn.encoders import GaussianRadialBasisExpansion


# =============================================================================
# REFERENCE IMPLEMENTATIONS
# =============================================================================

def reference_gaussian_basis_expansion(distances, means, stds):
    """Reference implementation of GaussianRadialBasisExpansion for testing."""
    distances = distances.detach().numpy()
    means = means.detach().numpy()
    vars = stds.detach().numpy()**2

    batch_size, n_atoms, _ = distances.shape
    n_gaussians = len(means)

    # Returned value.
    encoding = np.empty((batch_size, n_atoms, n_atoms, n_gaussians), dtype=distances.dtype)

    # Compute distances.
    for b in range(len(distances)):
        for i in range(len(distances[b])):
            for j in range(len(distances[b, i])):
                encoding[b, i, j] = np.exp(-(distances[b, i, j] - means)**2/vars)

    return  encoding


# =============================================================================
# TESTS GAUSSIAN RADIAL BASIS EXPANSION
# =============================================================================

@pytest.mark.parametrize('batch_size', [1, 3])
@pytest.mark.parametrize('n_features', [1, 8])
@pytest.mark.parametrize('seed', list(range(3)))
def test_gaussian_basis_expansion_reference(batch_size, n_features, seed):
    """Compare PyTorch and reference implementation of the gaussian basis expansion."""
    n_atoms = 5
    generator = torch.Generator()
    generator.manual_seed(seed)

    # Create random input and measure distances.
    x = torch.randn((batch_size, n_atoms, 3), generator=generator)
    distances = torch.cdist(x, x)

    # Create random means and stds.
    means = torch.rand(n_features, generator=generator)
    stds = torch.rand(n_features, generator=generator) + 0.2

    # Compute torch and reference encoding.
    expansion = GaussianRadialBasisExpansion(means, stds)
    encoding_pt = expansion(distances)
    encoding_np = reference_gaussian_basis_expansion(distances, means, stds)

    assert np.allclose(encoding_pt.detach().numpy(), encoding_np)


def test_gaussian_basis_expansion_equidistant():
    """Test that initializing an expansion of equidistant gaussian generates the correct means and stds."""
    expansion = GaussianRadialBasisExpansion.from_range(n_gaussians=3, max_mean=3, min_mean=1)
    assert torch.allclose(expansion._means, torch.tensor([1.0, 2.0, 3.0]))
    assert torch.allclose(expansion._log_gammas, torch.log(1./torch.full((3,), fill_value=9.0)))
