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

from tfep.nn.encoders import GaussianRadialBasisExpansion, BehlerParrinelloRadialExpansion


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
# UTILITY FUNCTION
# =============================================================================

def check_reference_expansion(
        torch_impl, reference_impl, batch_size=1, n_atoms=2, n_features=1, seed=0, **kwargs):
    """Compare the PyTorch vs the reference implementation of the expansion."""
    generator = torch.Generator()
    generator.manual_seed(seed)

    # Create random input and measure distances.
    x = torch.randn((batch_size, n_atoms, 3), generator=generator)
    distances = torch.cdist(x, x)

    # Create random means and stds.
    means = torch.rand(n_features, generator=generator)
    stds = torch.rand(n_features, generator=generator) + 0.2

    # Compute torch and reference encoding.
    encoding_pt = torch_impl(means=means, stds=stds, **kwargs)(distances)
    encoding_np = reference_impl(distances=distances, means=means, stds=stds, **kwargs)

    assert np.allclose(encoding_pt.detach().numpy(), encoding_np)


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


def reference_behler_parrinello_expansion(distances, means, stds, r_cutoff):
    """Reference implementation of BehlerParrinelloRadialExpansion for testing."""
    encoding = reference_gaussian_basis_expansion(distances, means, stds)

    distances = distances.detach().numpy()

    # Apply switching function.
    for b in range(len(distances)):
        for i in range(len(distances[b])):
            for j in range(len(distances[b, i])):
                switching = 0.5 * np.cos(torch.pi / r_cutoff * distances[b, i, j]) + 0.5
                encoding[b, i, j] *= switching

    return encoding


# =============================================================================
# TESTS GAUSSIAN RADIAL BASIS EXPANSION
# =============================================================================

@pytest.mark.parametrize('batch_size', [1, 3])
@pytest.mark.parametrize('n_features', [1, 8])
@pytest.mark.parametrize('seed', list(range(3)))
def test_gaussian_basis_expansion_reference(batch_size, n_features, seed):
    """Compare PyTorch and reference implementation of the gaussian basis expansion."""
    check_reference_expansion(
        torch_impl=GaussianRadialBasisExpansion,
        reference_impl=reference_gaussian_basis_expansion,
        batch_size=batch_size,
        n_atoms=5,
        n_features=n_features,
        seed=seed,
    )


def test_gaussian_basis_expansion_equidistant():
    """Test that initializing an expansion of equidistant gaussian generates the correct means and stds."""
    expansion = GaussianRadialBasisExpansion.from_range(n_gaussians=3, max_mean=3, min_mean=1)
    assert torch.allclose(expansion._means, torch.tensor([1.0, 2.0, 3.0]))
    assert torch.allclose(expansion._log_gammas, torch.log(1./torch.full((3,), fill_value=9.0)))


# =============================================================================
# TESTS BEHLER-PARRINELLO RADIAL BASIS EXPANSION
# =============================================================================

@pytest.mark.parametrize('batch_size', [1, 3])
@pytest.mark.parametrize('n_features', [1, 8])
@pytest.mark.parametrize('seed', list(range(3)))
def test_behler_parrinello_expansion_reference(batch_size, n_features, seed):
    """Compare PyTorch and reference implementation of the Behler-Parrinello radial expansion."""
    check_reference_expansion(
        torch_impl=BehlerParrinelloRadialExpansion,
        reference_impl=reference_behler_parrinello_expansion,
        batch_size=batch_size,
        n_atoms=5,
        n_features=n_features,
        seed=seed,
        r_cutoff=2.0,
    )


@pytest.mark.parametrize('batch_size', [1, 3])
@pytest.mark.parametrize('n_gaussians', [1, 3])
@pytest.mark.parametrize('min_mean', [0.0, 1.0])
def test_behler_parrinello_expansion_cutoff(batch_size, n_gaussians, min_mean):
    """Check that value and gradient of the Behler-Parrinello expansion is 0 at the cutoff."""
    r_cutoff = 3.0

    if n_gaussians == 1:
        expansion_layer = BehlerParrinelloRadialExpansion(
            r_cutoff=r_cutoff,
            means=torch.tensor([0.0]),
            stds=torch.tensor([r_cutoff])
        )
    else:
        expansion_layer = BehlerParrinelloRadialExpansion.from_range(
            r_cutoff=r_cutoff,
            n_gaussians=n_gaussians,
            max_mean=r_cutoff,
            min_mean=min_mean,
        )

    distances = torch.full((batch_size, 2, 2), fill_value=r_cutoff, requires_grad=True)

    # Compute the value of the expansion at the cutoff.
    expansion = expansion_layer(distances)
    assert torch.allclose(expansion, torch.zeros_like(expansion))

    # Compute the gradient, which should also be zero.
    loss = torch.sum(expansion)
    loss.backward()
    assert torch.allclose(distances.grad, torch.zeros_like(distances))
