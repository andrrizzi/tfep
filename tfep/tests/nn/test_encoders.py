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

from tfep.nn.encoders import GaussianBasisExpansion, BehlerParrinelloRadialExpansion


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
        torch_impl, reference_impl, data, n_features=1, generator=None, **kwargs):
    """Compare the PyTorch vs the reference implementation of the expansion."""
    # Create random means and stds.
    means = torch.rand(n_features, generator=generator)
    stds = torch.rand(n_features, generator=generator) + 0.2

    # Compute torch and reference encoding.
    encoding_pt = torch_impl(means=means, stds=stds, **kwargs)(data)
    encoding_np = reference_impl(data=data, means=means, stds=stds, **kwargs)

    assert np.allclose(encoding_pt.detach().numpy(), encoding_np)


# =============================================================================
# REFERENCE IMPLEMENTATIONS
# =============================================================================

def reference_gaussian_basis_expansion(data, means, stds):
    """Reference implementation of GaussianBasisExpansion for testing."""
    data = data.detach().numpy()
    means = means.detach().numpy()
    vars = stds.detach().numpy()**2

    n_gaussians = len(means)

    # Returned value.
    shape = list(data.shape) + [n_gaussians]
    encoding = np.empty(shape, dtype=data.dtype)

    # Compute data.
    for indices, val in np.ndenumerate(data):
        encoding[indices] = np.exp(-(val - means)**2/vars)

    return  encoding


def reference_behler_parrinello_expansion(data, means, stds, r_cutoff):
    """Reference implementation of BehlerParrinelloRadialExpansion for testing."""
    encoding = reference_gaussian_basis_expansion(data, means, stds)

    data = data.detach().numpy()

    # Apply switching function.
    for b in range(len(data)):
        for i in range(len(data[b])):
            for j in range(len(data[b, i])):
                if data[b, i, j] > r_cutoff:
                    switching = 0.0
                else:
                    switching = 0.5 * np.cos(torch.pi / r_cutoff * data[b, i, j]) + 0.5
                encoding[b, i, j] *= switching

    return encoding


# =============================================================================
# TESTS GAUSSIAN RADIAL BASIS EXPANSION
# =============================================================================

@pytest.mark.parametrize('batch_size', [1, 3])
@pytest.mark.parametrize('n_features', [1, 8])
@pytest.mark.parametrize('seed', list(range(3)))
@pytest.mark.parametrize('input_type', ['distance', 'scalar'])
def test_gaussian_basis_expansion_reference(batch_size, n_features, seed, input_type):
    """Compare PyTorch and reference implementation of the gaussian basis expansion."""
    generator = torch.Generator()
    generator.manual_seed(seed)

    # Create random input and measure distances.
    if input_type == 'distance':
        n_atoms = 5
        x = torch.randn((batch_size, n_atoms, 3), generator=generator)
        x = torch.cdist(x, x)
    else:
        x = torch.rand(batch_size)

    # Check against reference.
    check_reference_expansion(
        torch_impl=GaussianBasisExpansion,
        reference_impl=reference_gaussian_basis_expansion,
        data=x,
        n_features=n_features,
        generator=generator,
    )


def test_gaussian_basis_expansion_equidistant():
    """Test that initializing an expansion of equidistant gaussian generates the correct means and stds."""
    expansion = GaussianBasisExpansion.from_range(n_gaussians=3, max_mean=3, min_mean=1)
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
    generator = torch.Generator()
    generator.manual_seed(seed)

    # Create random input and measure distances.
    n_atoms = 5
    x = torch.randn((batch_size, n_atoms, 3), generator=generator)
    distances = torch.cdist(x, x)

    check_reference_expansion(
        torch_impl=BehlerParrinelloRadialExpansion,
        reference_impl=reference_behler_parrinello_expansion,
        data=distances,
        n_features=n_features,
        generator=generator,
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

    def check_distances(dist):
        # Compute the value of the expansion at the cutoff.
        distances = torch.full((batch_size, 2, 2), fill_value=dist, requires_grad=True)
        expansion = expansion_layer(distances)
        assert torch.allclose(expansion, torch.zeros_like(expansion))

        # Compute the gradient, which should also be zero.
        loss = torch.sum(expansion)
        loss.backward()
        assert torch.allclose(distances.grad, torch.zeros_like(distances))


    # Check the value and the gradient at and after the cutoff.
    check_distances(r_cutoff)
    check_distances(r_cutoff+1)
