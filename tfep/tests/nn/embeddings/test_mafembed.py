#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test MAF layer in ``tfep.nn.embeddings.mafembed``.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import pytest
import torch

from tfep.nn.embeddings.mafembed import PeriodicEmbedding
from .. import create_random_input


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

def create_input(batch_size, n_features_in, limits=None, periodic_indices=None, seed=0):
    """Create random input with correct boundaries.

    If limits=(x0, xf) are passed, the input is constrained to be within x0 and
    xf. x0 and xf can be both floats or Tensors of size (n_features,).

    If periodic_indices is passed, only these DOFs are restrained between x0
    and xf.

    """
    # The non-periodic features do not need to be restrained between 0 and 1.
    x = create_random_input(batch_size, n_features_in, x_func=torch.randn, seed=seed)

    if limits is not None:
        if periodic_indices is None:
            # All DOFs are periodic. Ignore previous random input.
            periodic_indices = torch.tensor(list(range(n_features_in)))

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
def test_periodic_embedding(n_periodic, limits):
    """Test that PeriodicEmbedding lifts the correct degrees of freedom."""
    batch_size = 3
    n_features_in = 5
    limits = torch.tensor(limits)

    # Select a few random indices for sampling.
    periodic_indices = torch.sort(torch.randperm(n_features_in)[:n_periodic]).values
    lifter = PeriodicEmbedding(n_features_in=n_features_in, periodic_indices=periodic_indices, limits=limits)

    # Create random input with the correct periodicity.
    x = create_input(batch_size, n_features_in, limits=limits, periodic_indices=periodic_indices)

    # Lift periodic DOFs.
    x_lifted = lifter(x)

    # The lifted input must have n_periodic more elements.
    assert x.shape[1] + n_periodic == x_lifted.shape[1]

    # The lifter leaves unaltered the non-periodic DOFs and duplicate the periodic ones.
    shift_idx = 0
    for i in range(n_features_in):
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
