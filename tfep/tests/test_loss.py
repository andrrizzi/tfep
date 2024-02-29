#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test objects and function in the module ``tfep.loss``.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import pytest
import torch

from tfep.loss import BoltzmannKLDivLoss


# =============================================================================
# TESTS
# =============================================================================

@pytest.mark.parametrize('ignore_nan', [False, True])
@pytest.mark.parametrize('log_weights', [False, True])
def test_ignore_nan(ignore_nan, log_weights):
    """Test that NaNs are ignored when requested."""
    batch_size = 5
    n_tensors = 4 if log_weights else 3
    input_tensors = torch.randn(batch_size*n_tensors)

    # Randomly make one entry NaN.
    nan_idx = torch.randint(low=0, high=len(input_tensors), size=(1,))[0]
    input_tensors[nan_idx] = float('nan')

    # Split.
    target_potentials = input_tensors[:batch_size]
    log_det_J = input_tensors[batch_size:2*batch_size]
    ref_potentials = input_tensors[2*batch_size:3*batch_size]

    if log_weights:
        log_weights = input_tensors[3*batch_size:]
    else:
        log_weights = None

    # Create and evaluate loss.
    loss_func = BoltzmannKLDivLoss(ignore_nan=ignore_nan)
    loss_value = loss_func(target_potentials, log_det_J, log_weights, ref_potentials)

    # Check NaNs.
    if ignore_nan:
        assert not torch.isnan(loss_value)
    else:
        assert torch.isnan(loss_value)
