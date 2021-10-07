#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test layers in tfep.nn.flows.partial.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import pytest
import torch

from tfep.nn.flows.maf import MAF
from tfep.nn.flows.partial import PartialFlow
from ..utils import create_random_input


# =============================================================================
# TESTS
# =============================================================================

@pytest.mark.parametrize('constant_input_indices', [None, [1, 4]])
@pytest.mark.parametrize('dimension_conditioning', [0, 1, 2])
@pytest.mark.parametrize('weight_norm', [False, True])
def test_round_trip_PartialFlow(constant_input_indices, dimension_conditioning, weight_norm):
    """Test that the PartialFlow.inverse(PartialFlow.forward(x)) equals the identity."""
    dimension = 7
    dimensions_hidden = 2
    batch_size = 2

    if constant_input_indices is None:
        n_constant_input_indices = 0
    else:
        n_constant_input_indices = len(constant_input_indices)

    # Add a stack of three MAF layers
    flows = []
    for degrees_in in ['input', 'reversed', 'input']:
        # We don't initialize as the identity function to make the test meaningful.
        flows.append(MAF(
            dimension_in=dimension - n_constant_input_indices,
            dimensions_hidden=dimensions_hidden,
            dimension_conditioning=dimension_conditioning,
            degrees_in=degrees_in,
            weight_norm=weight_norm,
            initialize_identity=False
        ))
    flow = PartialFlow(
        *flows,
        constant_indices=constant_input_indices,
    )

    # Create random input.
    x = create_random_input(batch_size, dimension)
    y, log_det_J = flow.forward(x)
    x_inv, log_det_J_inv = flow.inverse(y)

    assert torch.allclose(x, x_inv)
    assert torch.allclose(log_det_J + log_det_J_inv, torch.zeros(batch_size), atol=1e-04)


def test_constant_input_multiscale_architect():
    """Test the constant_input_indices multiscale architecture."""
    dimension = 10
    dimensions_hidden = 2
    batch_size = 5
    constant_input_indices = [0, 1, 4, 7]

    # Create a three-layer MAF flow.
    flows = []
    for degrees_in in ['input', 'reversed', 'input']:
        # We don't initialize as the identity function to make the test meaningful.
        flows.append(MAF(
            dimension_in=dimension - len(constant_input_indices),
            dimensions_hidden=dimensions_hidden,
            degrees_in=degrees_in,
            initialize_identity=False
        ))
    flow = PartialFlow(
        *flows,
        constant_indices=constant_input_indices
    )

    # Create random input.
    x = create_random_input(batch_size, dimension)

    # Make sure the flow is not the identity
    # function or the test doesn't make sense.
    y, log_det_J = flow.forward(x)
    assert not torch.allclose(x, y)

    # The gradient of the constant input should always be 0.0.
    loss = torch.sum(y)
    loss.backward()
    assert torch.all(x.grad[:, constant_input_indices] == 1.0)

    # Make sure the inverse also works.
    x_inv, log_det_J_inv = flow.inverse(y)
    assert torch.allclose(x, x_inv, atol=1e-04)
    assert torch.allclose(log_det_J + log_det_J_inv, torch.zeros(batch_size), atol=1e-04)
