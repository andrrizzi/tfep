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

import numpy as np
import pytest
import torch

import tfep.nn.conditioners.made
from tfep.nn.flows import MAF, SequentialFlow, PartialFlow
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
# TESTS
# =============================================================================

@pytest.mark.parametrize('conditioning_indices', [
    [],
    [0],
    [1, 3],
])
@pytest.mark.parametrize('weight_norm', [False, True])
@pytest.mark.parametrize('indices_type', [None, np.array, torch.tensor])
def test_round_trip_PartialFlow(conditioning_indices, weight_norm, indices_type):
    """Test that the PartialFlow.inverse(PartialFlow.forward(x)) equals the identity."""
    dimension = 7
    hidden_layers = 2
    batch_size = 2
    fixed_input_indices = [1, 4]

    # Test that this works with different types of fixed indices.
    if indices_type is not None:
        fixed_input_indices = indices_type(fixed_input_indices)

    # Add a stack of three MAF layers
    flows = []
    for degrees_in_order in ['ascending', 'descending', 'ascending']:
        # We don't initialize as the identity function to make the test meaningful.
        flows.append(MAF(
            degrees_in=tfep.nn.conditioners.made.generate_degrees(
                n_features=dimension - len(fixed_input_indices),
                conditioning_indices=conditioning_indices,
                order=degrees_in_order
            ),
            hidden_layers=hidden_layers,
            weight_norm=weight_norm,
            initialize_identity=False
        ))
    flow = PartialFlow(
        SequentialFlow(*flows),
        fixed_indices=fixed_input_indices,
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
    assert torch.all(x.grad[:, fixed_input_indices] == 1.0)

    # Make sure the inverse also works.
    x_inv, log_det_J_inv = flow.inverse(y)
    assert torch.allclose(x, x_inv)
    assert torch.allclose(log_det_J + log_det_J_inv, torch.zeros(batch_size))
