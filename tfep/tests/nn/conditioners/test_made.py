#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test classes and functions in tfep.nn.conditioners.made.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import pytest
import torch

from tfep.nn.conditioners.made import generate_degrees, MADE

from .. import check_autoregressive_property


# =============================================================================
# FIXTURES
# =============================================================================

# Each test case is a tuple:
# (degrees_in, degrees_out, hidden_layers, expected_degrees_hidden)
@pytest.fixture(
    params=[
        ([0, 1, 2], [0, 1, 2], 1, [[0, 1, 0]]),
        ([0, -1, 1, 2], [0, 1, 2, 3], 2, [[0, -1, 1, 2], [0, -1, 1, 2]]),
        ([3, 2, 1, -1, 0], [0, 0, 1, 1, 2, 2, 3, 3], 1, [[2, 1, -1, 0, 2, 1]]),
        ([2, -1, 0, 1], [1, 2, 0, 3]*3, 1, [[2, -1, 0, 1, 2, -1, 0]]),
        ([2, -1, 3, 0, 1], [1, 2, 0, 3]*3, [6], [[2, -1, 0, 1, 2, -1]]),
        ([2, -1, 3, 0, 1], [1, 2, 0, 3]*3, [6, 4], [[2, -1, 0, 1, 2, -1], [2, -1, 0, 1]]),
        ([2, -1, 3, 0, 1], [1, 2, 0, 3]*3, [[1, 0, -1, 2]], [[1, 0, -1, 2]]),
    ]
)
def init_args_cases(request):
    return request.param


# =============================================================================
# TESTS
# =============================================================================

@pytest.mark.parametrize('n_features,kwargs,expected', [
    (3, {}, [0, 1, 2]),
    (2, dict(order='descending'), [1, 0]),
    (5, dict(max_value=1), [0, 1, 0, 1, 0]),
    (5, dict(order='descending', max_value=1), [1, 0, 1, 0, 1]),
    (6, dict(conditioning_indices=[0, 3]), [-1, 0, 1, -1, 2, 3]),
    (5, dict(order='descending', conditioning_indices=[4]), [3, 2, 1, 0, -1]),
    (5, dict(max_value=2, conditioning_indices=[1]), [0, -1, 1, 2, 0]),
    (6, dict(order='descending', max_value=2, conditioning_indices=[0, 5]), [-1, 2, 1, 0, 2, -1]),
    (7, dict(max_value=1, conditioning_indices=[0, 4], repeats=2), [-1, 0, 0, 1, -1, 1, 0]),
    (6, dict(order='descending', conditioning_indices=[1, 5], repeats=3), [1, -1, 1, 1, 0, -1]),
    (7, dict(conditioning_indices=[1, 2], repeats=[1, 2, 3]), [0, -1, -1, 1, 1, 2, 2]),
    (6, dict(order='descending', max_value=1, conditioning_indices=[2], repeats=[1, 2]), [1, 0, -1, 0, 1, 0]),
])
def test_generate_degrees(n_features, kwargs, expected):
    """Test the function ``generate_degrees()``."""
    degrees = generate_degrees(n_features, **kwargs)
    assert torch.all(degrees == torch.as_tensor(expected))


def test_error_unknown_order():
    """An exception is raised if an unknown value for the order argument is passed."""
    with pytest.raises(ValueError, match='Accepted string values for'):
        generate_degrees(n_features=2, order='wrong')


def test_made_get_degrees_hidden(init_args_cases):
    """Test the method MADE._get_degrees_hidden()."""
    degrees_in, degrees_out, hidden_layers, expected_degrees_hidden = init_args_cases
    degrees_hidden = MADE._get_degrees_hidden(torch.tensor(degrees_in), torch.tensor(degrees_out), hidden_layers)
    for layer_degrees, expected_layer_degrees in zip(degrees_hidden, expected_degrees_hidden):
        assert torch.all(layer_degrees == torch.tensor(expected_layer_degrees))


@pytest.mark.parametrize('weight_norm', [True, False])
def test_made_linear_layers_dimensions(weight_norm, init_args_cases):
    """Test that the linear layers have the correct dimensions."""
    degrees_in, degrees_out, hidden_layers, expected_degrees_hidden = init_args_cases
    n_hidden_layers = len(expected_degrees_hidden)

    made = MADE(
        degrees_in=degrees_in,
        degrees_out=degrees_out,
        hidden_layers=hidden_layers,
        weight_norm=weight_norm,
    )

    # Masked linear layers are alternated with nonlinearities.
    masked_linear_modules = made.layers[::2]

    # Check all dimensions.
    assert len(masked_linear_modules) == n_hidden_layers + 1
    assert masked_linear_modules[0].in_features == len(degrees_in)
    for layer_idx in range(n_hidden_layers):
        layer_width = len(expected_degrees_hidden[layer_idx])
        masked_linear_modules[layer_idx].out_features == layer_width
        masked_linear_modules[layer_idx+1].in_features == layer_width
    assert masked_linear_modules[-1].out_features == len(degrees_out)


def test_error_to_narrow_hidden_layer():
    """An error is raised if a width smaller than the number of input features is requested."""
    with pytest.raises(ValueError, match='is too small for the number'):
        MADE._get_degrees_hidden(torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2]), [1])


def test_error_to_ignored_hidden_degree():
    """An error is raised if a hidden degree that would be ignored by the output is passed."""
    with pytest.raises(ValueError, match='nodes with degrees that will be ignored'):
        MADE._get_degrees_hidden(torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2]), [[2]])


@pytest.mark.parametrize('weight_norm', [True, False])
def test_made_autoregressive_property(weight_norm, init_args_cases):
    """Test the autoregressive property of a made network."""
    degrees_in, degrees_out, hidden_layers, expected_degrees_hidden = init_args_cases
    made = MADE(
        degrees_in=degrees_in,
        degrees_out=degrees_out,
        hidden_layers=hidden_layers,
        weight_norm=weight_norm,
    )

    # Shortcuts.
    dimension_in = len(degrees_in)
    dimension_out = len(degrees_out)

    # Create a random input and make it go through the net.
    x = torch.randn(dimension_in)
    y = made.forward(x.unsqueeze(0))
    assert y.shape == (1, dimension_out)

    # Check autoregressive property.
    check_autoregressive_property(made, x, degrees_in, degrees_out)
