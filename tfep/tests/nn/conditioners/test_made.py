#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test layers in tfep.nn.conditioners.made.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

from unittest import mock

import numpy as np
import pytest
import torch

from tfep.utils.misc import ensure_tensor_sequence
from tfep.nn.conditioners.made import MADE


# =============================================================================
# FIXTURES
# =============================================================================

# Each test case is a tuple:
# (dimension_in, dimensions_hidden, out_per_dimension, conditioning_indices,
#       expected_dimensions_hidden)
@pytest.fixture(
    params=[
        (3, 2, 1, [],
            [2, 2]),
        (3, [5], 1, [],
            [5]),
        (3, 4, 1, [1],
            [2]*4),
        (5, 7, 2, [],
            [8]*7),
        (5, 7, 2, [0, 1],
            [8]*7),
        (5, [4, 7, 9], 2, [1, 4],
            [4, 7, 9])
    ]
)
def dimensions(request):
    return request.param

# Each test case is a tuple:
# (blocks, degrees_in,
#       dimension_in, dimensions_hidden, out_per_dimension, conditioning_indices,
#       expected_dimensions_hidden)
@pytest.fixture(
    params=[
        (2, 'input',
            3, 2, 1, [],
            [2, 2]),
        (2, 'reversed',
            3, 2, 1, [],
            [1, 1]),
        ([1, 2], 'input',
            3, 2, 1, [],
            [1, 1]),
        ([1, 2], 'reversed',
            3, 2, 1, [],
            [2, 2]),
        (3, 'input',
            7, 3, 2, [0, 1],
            [10]*3),
        (3, 'reversed',
            7, 3, 2, [0, 3],
            [8]*3),
        ([1, 2, 2], 'input',
            7, 3, 2, [2, 5],
            [10]*3),
        ([1, 2, 2], 'reversed',
            7, 3, 2, [0, 6],
            [12]*3),
        ([2, 3], 'input',
            7, 3, 2, [3, 6],
            [8]*3),
        ([2, 3], 'reversed',
            7, 3, 2, [5, 6],
            [10]*3),
        (2, 'input',
            7, [6, 9, 11], 2, [1, 3],
            [6, 9, 11]),
        (2, 'reversed',
            7, [6, 9, 11], 2, [4, 6],
            [6, 9, 11]),
        ([1, 2, 3], np.array([0, 1, 2]),
            8, 3, 2, [0, 6],
            [10]*3),
        ([1, 2, 3], np.array([0, 2, 1]),
            8, 3, 2, [4, 5],
            [12]*3),
        ([1, 2, 3], np.array([2, 0, 1]),
            8, 3, 2, [1, 2],
            [14]*3),
    ]
)
def blocked_dimensions(request):
    return request.param


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def generate_random_degrees_in(dimension_in, conditioning_indices):
    # Make sure the test is reproducible with a random state.
    random_state = np.random.RandomState(dimension_in)
    return random_state.permutation(list(range(dimension_in-len(conditioning_indices))))


# =============================================================================
# TESTS
# =============================================================================

@pytest.mark.parametrize('degrees_in', ['input', 'reversed', 'random'])
def test_MADE_get_dimensions(degrees_in, dimensions):
    """Test the method MADE._get_dimensions without blocks.

    The dimensions should be independent of the degrees_in option.
    """
    if degrees_in == 'random':
        degrees_in = generate_random_degrees_in(dimensions[0], dimensions[3])
    check_MADE_get_dimensions(1, degrees_in, *dimensions)


def test_MADE_get_dimensions_blocks(blocked_dimensions):
    """Test the method MADE._get_dimensions with blocks."""
    check_MADE_get_dimensions(*blocked_dimensions)


def check_MADE_get_dimensions(
        blocks, degrees_in, dimension_in, dimensions_hidden, out_per_dimension,
        conditioning_indices, expected_dimensions_hidden
):
    """Used by test_MADE_get_dimensions and test_MADE_get_dimensions_blocks."""
    dimensions_hidden = ensure_tensor_sequence(dimensions_hidden, dtype=int)
    conditioning_indices = ensure_tensor_sequence(conditioning_indices, dtype=int)
    degrees_in = ensure_tensor_sequence(degrees_in, dtype=int)
    blocks = ensure_tensor_sequence(blocks, dtype=int)

    n_hidden_layers, dimensions_hidden, expanded_blocks = MADE._get_dimensions(
        dimension_in, dimensions_hidden, out_per_dimension, conditioning_indices,
        degrees_in, blocks, shorten_last_block=True)

    assert n_hidden_layers == len(expected_dimensions_hidden)
    assert dimensions_hidden.tolist() == expected_dimensions_hidden


@pytest.mark.parametrize(('dimension_in,conditioning_indices,degrees_in,blocks,'
                                'expected_degrees_in,expected_degrees_hidden_motif'), [
    (5, [], 'input', [3, 2],
        [0, 0, 0, 1, 1], [0, 0, 0]),
    (7, [0, 3], 'input', [3, 2],
        [-1, 0, 0, -1, 0, 1, 1], [-1, 0, 0, -1, 0]),
    (5, [], 'reversed', [3, 2],
        [1, 1, 1, 0, 0], [0, 0]),
    (7, [0, 5], 'reversed', [3, 2],
        [-1, 1, 1, 1, 0, -1, 0], [-1, 0, -1, 0]),
    (6, [], [2, 0, 1], [1, 3, 2],
        [2, 0, 0, 0, 1, 1], [0, 0, 0, 1, 1]),
    (7, [6], [2, 0, 1], [1, 3, 2],
        [2, 0, 0, 0, 1, 1, -1], [0, 0, 0, 1, 1, -1]),
])
def test_MADE_generate_degrees(dimension_in, conditioning_indices, degrees_in, blocks,
                               expected_degrees_in, expected_degrees_hidden_motif):
    """Test that the input degrees and the motif for the hidden nodes are correct."""
    # Create a mock MADE class with the blocks attribute.
    mock_made = mock.Mock(blocks=blocks)

    # _assign_degrees_in expects tensors.
    conditioning_indices = ensure_tensor_sequence(conditioning_indices, dtype=int)
    degrees_in = ensure_tensor_sequence(degrees_in, dtype=int)
    MADE._assign_degrees_in(mock_made, dimension_in, conditioning_indices, degrees_in)
    mock_made.degrees_in = mock_made._degrees_in

    # When not None, _assign_degrees_hidden_motif simply returns degrees_hidden_motif
    # so this simply tests that the returned motif is based on degrees_in.
    MADE._assign_degrees_hidden_motif(mock_made, degrees_hidden_motif=None)

    assert torch.all(mock_made._degrees_in == torch.tensor(expected_degrees_in))
    assert torch.all(mock_made._degrees_hidden_motif == torch.tensor(expected_degrees_hidden_motif))


@pytest.mark.parametrize('weight_norm', [False, True])
def test_MADE_mask_dimensions(weight_norm, dimensions):
    """Test that the dimension of the hidden layers without blocks follow the init options correctly."""
    check_MADE_mask_dimensions(1, 'input', *dimensions[:-1], weight_norm=weight_norm)


@pytest.mark.parametrize('weight_norm', [False, True])
def test_MADE_mask_dimensions_blocks(weight_norm, blocked_dimensions):
    """Test that the dimension of the hidden layers with blocks follow the init options correctly."""
    check_MADE_mask_dimensions(*blocked_dimensions[:-1], weight_norm=weight_norm)


def check_MADE_mask_dimensions(blocks, degrees_in, dimension_in, dimensions_hidden,
                               out_per_dimension, conditioning_indices, weight_norm):
    """Used by test_MADE_mask_dimensions and test_MADE_mask_dimensions_blocks."""
    made = MADE(
        dimension_in=dimension_in,
        dimensions_hidden=dimensions_hidden,
        out_per_dimension=out_per_dimension,
        conditioning_indices=conditioning_indices,
        degrees_in=degrees_in,
        weight_norm=weight_norm,
        blocks=blocks,
        shorten_last_block=True
    )

    # _get_dimensions expects tensors.
    dimensions_hidden = ensure_tensor_sequence(dimensions_hidden, dtype=int)
    conditioning_indices = ensure_tensor_sequence(conditioning_indices, dtype=int)
    degrees_in = ensure_tensor_sequence(degrees_in, dtype=int)
    blocks = ensure_tensor_sequence(blocks, dtype=int)

    # Compute the expected dimensions.
    n_hidden_layers, dimensions_hidden, expanded_blocks = made._get_dimensions(
        dimension_in, dimensions_hidden, out_per_dimension, conditioning_indices,
        degrees_in, blocks, shorten_last_block=True)
    dimension_out = (dimension_in - len(conditioning_indices)) * out_per_dimension

    # Masked linear layers are alternated with nonlinearities.
    masked_linear_modules = made.layers[::2]

    # Check all dimensions.
    assert len(masked_linear_modules) == n_hidden_layers + 1
    assert masked_linear_modules[0].in_features == dimension_in
    for layer_idx in range(n_hidden_layers):
        masked_linear_modules[layer_idx].out_features == dimensions_hidden[layer_idx]
        masked_linear_modules[layer_idx+1].in_features == dimensions_hidden[layer_idx]
    assert masked_linear_modules[-1].out_features == dimension_out

    # Test correct implementation of the Python properties.
    assert made.dimension_in == dimension_in
    assert made.n_layers == n_hidden_layers + 1
    assert torch.all(made.dimensions_hidden == dimensions_hidden)
    assert torch.all(made.conditioning_indices == conditioning_indices)
    assert made.dimension_conditioning == len(conditioning_indices)
    assert made.dimension_out == dimension_out


@pytest.mark.parametrize('weight_norm', [False, True])
@pytest.mark.parametrize('degrees_in', ['input', 'reversed', 'random'])
def test_MADE_autoregressive_property(weight_norm, degrees_in, dimensions):
    """Test that MADE without blocks satisfies the autoregressive property.

    The test creates a random input for a MADE network and then perturbs
    it one a time, making sure that output k changes if and only if
    input with a smaller degrees have changed.

    """
    # Generate a random permutation if requested.
    if degrees_in == 'random':
        degrees_in = generate_random_degrees_in(dimensions[0], dimensions[3])
    check_MADE_autoregressive_property(1, degrees_in, *dimensions[:-1], weight_norm=weight_norm)


@pytest.mark.parametrize('weight_norm', [False, True])
def test_MADE_autoregressive_property_blocks(weight_norm, blocked_dimensions):
    """Test that MADE with blocks satisfies the autoregressive property.

    The test creates a random input for a MADE network and then perturbs
    it one a time, making sure that output k changes if and only if
    input with a smaller degrees have changed.

    """
    check_MADE_autoregressive_property(*blocked_dimensions[:-1], weight_norm=weight_norm)


def check_MADE_autoregressive_property(blocks, degrees_in, dimension_in, dimensions_hidden,
                                       out_per_dimension, conditioning_indices, weight_norm):
    """Used by test_MADE_autoregressive_property and test_MADE_autoregressive_property_blocks."""
    made = MADE(
        dimension_in=dimension_in,
        dimensions_hidden=dimensions_hidden,
        out_per_dimension=out_per_dimension,
        conditioning_indices=conditioning_indices,
        degrees_in=degrees_in,
        blocks=blocks,
        shorten_last_block=True
    )

    # Shortcut.
    dimension_out = made.dimension_out

    # Create a random input and make it go through the net.
    x = np.random.randn(1, dimension_in)
    input = torch.tensor(x, dtype=torch.float, requires_grad=True)
    output = made.forward(input)
    assert output.shape == (1, dimension_out)

    # Make sure that there are no duplicate degrees in the input/output.
    assert len(set(made.degrees_in.tolist())) == len(made.blocks) + int(len(conditioning_indices) > 0)

    for out_idx in range(dimension_out // out_per_dimension):
        # Compute the gradient of the out_idx-th dimension of the
        # output with respect to the gradient vector.
        loss = torch.sum(output[0, out_idx:dimension_out:dimension_out//out_per_dimension])
        loss.backward(retain_graph=True)

        # In all cases, the conditioning features should affect the whole output.
        grad = input.grad[0]
        assert torch.all(grad[conditioning_indices] != 0.0)

        # Now consider the non-conditioning features only.
        conditioning_indices_set = set(conditioning_indices)
        mapped_indices = [i for i in range(dimension_in) if i not in conditioning_indices_set]
        grad = grad[mapped_indices]
        degrees = made.degrees_in[mapped_indices]

        # For the autoregressive property to hold, the k-th output should
        # have non-zero gradient only for the inputs with a smaller degree.
        degree_out = degrees[out_idx]
        for in_idx in range(len(degrees)):
            if degrees[in_idx] < degree_out:
                assert grad[in_idx] != 0
            else:
                assert grad[in_idx] == 0

        # Reset gradients for next iteration.
        made.zero_grad()
        input.grad.data.zero_()
