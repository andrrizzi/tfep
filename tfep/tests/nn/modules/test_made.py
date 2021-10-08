#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test layers in tfep.nn.modules.made.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

from unittest import mock

import numpy as np
import pytest
import torch

from tfep.nn.modules import MADE


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

@pytest.mark.parametrize('dimension_in,n_layers,conditioning_indices', [
    (5, 3, ()),
    (5, 3, [0]),
    (7, 2, range(2)),
    (7, 2, [1, 5]),
    (7, 4, range(1)),
    (7, 4, [2, 3]),
    (10, 3, range(3)),
    (10, 3, [8,9]),
])
def test_MADE_create_mask(dimension_in, n_layers, conditioning_indices):
    """Test the method MADE.create_mask().

    Simulate a multi-layer MADE network with sequential degree assignment
    and check that all the masks have the appropriate shape and are lower
    triangular.

    """
    if isinstance(conditioning_indices, range):
        conditioning_indices = list(conditioning_indices)

    first_layer_dim = dimension_in
    inner_layer_dim = dimension_in - 1
    n_conditioning_dofs = len(conditioning_indices)
    output_layer_dim = dimension_in - n_conditioning_dofs

    # Determine conditioning and mapped degrees of freedom.
    conditioning_indices_set = set(conditioning_indices)
    mapped_indices = [i for i in range(dimension_in) if i not in conditioning_indices_set]

    # Assign degrees sequentially for the simulated multi-layer network.
    degrees = []
    for layer_idx in range(n_layers+1):
        # The first and last layers have an extra unit.
        if layer_idx == 0:
            # First layer must have the conditioning degrees in the correct positions.
            if len(conditioning_indices) == 0:
                layer_degrees = np.arange(output_layer_dim)
            else:
                layer_degrees = np.empty(first_layer_dim)
                layer_degrees[mapped_indices] = np.arange(output_layer_dim)
                layer_degrees[conditioning_indices] = -1
        elif layer_idx == n_layers:
            # Last layer only have non-conditioning degrees.
            layer_degrees = np.arange(output_layer_dim)
        else:
            # For intermediate layers, the position of the conditioning
            # degrees does not matter so we keep them at the beginning.
            layer_degrees = np.full(n_conditioning_dofs, fill_value=-1)
            layer_degrees = np.concatenate([layer_degrees, np.arange(inner_layer_dim-n_conditioning_dofs)])
        degrees.append(layer_degrees)

    # Build masks for all 3 layers.
    masks = [MADE.create_mask(degrees[i], degrees[i+1], is_output_layer=(i==n_layers-1))
             for i in range(n_layers)]

    for layer_idx, mask in enumerate(masks):
        is_first_mask = layer_idx == 0
        is_last_mask = (layer_idx == n_layers-1)

        # Check mask dimension.
        if is_first_mask:
            assert mask.shape == (inner_layer_dim, first_layer_dim)
        elif is_last_mask:
            assert mask.shape == (output_layer_dim, inner_layer_dim)
        else:
            assert mask.shape == (inner_layer_dim, inner_layer_dim)

        # The conditioning degrees affect all the mapped DOFs.
        if is_first_mask:
            conditioning_mask = mask[:, conditioning_indices]
            mask = mask[:, mapped_indices]
        else:
            conditioning_mask = mask[:, :n_conditioning_dofs]
            mask = mask[:, n_conditioning_dofs:]
        assert torch.all(conditioning_mask == torch.ones(conditioning_mask.shape))

        # Because we assigned the degrees in the order of the input,
        # the mask of the non-conditioning DOFs must be lower triangular.
        if is_last_mask:
            # Contrarily to other layers, the output DOFs don't depend on themselves.
            assert torch.all(mask == torch.tril(mask, diagonal=-1))
        else:
            assert torch.all(mask == torch.tril(mask))

        # In the first layer, the last input unit must have no connection to the
        # first hidden layer since that unit would otherwise not affect any output.
        if is_first_mask:
            assert torch.all(mask[:,-1] == False)


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
    n_hidden_layers, dimensions_hidden, expanded_blocks = MADE._get_dimensions(
        dimension_in, dimensions_hidden, out_per_dimension, conditioning_indices,
        degrees_in, blocks, shorten_last_block=True)

    assert n_hidden_layers == len(expected_dimensions_hidden)
    assert dimensions_hidden == expected_dimensions_hidden


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
    mock_made.degrees_in = MADE._assign_degrees_in(mock_made, dimension_in, conditioning_indices, degrees_in)

    # When not None, _generate_degrees_hidden_motif simply returns degrees_hidden_motif
    # so this simply tests that the returned motif is based on degrees_in.
    motif = MADE._generate_degrees_hidden_motif(mock_made, degrees_hidden_motif=None)

    assert np.all(mock_made.degrees_in == np.array(expected_degrees_in))
    assert np.all(motif == np.array(expected_degrees_hidden_motif))


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
    assert made.dimensions_hidden == dimensions_hidden
    assert made.conditioning_indices == conditioning_indices
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
    assert len(set(made.degrees_in)) == len(made.blocks) + int(len(conditioning_indices) > 0)

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
