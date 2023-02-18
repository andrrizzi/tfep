#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test objects and functions in tfep.nn.conditioners.masked.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import numpy as np
import pytest
import torch
import torch.autograd

from tfep.nn.masked import (
    create_autoregressive_mask,
    MaskedLinear,
    masked_linear,
    masked_weight_norm,
)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def normalize_weight_vectors(w):
    w = w.detach()
    vector_norms = torch.tensor([torch.norm(x) for x in w])
    for v_idx, v_norm in enumerate(vector_norms):
        if v_norm != 0:
            w[v_idx] /= v_norm
    return w


def check_wnorm_components(layer, mask):
    masked_weights = layer.weight.detach() * mask.detach()
    expected_g = torch.tensor([[torch.norm(x)] for x in masked_weights])
    expected_normalized_v = normalize_weight_vectors(masked_weights)

    # Compute the normalized v.
    normalized_weight_v = normalize_weight_vectors(layer.weight_v)

    assert torch.allclose(layer.weight_g, expected_g)
    assert torch.allclose(normalized_weight_v, expected_normalized_v)


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
def test_create_autoregressive_mask(dimension_in, n_layers, conditioning_indices):
    """Test the method create_autoregressive_mask().

    Simulate a multi-layer MADE network with sequential degree assignment
    and check that all the masks have the appropriate shape and are upper
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
    masks = [create_autoregressive_mask(degrees[i], degrees[i+1], strictly_less=(i==n_layers-1))
             for i in range(n_layers)]

    for layer_idx, mask in enumerate(masks):
        is_first_mask = layer_idx == 0
        is_last_mask = (layer_idx == n_layers-1)

        # Check mask dimension.
        if is_first_mask:
            assert mask.shape == (first_layer_dim, inner_layer_dim)
        elif is_last_mask:
            assert mask.shape == (inner_layer_dim, output_layer_dim)
        else:
            assert mask.shape == (inner_layer_dim, inner_layer_dim)

        # The conditioning degrees affect all the mapped DOFs.
        if is_first_mask:
            conditioning_mask = mask[conditioning_indices, :]
            mask = mask[mapped_indices, :]
        else:
            conditioning_mask = mask[:n_conditioning_dofs, :]
            mask = mask[n_conditioning_dofs:, :]
        assert torch.all(conditioning_mask == torch.ones(conditioning_mask.shape))

        # Because we assigned the degrees in the order of the input,
        # the mask of the non-conditioning DOFs must be upper triangular.
        if is_last_mask:
            # Contrarily to other layers, the output DOFs don't depend on themselves.
            assert torch.all(mask == torch.triu(mask, diagonal=-1))
        else:
            assert torch.all(mask == torch.triu(mask))

        # In the first layer, the last input unit must have no connection to the
        # first hidden layer since that unit would otherwise not affect any output.
        if is_first_mask:
            assert torch.all(mask[-1, :] == False)


def test_masked_linear_gradcheck():
    """Run autograd.gradcheck on the masked_linear function."""
    batch_size = 2
    in_features = 3
    out_features = 5

    # Normal linear arguments.
    input = torch.randn(batch_size, in_features, dtype=torch.double, requires_grad=True)
    weight = torch.randn(out_features, in_features, dtype=torch.double, requires_grad=True)
    bias = torch.randn(out_features, dtype=torch.double, requires_grad=True)

    # Lower triangular mask.
    mask = torch.tril(torch.ones(out_features, in_features, dtype=torch.double, requires_grad=False))

    # With a None mask, the module should fall back to the native implementation.
    for m in [mask, None]:
        result = torch.autograd.gradcheck(
            func=masked_linear,
            inputs=[input, weight, bias, m]
        )
        assert result


@pytest.mark.parametrize('diagonal', [0])#[0, -1, -2])
@pytest.mark.parametrize('wnorm', [True])#[False, True])
def test_masked_linear_wnorm_compatibility(diagonal, wnorm):
    """Check that training of the masked linear layer is compatible with weight normalization."""
    batch_size = 2
    in_features = 4
    out_features = 5

    # Generate random input. Make sure the test is reproducible.
    generator = torch.Generator()
    generator.manual_seed(0)
    x = torch.randn(batch_size, in_features, generator=generator, requires_grad=True)

    # Lower triangular mask.
    mask = torch.tril(torch.ones(out_features, in_features, requires_grad=False),
                      diagonal=diagonal)

    # Create a weight-normalized masked linear layer.
    masked_linear = MaskedLinear(in_features, out_features, bias=True, mask=mask)
    if wnorm:
        masked_linear = masked_weight_norm(masked_linear, name='weight')

        # The norm and direction vectors are also masked.
        check_wnorm_components(masked_linear, mask)

    # The gradient of the masked parameters should be zero.
    y = masked_linear(x)
    loss = torch.sum(y)
    loss.backward()

    if wnorm:
        assert (masked_linear.weight_g.grad[:abs(diagonal)] == 0).detach().byte().all()
        assert (masked_linear.weight_v.grad * (1 - mask) == 0).detach().byte().all()
    else:
        assert (masked_linear.weight.grad * (1 - mask) == 0).detach().byte().all()

    if wnorm:
        # Simulate one batch update.
        sgd = torch.optim.SGD(masked_linear.parameters(), lr=0.01, momentum=0.9)
        sgd.step()

        # Make a forward pass so that the wnorm wrapper will update masked_linear.weight
        masked_linear(x)

        # Check that g and v are still those expected.
        check_wnorm_components(masked_linear, mask)
