
# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test objects and function of the ``tfep.nn`` package.
"""

# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import numpy as np
import torch


# =============================================================================
# PACKAGE-WIDE TEST UTILITIES
# =============================================================================

def check_autoregressive_property(model, x, degrees_in, degrees_out):
    """Raises an ``AssertionError`` if y = model(x) does not satisfy the autoregressive property.

    Parameters
    ----------
    model : torch.nn.Module
        The model.
    x : torch.Tensor
        Shape (n_features_in,). Input.
    degrees_in : Sequence[int]
        Shape (n_features_in,). Degrees for the input features.
    degrees_out : Sequence[int]
        Shape (n_features_out,). Degrees for the output features.

    """
    # Convert to list.
    try:
        degrees_in = degrees_in.tolist()
    except AttributeError:
        pass
    try:
        degrees_out = degrees_out.tolist()
    except AttributeError:
        pass

    # Identify conditioning and mapped features.
    conditioning_mask = np.array(degrees_in) == -1
    conditioning_indices = np.where(conditioning_mask)[0]
    mapped_indices = np.where(~conditioning_mask)[0]

    # Add batch dimension and compute output.
    x = x.unsqueeze(0)
    x.requires_grad = True
    y = model(x)

    # Check degree by degree.
    for degree_out in set(degrees_out):
        # Compute the gradient of the output considering only the output features
        # associated to that degree.
        degree_indices = np.where(np.array(degrees_out) == degree_out)[0]
        loss = torch.sum(y[0, degree_indices])
        loss.backward(retain_graph=True)

        # In all cases, the conditioning features should affect the whole output.
        grad = x.grad[0]
        assert torch.all(grad[conditioning_indices] != 0.0)

        # Now consider the non-conditioning features only.
        for idx in mapped_indices:
            degree_in = degrees_in[idx]
            if degree_in < degree_out:
                assert grad[idx] != 0.
            else:
                assert grad[idx] == 0.

        # Reset gradients for next iteration.
        model.zero_grad()
        x.grad.data.zero_()
