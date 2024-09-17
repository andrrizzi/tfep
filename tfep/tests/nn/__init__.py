
# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test objects and function of the ``tfep.nn`` package.
"""

# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

from collections.abc import Callable, Sequence
from typing import Optional, Union

import numpy as np
import torch


# =============================================================================
# PACKAGE-WIDE TEST UTILITIES
# =============================================================================

def check_autoregressive_property(
        model: torch.nn.Module,
        x: torch.Tensor,
        degrees_in: Sequence[int],
        degrees_out: Sequence[int],
):
    """Raises an ``AssertionError`` if y = model(x) does not satisfy the autoregressive property.

    The test will pass if all the outputs will depend on the inputs such that
    ``degree_in < degree_out``.

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

    # Add batch dimension and make output differentiable w.r.t. input.
    x = x.unsqueeze(0).clone().detach()
    x.requires_grad = True

    # If the model is a flow (not a conditioner), it returns a tuple y, log|det(J)|.
    y = model(x)
    try:
        y.shape
    except AttributeError:
        y, log_det_J = y

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


def create_random_input(
        batch_size: int,
        n_features: int,
        n_parameters: int = 0,
        dtype: Optional[type] = None,
        seed: Optional[int] = None,
        x_func: Callable = torch.randn,
        par_func: Callable = torch.randn,
) -> Union[torch.Tensor, tuple[torch.Tensor]]:
    """Create random input and parameters.

    Parameters
    ----------
    x_func : Callable, optional
        The random function used to generate ``x``. Default is ``torch.randn``.
    par_func : Callable, optional
        The random function used to generate ``parameters``. Default is
        ``torch.randn``.

    Returns
    -------
    x : torch.Tensor
        Shape ``(batch_size, n_features)``. The random input.
    parameters : torch.Tensor, optional
        Shape ``(batch_size, n_parameters)``. The random parameters. This is
        returned only if ``n_parameters > 0``.

    """
    # Make sure the input arguments are deterministic.
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    if dtype is None:
        dtype = torch.get_default_dtype()

    x = x_func(batch_size, n_features, generator=generator,
               dtype=dtype, requires_grad=True)
    returned_values = [x]

    if n_parameters > 0:
        parameters = par_func(batch_size, n_parameters, generator=generator,
                              dtype=dtype, requires_grad=True)
        returned_values.append(parameters)

    if len(returned_values) == 1:
        return returned_values[0]
    return returned_values
