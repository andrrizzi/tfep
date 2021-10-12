#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Utility functions for the tests of the tfep.nn subpackage.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import numpy as np
import torch


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_random_input(batch_size, n_features, n_parameters=0, dtype=None,
                        seed=0, x_func=torch.randn, par_func=torch.randn):
    """Create input, parameters and gates.

    Returns random input data with shape ``(batch_size, n_features)`` and
    parameters with shape ``(batch_size, n_parameters, n_features)`` generated
    from the given ``seed``.

    ``x_func`` and ``par_func`` are the PyTorch functions used to generate the
    random data.

    """
    # Make sure the input arguments are deterministic.
    generator = torch.Generator()
    generator.manual_seed(seed)

    if dtype is None:
        dtype = torch.get_default_dtype()

    x = x_func(batch_size, n_features, generator=generator,
               dtype=dtype, requires_grad=True)
    returned_values = [x]

    if n_parameters > 0:
        parameters = par_func(batch_size, n_parameters, n_features, generator=generator,
                              dtype=dtype, requires_grad=True)
        returned_values.append(parameters)

    if len(returned_values) == 1:
        return returned_values[0]
    return returned_values


# =============================================================================
# REFERENCE IMPLEMENTATIONS
# =============================================================================

def reference_log_det_J(x, y):
    """Compute the log(abs(det(J))) with autograd and numpy.

    ``x`` and ``y`` must be the input and output of the PyTorch function of
    which the Jacobian is computed.
    """
    batch_size, n_features = x.shape

    # Compute the jacobian with autograd.
    jacobian = np.empty((batch_size, n_features, n_features))
    for i in range(n_features):
        loss = torch.sum(y[:, i])
        loss.backward(retain_graph=True)

        jacobian[:, i] = x.grad.detach().numpy()

        # Reset gradient for next calculation.
        x.grad.data.zero_()

    # Compute the log det J numerically.
    log_det_J = np.empty(batch_size)
    for batch_idx in range(batch_size):
        log_det_J[batch_idx] = np.log(np.abs(np.linalg.det(jacobian[batch_idx])))

    return log_det_J

