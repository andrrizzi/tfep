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

import torch


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_random_input(batch_size, n_features, n_parameters=0, dtype=None,
                        seed=0, x_func=torch.randn, par_func=torch.randn):
    """Create input and parameters.

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
