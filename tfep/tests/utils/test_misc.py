#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test objects and function in tfep.utils.misc.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import numpy as np
import pytest
import torch

from tfep.utils.misc import ensure_tensor_sequence


# =============================================================================
# TESTS
# =============================================================================

@pytest.mark.parametrize('x,expected', [
    (1, 1),
    (2., 2.),
    (None, None),
    ('string', 'string'),
    ([0, 1, 2], torch.tensor([0, 1, 2])),
    ((2., 1., 0.), torch.tensor([2., 1., 0.])),
    (np.array([3, 4, 5]), torch.tensor([3, 4, 5])),
    (torch.tensor([6, 7, 8]), torch.tensor([6, 7, 8])),
])
def test_ensure_int_tensor_sequence(x, expected):
    """Test method ensure_tensor_sequence."""
    y = ensure_tensor_sequence(x)
    try:
        assert torch.all(y == expected)
        assert isinstance(y, torch.Tensor)
    except TypeError:
        assert y == expected
        assert type(y) == type(expected)
