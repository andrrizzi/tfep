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

from tfep.utils.misc import ensure_tensor_sequence, remove_and_shift_sorted_indices


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


@pytest.mark.parametrize('indices,removed_indices,remove,shift,expected', [
    (
        [0, 2, 3, 7],
        [1, 3, 5],
        True, True,
        [0, 1, 4],
    ), (
        [0, 2, 3, 7],
        [1, 3, 5],
        True, False,
        [0, 2, 7],
    ), (
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        True, True,
        [],
    ), (
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        True, False,
        [],
    ), (
        [3, 5, 8, 12, 15],
        [2, 7],
        True, True,
        [2, 4, 6, 10, 13],
    ), (
        [3, 5, 8, 12, 15],
        [2, 7],
        True, False,
        [3, 5, 8, 12, 15],
    ), (
        [3, 5, 8, 12, 15],
        [2, 7],
        False, True,
        [2, 4, 6, 10, 13],
    )
])
def test_remove_and_shift_sorted_indices(
        indices,
        removed_indices,
        remove,
        shift,
        expected,
):
    """Test utility method remove_and_shift_sorted_indices()."""
    out = remove_and_shift_sorted_indices(
        indices=torch.tensor(indices),
        removed_indices=torch.tensor(removed_indices),
        remove=remove,
        shift=shift,
    )
    assert torch.all(torch.tensor(expected) == out)
