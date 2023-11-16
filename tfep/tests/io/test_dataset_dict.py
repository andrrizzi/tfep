#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test objects and function in the module ``tfep.io.dataset.dict``.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import numpy as np
import pytest
import torch

from tfep.io.dataset.dict import DictDataset


# =============================================================================
# TEST DICTDATASET
# =============================================================================

def test_conversion():
    """Lists/arrays etc. are converted to tensors."""
    d = DictDataset({
        'a': [0, 1],
        'b': np.array([2, 3]),
        'c': torch.tensor([4, 5])
    })
    sample = d[0]
    samples = d[:]

    for s in [sample, samples]:
        for val in s.values():
            assert isinstance(val, torch.Tensor)


def test_raise_different_length():
    """An error is raised if the dictionary's elements have different lengths."""
    with pytest.raises(ValueError, match='must all have the same length'):
        DictDataset({'a': [0, 1], 'b': [2]})
