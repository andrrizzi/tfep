#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test objects and function in the module ``tfep.io.dataset.merged``.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import pytest
import torch
import torch.utils.data

from tfep.io.dataset.merged import MergedDataset


# =============================================================================
# TEST UTILITY FUNCTIONS
# =============================================================================

class DictDataset(torch.utils.data.Dataset):
    """Dataset to merge for the tests below."""
    def __init__(self, tensor_dict):
        self.tensor_dict = tensor_dict

    def __getitem__(self, item):
        return {k: v[item] for k, v in self.tensor_dict.items()}

    def __len__(self):
        for vals in self.tensor_dict.values():
            return len(vals)


# =============================================================================
# TEST MERGED DATASET
# =============================================================================

@pytest.mark.parametrize('sizes', [(2, 3), (2, 2, 3)])
def test_raise_different_dataset_size(sizes):
    """MergedDataset raises an error if the merged datasets have different sizes."""
    # Build the len(sizes) datasets to merge.
    datasets = []
    for dataset_idx, size in enumerate(sizes):
        tensor_dict = {str(dataset_idx): torch.rand(size)}
        datasets.append(DictDataset(tensor_dict))

    with pytest.raises(ValueError, match='different numbers of samples'):
        MergedDataset(*datasets)


@pytest.mark.parametrize('data_keys', [
    [('a', 'b'), ('c', 'b')],
    [('a', 'b'), ('c', 'd'), ('e', 'f', 'a')],
])
def test_raise_overlapping_keys(data_keys):
    """MergedDataset raises an error if the merged datasets have overlapping keys."""
    # Build the len(sizes) datasets to merge.
    datasets = []
    for dataset_idx, keys in enumerate(data_keys):
        tensor_dict = {k: torch.rand(2) for k in keys}
        datasets.append(DictDataset(tensor_dict))

    with pytest.raises(ValueError, match='overlapping keys'):
        MergedDataset(*datasets)

def test_merged_dataset():
    """MergedDataset returns the correct sample from each merged dataset."""
    size = 2

    # Build the datasets to merge.
    data1 = DictDataset(tensor_dict={
        '0': torch.arange(size),
        '1': torch.arange(size)+1,
        '2': torch.arange(size)+2,
    })
    data2 = DictDataset(tensor_dict={
        '3': torch.arange(size)+3,
        '4': torch.arange(size)+4,
    })
    merged = MergedDataset(data1, data2)

    # Check samples.
    for sample_idx in range(size):
        expected = {str(i): torch.tensor(i+sample_idx) for i in range(5)}
        assert merged[sample_idx] == expected
