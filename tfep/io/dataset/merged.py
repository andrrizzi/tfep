#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Utility class to merge multiple PyTorch ``Dataset``s.

For usage examples see the documentation of :class:`.MergedDataset`.

"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import copy

import numpy as np
import pint
import torch.utils.data


# =============================================================================
# MERGED DATASET
# =============================================================================

class MergedDataset(torch.utils.data.Dataset):
    """Dataset merging multiple ``Dataset``s.

    The dataset constructs a batch by merging batches from the wrapped datasets.
    Currently, it supports only map-style datasets that return samples in
    dictionary format.

    Parameters
    ----------
    *datasets : torch.utils.data.Dataset
        The map-style datasets to be merged. These must all have the same number
        of samples and return samples in dictionary format.

    """
    def __init__(self, *datasets):
        super().__init__()

        # Check that the datasets all have the same number of samples.
        for i in range(len(datasets)-1):
            if len(datasets[i]) != len(datasets[i+1]):
                raise ValueError(f'Datasets {i} and {i+1} have different numbers '
                                 f'of samples ({len(datasets[i])} and {len(datasets[i+1])})')

        # Check that the datasets have different keys so that no data is overridden.
        n_keys = 0
        all_keys = set()
        for dataset in datasets:
            keys = list(dataset[0].keys())
            n_keys += len(keys)
            all_keys.update(keys)
        if len(all_keys) != n_keys:
            raise ValueError(f'The merged datasets have overlapping keys.')

        # We save the datasets as an internal attribute because we don't perform
        # any other checks if the datasets are modified.
        self._datasets = datasets

    def __getitem__(self, item):
        sample = {}
        for dataset in self._datasets:
            # We have already calculated in __init__ that keys in different
            # datasets do not overlap.
            sample.update(dataset[item])
        return sample