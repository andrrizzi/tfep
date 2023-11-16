#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

r"""
Utility class to create a map-style PyTorch ``Dataset``\ s from a dictionary of tensors.

For usage examples see the documentation of :class:`.DictDataset`.

"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

from collections.abc import Sequence

import torch.utils.data


# =============================================================================
# DICTIONARY DATASET
# =============================================================================

class DictDataset(torch.utils.data.Dataset):
    r"""Utility class to create a map-style PyTorch ``Dataset``\ s from a dictionary of tensors.

    The class automatically converts non-tensor dictionary values into tensors.

    Examples
    --------
    >>> import torch
    >>> data = {'a': torch.tensor([1.0, 2.0]), 'b': [3, 4]}
    >>> dict_dataset = DictDataset(data)
    >>> dict_dataset[1]
    {'a': tensor(2.), 'b': tensor(4)}

    """

    def __init__(self, tensor_dict : dict[str, Sequence]):
        """Constructor.

        Parameters
        ----------
        tensor_dict : dict[str, torch.Tensor]
            A dictionary of named tensors.

        """
        # Check that all the columns have the same lengths.
        lengths = set(len(v) for v in tensor_dict.values())
        if len(lengths) > 1:
            raise ValueError('The values of tensor_dict must all have the same length.')

        # Convert all values to tensors.
        self._tensor_dict = {}
        for k, v in tensor_dict.items():
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v)
            self._tensor_dict[k] = v

    def __getitem__(self, item):
        """Retrieve a dataset sample.

        Parameters
        ----------
        item : int or slice
            The index (or slice) of the sample(s).

        Returns
        -------
        samples : dict[str, torch.Tensor]
            A dictionary of named tensor.
        """
        return {k: v[item] for k, v in self._tensor_dict.items()}

    def __len__(self):
        """The number of samples in the dataset."""
        for vals in self._tensor_dict.values():
            return len(vals)
