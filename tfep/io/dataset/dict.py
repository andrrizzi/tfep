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

import torch.utils.data


# =============================================================================
# MERGED DATASET
# =============================================================================

class DictDataset(torch.utils.data.Dataset):
    r"""Utility class to create a map-style PyTorch ``Dataset``\ s from a dictionary of tensors.

    Examples
    --------
    >>> import torch
    >>> data = {'a': torch.tensor([1.0, 2.0]), 'b': torch.tensor([3, 4])}
    >>> dict_dataset = DictDataset(data)
    >>> dict_dataset[1]
    {'a': tensor(2.), 'b': tensor(4)}

    """

    def __init__(self, tensor_dict : dict[str, torch.Tensor]):
        """Constructor.

        Parameters
        ----------
        tensor_dict : dict[str, torch.Tensor]
            A dictionary of named tensors.

        """
        self.tensor_dict = tensor_dict

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
        return {k: v[item] for k, v in self.tensor_dict.items()}

    def __len__(self):
        """The number of samples in the dataset."""
        for vals in self.tensor_dict.values():
            return len(vals)
