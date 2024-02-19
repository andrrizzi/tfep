#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
A PyTorch stateful batch sampler that can be used to correctly resume training mid-epoch.

See documentation of the class :class:`StatefulBatchSampler` for more details.

"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

from typing import Any, Iterator, Optional

import torch


# =============================================================================
# STATEFUL BATCH SAMPLER
# =============================================================================

class StatefulBatchSampler(torch.utils.data.Sampler):
    """A PyTorch stateful batch sampler to resume training mid-epoch.

    This class can be used with a PyTorch Lightning ``Trainer`` to implement
    a correct data checkpointing. If the training is interrupted mid-epoch, and
    the ``DataLoader`` uses this batch sampler, the training will resume correctly,
    i.e., it will complete the epoch by training only on the data points that
    were not previously seen.

    Examples
    --------
    >>> import torch
    >>> import lightning
    >>> import tfep.io
    >>>
    >>> # Initialize the trainer. This must be passed to StatefulBatchSampler.
    >>> trainer = lightning.Trainer()
    >>>
    >>> # Initialize the dataset and data loader.
    >>> dataset = tfep.io.DictDataset({'a': [0, 1, 2, 3, 4]})
    >>> sampler = StatefulBatchSampler(
    ...     dataset,
    ...     batch_size=2,
    ...     shuffle=True,
    ...     drop_last=True,
    ...     trainer=trainer
    ... )
    >>> dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
    >>>
    >>> # Train your model.
    >>> trainer.fit(your_model, dataloader)  # doctest: +SKIP

    """

    def __init__(
            self,
            dataset : torch.utils.data.Dataset,
            batch_size : int = 1,
            shuffle : bool = False,
            drop_last : bool = False,
            trainer = None
    ):
        """Constructor.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            The dataset to sample.
        batch_size : int, optional
            The batch size.
        shuffle : bool, optional
            If ``True``, data samples are reshuffled at every epoch.
        drop_last : bool, optional
            If the dataset size is not divisible by the batch size the last batch
            is dropped if this is ``True`` or just smaller if this is ``False``.
        trainer : object or None, optional
            An object exposing a ``global_step`` attribute that holds the total
            number of seen batches during the entire training. This is usually a
            PyTorch Lightning ``Trainer`` object. If not given on initialization,
            this must be passed later through the :attr:`~StatefulBatchSampler.trainer`
            attribute.

        """
        try:
            super().__init__()
        except TypeError:
            # PyTorch < 2.1 requires passing a dataset but Pytorch > 2.1 requires not passing it.
            super().__init__(dataset)

        self._dataset = dataset
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._drop_last = drop_last

        # Keeps track of the seed used to shuffle the data in the current epoch.
        self._current_epoch_seed = None

        #: The trainer object exposing a ``global_step`` attribute with the total number of batches seen during the entire training.
        self.trainer : Optional[Any] = trainer

    @property
    def batch_size(self) -> int:
        """The batch size."""
        return self._batch_size

    @property
    def shuffle(self) -> bool:
        """Whether to reshuffle the data at each new epoch."""
        return self._shuffle

    @property
    def drop_last(self) -> bool:
        """Whether the last incomplete batch is dropped or yielded."""
        return self._drop_last

    def __len__(self) -> int:
        """The number of batches per epoch."""
        if self.drop_last:
            return len(self._dataset) // self.batch_size
        return (len(self._dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Iterate over batches.

        Yields
        ------
        batch_indices : torch.Tensor[int]
            A tensor of sample indices forming the batch.

        """
        if self.trainer is None:
            raise RuntimeError('trainer must be set before starting the training.')

        # This is usually called at the start of each epoch but current_batch_idx
        # might be != 0 if this is resumed from a mid-epoch checkpoint.
        current_batch_idx = self.trainer.global_step % len(self)

        # Get the random indices.
        if self.shuffle:
            # If this is a new epoch, regenerate the seed.
            if current_batch_idx == 0:
                self._current_epoch_seed = int(torch.empty((), dtype=torch.int64).random_().item())

            # Create a random permutation of the sample indices.
            generator = torch.Generator()
            generator.manual_seed(self._current_epoch_seed)
            epoch_indices = torch.randperm(len(self._dataset), generator=generator)
        else:  # Sequential.
            epoch_indices = torch.arange(0, len(self._dataset), dtype=int)

        # Yield indices.
        for batch_idx in range(current_batch_idx, len(self)):
            start = batch_idx * self.batch_size
            end = (batch_idx + 1) * self.batch_size
            yield epoch_indices[start:end]

    def state_dict(self) -> dict[str, Any]:
        """Serialize the internal state in dictionary format.

        Note that the parameters passed in the constructor are not serialized.

        Returns
        -------
        state_dict : dict[str, Any]
            The serialized internal state.

        """
        return {'current_epoch_seed': self._current_epoch_seed}

    def load_state_dict(self, state_dict: dict[str, Any]):
        """Load the internal state from a dictionary.

        The dictionary must be generated with :func:`~StatefulBatchSampler.state_dict`.
        Note that the parameters passed in the constructor are not serialized,
        and thus the object must be initialized with the same arguments to recover
        the same object.

        Parameters
        ----------
        state_dict : dict[str, Any]
            The serialized internal state.

        """
        self._current_epoch_seed = state_dict['current_epoch_seed']
