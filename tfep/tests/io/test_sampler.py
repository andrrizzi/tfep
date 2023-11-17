#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test objects and function in the module ``tfep.io.sampler``.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import pytest
import torch

from tfep.io.dataset import DictDataset
from tfep.io.sampler import StatefulBatchSampler


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def dataset_len5():
    return DictDataset({
        'x': list(range(5)),
        'y': list(range(5, 10)),
    })


# =============================================================================
# TEST UTILITIES
# =============================================================================

class MockTrainer:
    """A mock trainer holding the global step."""
    def __init__(self):
        self.global_step = 0


# =============================================================================
# TESTS
# =============================================================================

@pytest.mark.parametrize('shuffle', [False, True])
def test_shuffle(dataset_len5, shuffle):
    """The shuffle argument controls the data order."""
    trainer = MockTrainer()
    sampler = StatefulBatchSampler(dataset_len5, batch_size=len(dataset_len5), shuffle=shuffle, trainer=trainer)
    dataloader = torch.utils.data.DataLoader(dataset_len5, batch_sampler=sampler)

    # Check the data order.
    x = next(iter(dataloader))['x']
    sequential = torch.range(0, len(dataset_len5)-1, dtype=int)
    if shuffle:
        assert not torch.equal(x, sequential)
    else:
        assert torch.equal(x, sequential)


@pytest.mark.parametrize('drop_last', [False, True])
def test_drop_last(dataset_len5, drop_last):
    """The drop_last argument works as expected."""
    trainer = MockTrainer()
    sampler = StatefulBatchSampler(dataset_len5, batch_size=2, drop_last=drop_last, trainer=trainer)
    dataloader = torch.utils.data.DataLoader(dataset_len5, batch_sampler=sampler)

    # Count the number of batches.
    for batch in dataloader:
        trainer.global_step += 1

    # Test drop last.
    if drop_last:
        assert len(sampler) == 2
    else:
        assert len(sampler) == 3
    assert trainer.global_step == len(sampler)


def test_resuming(dataset_len5):
    """A new StatefulBatchSampler can correctly recover the data order given its internal state."""
    trainer = MockTrainer()
    sampler = StatefulBatchSampler(dataset_len5, shuffle=True, batch_size=2, drop_last=False, trainer=trainer)
    dataloader = torch.utils.data.DataLoader(dataset_len5, batch_sampler=sampler)

    # Get the samples order and save the state after the first step.
    samples1 = []
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx == 0:
            sampler_state = sampler.state_dict()
        samples1.extend(batch['x'].detach().tolist())

    # Now simulate resuming after the first batch with a new sampler and dataloader.
    sampler = StatefulBatchSampler(dataset_len5, shuffle=True, batch_size=2, drop_last=False, trainer=trainer)
    dataloader = torch.utils.data.DataLoader(dataset_len5, batch_sampler=sampler)

    sampler.load_state_dict(sampler_state)
    trainer.global_step = 1

    samples2 = []
    for batch in dataloader:
        samples2.extend(batch['x'].detach().tolist())

    # In the second run, the first batch is not returned.
    assert len(samples2) == len(samples1) - 2
    assert samples2 == samples1[2:]
