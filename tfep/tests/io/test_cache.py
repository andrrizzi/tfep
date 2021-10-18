#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test objects and function in the module ``tfep.io.cache``.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import tempfile

import numpy as np
import pytest
import torch

from tfep.io.cache import TFEPCache


# =============================================================================
# GLOBAL VARIABLES
# =============================================================================

# Random number generator. Makes sure tests are reproducible from run to run.
GENERATOR = torch.Generator()
GENERATOR.manual_seed(0)

# =============================================================================
# TEST UTILITIES
# =============================================================================

class DummyTrajectoryDataset(torch.utils.data.Dataset):
    """Dummy TrajectoryDataset for unit testing."""

    def __init__(self, n_frames=14, n_dofs=9):
        # Make sure the tests are reproducible from run to run.
        torch.manual_seed(0)
        self.data = torch.randn(n_frames, n_dofs, generator=GENERATOR)
        self.trajectory_sample_indices = torch.randint(
            high=n_frames*5, size=(n_frames,), generator=GENERATOR)

    def __getitem__(self, index):
        return {'positions': self.data[index], 'dataset_sample_index': index}

    def __len__(self):
        return len(self.data)


def check_all_and_named_train_tensors(cache, read_data, **kwargs):
    """
    Check whether reading all tensors or specific tensors results in the same behavior.
    Finally, assigns the read data to ``read_data``, which is a dict name -> tensor with
    shape (n_epochs, n_steps_per_epochs).
    """
    # Read all tensors at the same time.
    data_all = cache.read_train_tensors(**kwargs)

    # Find epoch/batch for assignment.
    if 'step_idx' in kwargs:
        epoch_idx, batch_idx = divmod(kwargs['step_idx'], cache.n_batches_per_epoch)
    else:
        epoch_idx = kwargs['epoch_idx']
        batch_idx = kwargs.get('batch_idx', None)

    # Compare it to reading one tensor at a time
    for name in data_all:
        data_single_tensor = cache.read_train_tensors(names=[name], **kwargs)
        assert np.allclose(data_single_tensor[name], data_all[name])

        # Assign it to pre-allocated data array.
        if batch_idx is None:
            read_data[name][epoch_idx] = data_single_tensor[name]
        else:
            first = batch_idx * cache.batch_size
            last = first + cache.batch_size
            read_data[name][epoch_idx, first:last] = data_single_tensor[name]


# =============================================================================
# TESTS
# =============================================================================

def test_tfep_cache_init_no_data_loader():
    """An error is raised if a new TFEPCache is initialized without DataLoader."""
    with pytest.raises(ValueError, match='data_loader'):
        TFEPCache(save_dir_path='tmp')


@pytest.mark.parametrize('drop_last', [False, True])
def test_tfep_cache_metadata(drop_last):
    """Metadata is read from DataLoader and file correctly."""
    # Create data loader to test where n_frames is not divisible by batch_size.
    n_frames, batch_size = 9, 4
    data_loader = torch.utils.data.DataLoader(
        DummyTrajectoryDataset(n_frames=n_frames),
        batch_size=batch_size,
        drop_last=drop_last
    )

    # n_samples_per_epoch depends on the value of drop_last.
    if drop_last:
        n_samples_per_epoch = n_frames - n_frames%batch_size
    else:
        n_samples_per_epoch = n_frames

    with tempfile.TemporaryDirectory() as tmp_dir_path:
        # Test loading of metadata from DataLoader.
        cache = TFEPCache(save_dir_path=tmp_dir_path, data_loader=data_loader)
        assert cache.batch_size == batch_size
        assert cache.n_samples_per_epoch == n_samples_per_epoch

        # Now load from disk.
        cache = TFEPCache(save_dir_path=tmp_dir_path)
        assert cache.batch_size == batch_size
        assert cache.n_samples_per_epoch == n_samples_per_epoch


def test_tfep_cache_save_tensors_no_indices():
    """A warning is raised if samples indices are not passed to save_X_tensors."""
    with tempfile.TemporaryDirectory() as tmp_dir_path:
        cache = TFEPCache(
            save_dir_path=tmp_dir_path,
            data_loader=torch.utils.data.DataLoader(DummyTrajectoryDataset())
        )

        # Training data.
        with pytest.warns(UserWarning, match='trajectory_sample_index'):
            cache.save_train_tensors({'test': torch.randn(4, generator=GENERATOR)}, step_idx=0)

        # Evaluation data.
        with pytest.warns(UserWarning, match='trajectory_sample_index'):
            cache.save_eval_tensors({'test': torch.randn(4, generator=GENERATOR)}, step_idx=0)


@pytest.mark.parametrize('n_frames,batch_size', [(9, 3), (9, 2)])
@pytest.mark.parametrize('drop_last', [False, True])
@pytest.mark.parametrize('resume_train', [False, True])
def test_tfep_cache_save_read_train_tensors(n_frames, batch_size, drop_last, resume_train):
    """Simulate training for a few epochs, save and read everything.

    If resume_train is True, the cache will be reinizialized in the midst
    of the epoch (in each epoch) to simulate resuming training.
    """
    n_epochs = 2

    dataset = DummyTrajectoryDataset(n_frames=n_frames)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, drop_last=drop_last, shuffle=True)

    # The batch index at which to reinizialize the cache if resume_train is True.
    reinit_batch_idx = len(data_loader) // 2

    with tempfile.TemporaryDirectory() as tmp_dir_path:
        cache = TFEPCache(save_dir_path=tmp_dir_path, data_loader=data_loader)

        # This is the data we'll be writing.
        ref_data = {k: torch.empty(n_epochs, cache.n_samples_per_epoch)
                    for k in ['dataset_sample_index', 'mean']}

        # Simulate training for 2 epochs.
        for epoch_idx in range(n_epochs):
            for batch_idx, batch_data in enumerate(data_loader):
                # Generate some fake data to store.
                batch_means = torch.mean(batch_data['positions'], dim=1) + epoch_idx

                # Reinitialize the cache to test resuming.
                if resume_train and (batch_idx == reinit_batch_idx):
                    cache = TFEPCache(save_dir_path=tmp_dir_path)

                # Store data using step.
                saved_tensors = {
                    'dataset_sample_index': batch_data['dataset_sample_index'],
                    'mean': batch_means
                }
                cache.save_train_tensors(
                    tensors=saved_tensors,
                    step_idx=epoch_idx*len(data_loader) + batch_idx
                )

                # Keep track of the data for testing later.
                first = batch_idx * batch_size
                last = first + batch_size
                ref_data['dataset_sample_index'][epoch_idx, first:last] = batch_data['dataset_sample_index']
                ref_data['mean'][epoch_idx, first:last] = batch_means

        # Reconstruct all data by reading it from file. We read by epoch,
        # epoch/batch or step; all tensors at the same time or specific tensors.
        # We will compare all the reconstructed data against the input at the end.

        # First, read one epoch at a time.
        data_to_test = [{k: torch.empty_like(v) for k, v in ref_data.items()}]
        for epoch_idx in range(n_epochs):
            check_all_and_named_train_tensors(
                cache, read_data=data_to_test[-1], epoch_idx=epoch_idx)

        # Read one batch at a time.
        data_to_test.append({k: torch.empty_like(v) for k, v in ref_data.items()})
        for epoch_idx in range(n_epochs):
            for batch_idx in range(len(data_loader)):
                check_all_and_named_train_tensors(
                    cache, read_data=data_to_test[-1], epoch_idx=epoch_idx, batch_idx=batch_idx)

        # Read one step at a time.
        data_to_test.append({k: torch.empty_like(v) for k, v in ref_data.items()})
        n_steps = len(data_loader) * n_epochs
        for step_idx in range(n_steps):
            check_all_and_named_train_tensors(
                cache, read_data=data_to_test[-1], step_idx=step_idx)

        # Compare all reconstructed data with reference.
        for read_data in data_to_test:
            for name, ref_tensor in ref_data.items():
                assert np.allclose(read_data[name], ref_tensor)

        # The returned type must be the same as those saved.
        cache = TFEPCache(tmp_dir_path)
        for epoch_idx in range(n_epochs):
            read_data = cache.read_train_tensors(epoch_idx=0)
            for name in read_data:
                assert read_data[name].dtype == saved_tensors[name].dtype


def test_tfep_cache_mask():
    """When not all batch have been saved, reading the TFEPCache returns only the defined values."""
    dataset = DummyTrajectoryDataset(n_frames=6)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, drop_last=False, shuffle=True)

    with tempfile.TemporaryDirectory() as tmp_dir_path:
        cache = TFEPCache(save_dir_path=tmp_dir_path, data_loader=data_loader)

        # Save some data for the last batch.
        cache.save_train_tensors({'dataset_sample_index': torch.tensor([4, 5])},
                                 epoch_idx=1, batch_idx=2)
        read = cache.read_train_tensors(epoch_idx=1)
        assert torch.all(read['dataset_sample_index'] == torch.tensor([4, 5]))

        # Save also the first batch.
        cache.save_train_tensors({'dataset_sample_index': torch.tensor([0, 1])},
                                 epoch_idx=1, batch_idx=0)
        read = TFEPCache(tmp_dir_path).read_train_tensors(epoch_idx=1)
        assert torch.all(read['dataset_sample_index'] == torch.tensor([0, 1, 4, 5]))

        # Finally the second batch.
        cache.save_train_tensors({'dataset_sample_index': torch.tensor([2, 3])},
                                 epoch_idx=1, batch_idx=1)
        read = cache.read_train_tensors(epoch_idx=1)
        assert torch.all(read['dataset_sample_index'] == torch.tensor([0, 1, 2, 3, 4, 5]))


@pytest.mark.parametrize('n_frames,batch_size', [(9, 3), (9, 2)])
@pytest.mark.parametrize('resume_eval', [False, True])
def test_tfep_cache_save_read_eval_tensors(n_frames, batch_size, resume_eval):
    """Simulate evaluating a model, save and read everything.

    If resume_eval is True, the cache will be reinizialized in the midst
    of the epoch (in each epoch) to simulate resuming training.
    """
    # This can be a random number since it doesn't refer to a real model.
    step_idx = 2

    # Create test dataset.
    dataset = DummyTrajectoryDataset(n_frames=n_frames)

    # We evaluate the samples in the dataset in random order since
    # the result should be sorted when we'll read the data.
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True)

    # The batch index at which to reinizialize the cache if resume_eval is True.
    reinit_batch_idx = len(data_loader) // 2

    with tempfile.TemporaryDirectory() as tmp_dir_path:
        cache = TFEPCache(save_dir_path=tmp_dir_path, data_loader=data_loader)

        # Generate the reference data.
        ref_data = {
            'dataset_sample_index': torch.Tensor(list(range(n_frames))),
            'mean': torch.Tensor([torch.mean(b['positions']) for b in dataset]),
        }

        for batch_idx, batch_data in enumerate(data_loader):
            # Reinitialize the cache to test resuming.
            if resume_eval and (batch_idx == reinit_batch_idx):
                cache = TFEPCache(save_dir_path=tmp_dir_path)

            # Generate some fake data to store.
            saved_tensors = {
                'dataset_sample_index': batch_data['dataset_sample_index'],
                'mean': torch.mean(batch_data['positions'], dim=1)
            }
            cache.save_eval_tensors(
                tensors=saved_tensors,
                step_idx=step_idx
            )

        # Read the data from a file with the same cache and with a brand new one.
        for cache in [cache, TFEPCache(save_dir_path=tmp_dir_path)]:
            read_data = cache.read_eval_tensors(step_idx=step_idx, sort_by='dataset_sample_index')

            # Reading the data of named tensors should be identical.
            # Internally the data should be already sorted so no need to use sort_by again.
            for name in ref_data:
                named_data = cache.read_eval_tensors(step_idx=step_idx)
                assert np.allclose(read_data[name], named_data[name])
                assert np.allclose(read_data[name], ref_data[name])
                # The returned type must be the same as those saved.
                assert read_data[name].dtype == saved_tensors[name].dtype

        # Now writing data with the same indices overwrite previous values.
        changed_indices = torch.Tensor(np.array([0, 3, 4]))
        cache.save_eval_tensors(
            tensors={'dataset_sample_index': changed_indices, 'mean': changed_indices},
            step_idx=step_idx, update=True
        )

        read_data = TFEPCache(save_dir_path=tmp_dir_path).read_eval_tensors(step_idx=step_idx)
        assert np.allclose(read_data['dataset_sample_index'], ref_data['dataset_sample_index'])
        assert np.allclose(read_data['mean'][changed_indices.to(torch.int64)], changed_indices)
