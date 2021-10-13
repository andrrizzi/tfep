#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Utility classes to store potential energies and CVs necessary for TFEP.

The module provides a class :class:`.TFEPCache` that provides an interface to
store on/read from disk the potentials energies and CVs quantities that enter
the TFEP equations.

"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import json
import os
import warnings

import numpy as np
import torch


# =============================================================================
# TRAJECTORY DATASET
# =============================================================================

class TFEPCache:
    """Store and retrieve potential energies and CVs during training/evaluations.

    The user can use this to easily store and retrieve arbitrary per-sample
    quantities such as potential energies and CV values by epoch, batch, or step.

    .. warning::

        Currently, this class does is not multi-process or thread safe.

    Current database format
    -----------------------

    Currently, the all data is stored in compressed numpy format with a different
    format depending on whether the data was generated during training set or
    evaluation. In both cases, the data is store as an ``.npz`` numpy compressed
    archive of named numpy 1D arrays. However, training and evaluation data differ
    in the array dimensions and file naming.

    For training, each ``.npz`` file is saved in a ``train/`` subdirectory with
    name ``epoch-X.npz``, where ``X`` correspond to the training epoch index used
    for the data. Each array in the archive has length ``n_samples_per_epoch``,
    whose value takes into account whether ``drop_last`` is set in the PyTorch
    ``DataLoader``). In each array in the archive ``archive['name'][i]`` is the
    quantity corresponding to the ``i%batch_size`` data point in the
    ``i//batch_size``-th batch.

    For the evaluation data, each ``.npz`` file is saved in a ``eval/`` subdirectory
    with name ``step-X.npz``, which correspond to the quantities evaluated using
    the neural network optimized for ``X`` steps.

    Finally, a JSON file is used to store metadata about the experiment such as
    batch and epoch sizes.

    Parameters
    ----------
    save_dir_path : str, optional
        The main directory where to save the training and evaluation data. If not
        given, it defaults to the current working directory.
    data_loader : torch.utils.data.DataLoader, optional
        The data loader used for training wrapping a :class:``tfep.io.dataset.TrajectoryDataset``.
        This must be passed when a new cache is created as it is used to determine
        epoch, batch, and trajectory dimensions. If ``save_dir_path`` points to
        an existing cache, then this is ignored.
    train_subdir_name : str, optional
        The name of the subdirectory where the training data is stored.
    eval_subdir_name : str, optional
        The name of the subdirectory where the evaluation data is stored.

    """

    VERSION = '0.1'
    METADATA_FILE_NAME = 'metadata.json'
    INDEX_NAMES = ['trajectory_sample_index', 'dataset_sample_index']

    def __init__(
            self,
            save_dir_path='tfep_data',
            data_loader=None,
            train_subdir_name='train',
            eval_subdir_name='eval',
    ):
        self._save_dir_path = os.path.realpath(save_dir_path)
        self._train_dir_path = os.path.join(save_dir_path, train_subdir_name)
        self._eval_dir_path = os.path.join(save_dir_path, eval_subdir_name)

        # This keep track of the currently in-memory training/evaluation data.
        self._loaded_train_idx = None
        self._loaded_train_data = None  # Dict[name, Tensor].
        self._loaded_eval_idx = None
        self._loaded_eval_data = None
        self._loaded_eval_sample_indices_set = None

        # Determine whether this is a new cache or we are resuming. The
        # metadata file is the last file that is created in __init__().
        metadata_file_path = os.path.join(save_dir_path, self.METADATA_FILE_NAME)
        resume = os.path.isfile(metadata_file_path)

        # Load metadata.
        if resume:
            self._metadata_from_file(metadata_file_path)
        elif data_loader is None:
            raise ValueError("When creating a new cache, 'data_loader' must be passed.")
        else:
            self._metadata_from_data(data_loader)

        # Create directory structure.
        os.makedirs(save_dir_path, exist_ok=True)
        for dir_path in [self._train_dir_path, self._eval_dir_path]:
            os.makedirs(dir_path, exist_ok=True)

        # Save sample to trajectory indices map and metadata.
        # TODO: Save trajectory indices.

        # Save metadata. This must be the very last thing to do in __init__ for
        # resuming to work. No need to write on disk if file is already there.
        if not resume:
            self._save_metadata(metadata_file_path)

    @property
    def batch_size(self):
        """The batch size of the training dataset."""
        return self._batch_size

    @property
    def n_samples_per_epoch(self):
        """The number of samples per training epoch.

        This may be equal to the dataset size, depending on the value of the
        ``drop_last`` option in ``DataLoader``.

        """
        return self._n_samples_per_epoch

    @property
    def n_batches_per_epoch(self):
        """the number of batches per training epoch."""
        return int(np.ceil(self.n_samples_per_epoch / self.batch_size))

    def read_eval_tensors(
            self,
            names=None,
            step_idx=None,
            epoch_idx=None,
            batch_idx=None,
            sort_by=None,
            as_numpy=False,
    ):
        """Read the tensors generated with the NN model trained for the given number of epoch/batch/step.

        Either ``step_idx`` or both ``epoch_idx`` and ``batch_idx`` must be passed.

        Parameters
        ----------
        names : List[str], optional
            If given, only the tensors saved with the names in this list are
            returned. Otherwise, all the saved tensors for this step/epoch/batch
            are returned.
        step_idx : int, optional
            If given, the tensors for this optimization step are returned.
        epoch_idx : int, optional
            If given, the tensors for this epoch are returned. If ``step_idx``
            is passed, this is ignored.
        batch_idx : int, optional
            If given together with ``epoch_idx``, the tensors for this epoch/batch
            are returned. If ``step_idx`` is passed, this is ignored.
        sort_by : str, optional
            If given, all the returned tensors will be sorted based on the tensor
            with this name (useful if ``sort_by`` is ``'trajectory_sample_index'``).
            The new data order is also stored on disk thus subsequent calls without
            saving new data are guaranteed to follow in the same order.
        as_numpy : bool, optional
            If ``True``, the tensors are returned as a numpy array rather than
            PyTorch ``Tensors``.

        Returns
        -------
        tensors : Dict[str, torch.Tensor]
            A dictionary mapping the name of the saved tensors to their values.

        """
        # Validate input arguments.
        step_idx, _, _ = self._validate_indices(
            step_idx, epoch_idx, batch_idx, need_batch=True)

        # Load in memory the data of this NN.
        self._load_eval_data(step_idx)

        # Sort the data if requested.
        if sort_by is not None:
            sort_indices = np.argsort(self._loaded_eval_data[sort_by])
            self._loaded_eval_data = {k: v[sort_indices] for k, v in self._loaded_eval_data.items()}

            # Update the data on disk.
            self._dump_data(type='eval')

        # Convert to tensors.
        if as_numpy:
            data = self._loaded_eval_data
        else:
            data = {k: torch.Tensor(v) for k, v in self._loaded_eval_data.items()}

        return data

    def read_train_tensors(
            self,
            names=None,
            step_idx=None,
            epoch_idx=None,
            batch_idx=None,
            as_numpy=False
    ):
        """Read the tensors saved during the given training epoch/batch/step.

        At least one between ``step_idx`` and ``epoch_idx`` must be passed.

        Parameters
        ----------
        names : List[str], optional
            If given, only the tensors saved with the names in this list are
            returned. Otherwise, all the saved tensors for this step/epoch/batch
            are returned.
        step_idx : int, optional
            If given, the tensors for this optimization step are returned.
        epoch_idx : int, optional
            If given, the tensors for this epoch are returned. If ``step_idx``
            is passed, this is ignored.
        batch_idx : int, optional
            If given together with ``epoch_idx``, the tensors for this epoch/batch
            are returned. If ``step_idx`` is passed, this is ignored.
        as_numpy : bool, optional
            If ``True``, the tensors are returned as a numpy array rather than
            PyTorch ``Tensors``.

        Returns
        -------
        tensors : Dict[str, torch.Tensor]
            A dictionary mapping the name of the saved tensors to their values.

        """
        # Check input arguments.
        _, epoch_idx, batch_idx = self._validate_indices(
            step_idx, epoch_idx, batch_idx, need_batch=False)

        # Load in memory the data of this epoch.
        self._load_data(epoch_idx, type='train')

        # Determine which names to load.
        if names is None:
            names = self._loaded_train_data.keys()

        # Return data.
        tensors = {}
        for name in names:
            if batch_idx is None:
                # Collect data for entire epoch.
                tensors[name] = self._loaded_train_data[name]
            else:
                # Collect data for a single batch.
                first = self.batch_size * batch_idx
                last = first + self.batch_size
                tensors[name] = self._loaded_train_data[name][first:last]

            # Convert from numpy to Tensors.
            if not as_numpy:
                tensors[name] = torch.Tensor(tensors[name])

        return tensors

    def save_eval_tensors(
            self,
            tensors,
            step_idx=None,
            epoch_idx=None,
            batch_idx=None,
            update=False,
    ):
        """Save the tensors generated with the NN model trained for the given number of epoch/batch/step.

        Either ``step_idx`` or both ``epoch_idx`` and ``batch_idx`` must be passed.

        Currently, saving only some of the tensors already on disk is not supported.
        In other words, the tensor names that will be saved in the first call to
        ``save_eval_tensors`` for the given step will have to be in all subsequent
        calls.

        .. warning::

            By default, no check is performed on writing twice data for the same
            sample indices and the data is simply appended to the existing one.
            This check is performed only if  ``update`` is set ``True`` and the
            existing data is overwritten. This check is based on the tensors
            named (in order of priority) ``'trajectory_sample_indices'`` and
            ``'dataset_sample_indices'``. Note that this might be an expensive
            operation and should not be used if not necessary.

        Parameters
        ----------
        tensors : Dict[str, torch.Tensor]
            A dictionary mapping the name of the saved tensors to their values.
        step_idx : int or None
            If given, the tensors for this optimization step are saved.
        epoch_idx : int or None
            If given, the tensors for this epoch are saved.
        batch_idx : int or None
            If given together with ``epoch_idx``, the tensors for this epoch/batch
            are saved. Otherwise, the data is assumed to be for the entire epoch.
        update : bool, optional
            If ``True``, data points corresponding to already stored sample
            indices are updated (this slows down the method). If ``False``, this
            check is not performed and all tensors are simply added to the cache.

        """
        # Warn the user about missing dataset_sample_index nor trajectory_sample_index.
        self._warn_if_no_indices(tensors)

        # Validate input arguments.
        step_idx, _, _ = self._validate_indices(
            step_idx, epoch_idx, batch_idx, need_batch=True)

        # Load in memory the data of this NN.
        self._load_eval_data(step_idx)

        # Make sure all known tensors are updated.
        if len(self._loaded_eval_data) == 0:
            names = tensors.keys()
        else:
            names = self._loaded_eval_data.keys()

        # Convert everything to numpy arrays.
        try:
            tensors = {n: tensors[n].detach().numpy() for n in names}
        except KeyError:
            raise KeyError("'tensors' must include all the following Tensors: " + str(list(names)))

        # Check if we need to update some entries (priority to trajectory, then
        # dataset indices). If we don't find anything, we append all data.
        if update:
            # Find the indices and index name.
            for index_name in self.INDEX_NAMES:
                try:
                    tensor_sample_indices = tensors[index_name]
                except KeyError:
                    continue

                # Find the common elements.
                _, tensor_indices, loaded_indices = np.intersect1d(
                    tensor_sample_indices, self._loaded_eval_data[index_name],
                    assume_unique=True, return_indices=True
                )

                # Update the elements in data and remove them from tensors so
                # that the remaining ones will get appended.
                if len(tensor_indices) == 0:
                    break

                for name in names:
                    self._loaded_eval_data[name][loaded_indices] = tensors[name][tensor_indices]
                    tensors[name] = np.delete(tensors[name], tensor_indices)

        # Update all tensors.
        for name in names:
            # Convert from tensor to array.
            try:
                value = tensors[name]
            except KeyError:
                raise KeyError("'tensors' must include a Tensor named " + name)

            # Append to current data.
            try:
                current_arr = self._loaded_eval_data[name]
            except KeyError:
                self._loaded_eval_data[name] = value
            else:
                self._loaded_eval_data[name] = np.concatenate((current_arr, value))

        # Update cached file on disk.
        self._dump_data(type='eval')


    def save_train_tensors(self, tensors, step_idx=None, epoch_idx=None, batch_idx=None):
        """Save the tensors generated during the given epoch/batch/step of training.

        At least one between ``step_idx`` and ``epoch_idx``/``batch_idx`` must
        be passed.

        Parameters
        ----------
        tensors : Dict[str, torch.Tensor]
            A dictionary mapping the name of the saved tensors to their values.
        step_idx : int or None
            If given, the tensors for this optimization step are saved.
        epoch_idx : int or None
            If given, the tensors for this epoch are saved.
        batch_idx : int or None
            If given together with ``epoch_idx``, the tensors for this epoch/batch
            are saved. Otherwise, the data is assumed to be for the entire epoch.

        """
        # Warn the user about missing dataset_sample_index nor trajectory_sample_index.
        self._warn_if_no_indices(tensors)

        # Validate input arguments.
        _, epoch_idx, batch_idx = self._validate_indices(
            step_idx, epoch_idx, batch_idx, need_batch=False)

        # Load in memory the data of this epoch.
        self._load_data(epoch_idx, type='train')

        # This is the first index pointing to the start of the batch.
        if batch_idx is not None:
            first = self.batch_size*batch_idx

        # Update all tensors.
        for name, value in tensors.items():
            # Convert from tensor to array.
            value = value.detach().numpy()

            if batch_idx is None:
                # Assume data is for the entire epoch.
                self._loaded_train_data[name] = value
            else:
                # Initialize the tensor if this is the first time we see it.
                try:
                    saved_array = self._loaded_train_data[name]
                except KeyError:
                    saved_array = np.full(self.n_samples_per_epoch, fill_value=np.nan)
                    self._loaded_train_data[name] = saved_array
                saved_array[first:first+len(value)] = value

        # Update cached file on disk.
        self._dump_data(type='train')

    # --------------------- #
    # Private class members #
    # --------------------- #

    @classmethod
    def _warn_if_no_indices(cls, tensors):
        """Raise a warning if tensors does not contain dataset or trajectory sample indices."""
        for index_name in cls.INDEX_NAMES:
            if index_name in tensors:
                return  # Found.
        warnings.warn(("tensors does not contain any sample indices among the "
                       "following attributes: {}. Without it, it might be "
                       "difficult to match training and evaluation configurations "
                       "to their reference potential.").format(cls.INDEX_NAMES))

    def _get_data_file_path(self, type):
        """The file path where the currently loaded training/evaluation data is stored.

        type can be 'train' or 'eval'
        """
        if type == 'train':
            file_name = 'epoch-' + str(self._loaded_train_idx)
            dir_path = self._train_dir_path
        else:
            file_name = 'step-'+str(self._loaded_eval_idx)
            dir_path = self._eval_dir_path
        return os.path.join(dir_path, file_name + '.npz')

    def _dump_data(self, type):
        """Dump on disk the currently loaded training data."""
        data_attr = '_loaded_' + type + '_data'  # e.g., _loaded_train_data
        file_path = self._get_data_file_path(type=type)
        np.savez_compressed(file_path, **getattr(self, data_attr))

    def _load_data(self, idx, type):
        """Load/initialize the training/evaluation data in memory.

        type can be 'train' or 'eval'.
        """
        # Check if we have already loaded the data.
        idx_attr = '_loaded_' + type + '_idx'  # e.g., _loaded_train_idx
        if getattr(self, idx_attr) == idx:
            return

        # Point to new data file.
        setattr(self, idx_attr, idx)

        # Check if there is data on disk.
        data_attr = '_loaded_' + type + '_data'  # e.g., _loaded_train_data
        file_path = self._get_data_file_path(type)
        if os.path.isfile(file_path):
            # NpzFile offers lazy loading, but we load everything into memory
            # for now to avoid having to deal with correctly closing the file.
            npz_file = np.load(file_path)
            setattr(self, data_attr, {k: v for k, v in npz_file.items()})
        else:
            # Initialize the data for this epoch.
            setattr(self, data_attr, {})

    def _load_eval_data(self, step_idx):
        """Load/initialize the evaluation data for the given NN model.

        On top of loading the npz archive, we cache a set of the indices already
        loaded that we can check on save.
        """
        self._load_data(step_idx, type='eval')

        # Build a set of sample indices (trajectory has priority over dataset).
        indices = None
        for index_name in self.INDEX_NAMES:
            try:
                indices = self._loaded_eval_data[index_name]
            except:
                pass

        # If there are no indices we have already raised a warning in save_eval_tensors.
        if indices is not None:
            self._loaded_eval_sample_indices_set = set(indices)
        else:
            self._loaded_eval_sample_indices_set = set()

    def _metadata_from_data(self, data_loader):
        """Load metadata from the DataLoader."""
        self._batch_size = data_loader.batch_size
        len_dataset = len(data_loader.dataset)
        if data_loader.drop_last:
            self._n_samples_per_epoch = len_dataset - len_dataset%self._batch_size
        else:
            self._n_samples_per_epoch = len_dataset

    def _metadata_from_file(self, file_path):
        """Load batch and epoch size from disk."""
        with open(file_path, 'r') as f:
            metadata = json.load(f)
        self._batch_size = metadata['batch_size']
        self._n_samples_per_epoch = metadata['n_samples_per_epoch']

    def _save_metadata(self, file_path):
        """Save metadata to disk."""
        metadata = {
            'batch_size': self.batch_size,
            'n_samples_per_epoch': self.n_samples_per_epoch,
            'version': self.VERSION
        }
        with open(file_path, 'w') as f:
            json.dump(metadata, f)

    def _validate_indices(self, step_idx, epoch_idx, batch_idx, need_batch):
        """Check and return step epoch and batch indices.

        If need_batch is True, an error is raised if the batch cannot be determined.
        """
        n_batches_per_epoch = self.n_batches_per_epoch

        if step_idx is not None:
            # Both epoch and batch can be determined.
            epoch_idx, batch_idx = divmod(step_idx, n_batches_per_epoch)
        elif epoch_idx is None:
            raise ValueError("Either step_idx or epoch_idx must be passed.")
        elif batch_idx is None:
            if need_batch:
                raise ValueError("To save tensors either 'step_idx' or both "
                                 "'epoch_idx' and 'batch_idx' must be passed.")
        else:
            step_idx = epoch_idx * n_batches_per_epoch + batch_idx

        return step_idx, epoch_idx, batch_idx