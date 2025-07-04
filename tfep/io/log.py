#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Utility classes to store potential energies and CVs necessary for TFEP.

The module provides a class :class:`.TFEPLogger` that provides an interface to
store on/read from disk the potentials energies and CVs quantities that enter
the TFEP equations.

"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import json
import os
import warnings
from typing import Optional, Union

import numpy as np
import torch
import h5py


# =============================================================================
# BASE LOGGER
# =============================================================================

class BaseLogger:
    VERSION = '0.1'
    METADATA_FILE_NAME = 'metadata.json'
    INDEX_NAMES = ['trajectory_sample_index', 'dataset_sample_index']
    MASK_NAME = '__mask'

    def __init__(self, save_dir_path='tfep_logs', data_loader=None, train_subdir_name='train', eval_subdir_name='eval'):
        self._save_dir_path = os.path.realpath(save_dir_path)
        self._train_dir_path = os.path.join(save_dir_path, train_subdir_name)
        self._eval_dir_path = os.path.join(save_dir_path, eval_subdir_name)

        metadata_file_path = os.path.join(save_dir_path, self.METADATA_FILE_NAME)
        resume = os.path.isfile(metadata_file_path)
        if resume:
            self._metadata_from_file(metadata_file_path)
        elif data_loader is None:
            raise ValueError("When creating a new logger, 'data_loader' must be passed.")
        else:
            self._metadata_from_data(data_loader)

        os.makedirs(save_dir_path, exist_ok=True)
        for dir_path in [self._train_dir_path, self._eval_dir_path]:
            os.makedirs(dir_path, exist_ok=True)
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
        """The number of batches per training epoch."""
        return int(np.ceil(self.n_samples_per_epoch / self.batch_size))

    @property
    def save_dir_path(self):
        """The path to the main directory where the data is stored."""
        return self._save_dir_path

    def _metadata_from_data(self, data_loader):
        """Load metadata from the DataLoader."""
        # batch_size and drop_last could be in DataLoader or in a batch sampler.
        self._batch_size = data_loader.batch_size
        if self._batch_size is None:
            self._batch_size = data_loader.batch_sampler.batch_size
            drop_last = data_loader.batch_sampler.drop_last
        else:
            drop_last = data_loader.drop_last
        n_dataset_samples = len(data_loader.dataset)
        if drop_last:
            self._n_samples_per_epoch = n_dataset_samples - n_dataset_samples % self._batch_size
        else:
            self._n_samples_per_epoch = n_dataset_samples

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
        
    def _validate_indices(self, step_idx, epoch_idx, batch_idx, need_batch):
        """Check and return step epoch and batch indices.

        If need_batch is True, an error is raised if the batch cannot be determined.
        """
        n_batches_per_epoch = self.n_batches_per_epoch
        if step_idx is not None:
            epoch_idx, batch_idx = divmod(step_idx, n_batches_per_epoch)
        elif epoch_idx is None:
            raise ValueError("Either step_idx or epoch_idx must be passed.")
        elif batch_idx is None:
            if need_batch:
                raise ValueError("To save tensors either 'step_idx' or both 'epoch_idx' and 'batch_idx' must be passed.")
        else:
            step_idx = epoch_idx * n_batches_per_epoch + batch_idx
        return step_idx, epoch_idx, batch_idx


# =============================================================================
# TFEP LOGGER
# =============================================================================

class TFEPLogger(BaseLogger):
    """Store and retrieve potential energies and CVs during training/evaluations.

    The user can use this to easily store and retrieve arbitrary per-sample
    quantities such as potential energies and CV values by epoch, batch, or step.

    .. warning::

        Currently, this class is not multi-process or thread safe.

    Current database format
    -----------------------

    Currently, the data is stored in compressed numpy format with a different
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

    """

    def __init__(
            self,
            save_dir_path='tfep_logs',
            data_loader=None,
            train_subdir_name='train',
            eval_subdir_name='eval',
    ):
        """Constructor.

        Parameters
        ----------
        save_dir_path : str, optional
            The main directory where to save the training and evaluation data.
        data_loader : torch.utils.data.DataLoader, optional
            The data loader used for training wrapping a :class:``tfep.io.dataset.TrajectoryDataset``.
            This must be passed when a new logger is created as it is used to
            determine epoch, batch, and trajectory dimensions. If ``save_dir_path``
            points to an existing logger, then this is ignored.
        train_subdir_name : str, optional
            The name of the subdirectory where the training data is stored.
        eval_subdir_name : str, optional
            The name of the subdirectory where the evaluation data is stored.

        """
        super().__init__(save_dir_path, data_loader, train_subdir_name, eval_subdir_name)

        # This keep track of the currently in-memory training/evaluation data.
        self._loaded_train_idx = None
        self._loaded_train_data = None  # Dict[name, Tensor].
        self._loaded_eval_idx = None
        self._loaded_eval_data = None  # Dict[name, Tensor].


    def read_eval_tensors(
            self,
            names: Optional[list[str]] = None,
            step_idx: Optional[int] = None,
            epoch_idx: Optional[int] = None,
            batch_idx: Optional[int] = None,
            remove_nans: Union[bool, str] = False,
            sort_by: Optional[str] = None,
            as_numpy: bool = False,
    ):
        """Read the tensors generated with the NN model trained for the given number of epoch/batch/step.

        Either ``step_idx`` or both ``epoch_idx`` and ``batch_idx`` must be passed.

        Parameters
        ----------
        names : list[str], optional
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
        remove_nans : bool or str, optional
            If ``True`` only the indices corresponding to non NaN entries are
            returned. If a string, only the indices corresponding to NaN values
            of ``tensors[remove_nans]`` are returned.
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
        self._load_data(step_idx, data_type='eval')

        # Sort the data if requested.
        if sort_by is not None:
            sort_indices = np.argsort(self._loaded_eval_data[sort_by])
            self._loaded_eval_data = {k: v[sort_indices] for k, v in self._loaded_eval_data.items()}

            # Update the data on disk.
            self._dump_data(data_type='eval')

        # Remove undesired names.
        if names is None:
            data = self._loaded_eval_data
        else:
            data = {name: self._loaded_eval_data[name] for name in names}

        # Build NaN mask.
        mask = self._build_mask(remove_nans, data_type='eval')
        if mask is not None:
            data = {k: v[mask] for k, v in data.items()}

        # Convert to tensors.
        if not as_numpy:
            data = {k: torch.tensor(v) for k, v in data.items()}

        return data

    def read_train_tensors(
            self,
            names: Optional[list[str]] = None,
            step_idx: Optional[int] = None,
            epoch_idx: Optional[int] = None,
            batch_idx: Optional[int] = None,
            remove_nans: Union[bool, str] = False,
            as_numpy: bool = False,
    ):
        """Read the tensors saved with ``save_train_tensors``.

        At least one between ``step_idx`` and ``epoch_idx`` must be passed.
        Note that only the data for the batches that have been saved are returned.
        As a consequence, the returned tensors might be smaller than the
        number of samples per epoch if the training was interrupted before the
        end of the epoch.

        Parameters
        ----------
        names : list[str], optional
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
        remove_nans : bool or str, optional
            If ``True`` only the indices corresponding to non NaN entries are
            returned. If a string, only the indices corresponding to NaN values
            of ``tensors[remove_nans]`` are returned.
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
        self._load_data(epoch_idx, data_type='train')

        # Determine which names have to be returned.
        if names is None:
            names = [k for k in self._loaded_train_data.keys() if k != self.MASK_NAME]

        # Build mask.
        mask = self._build_mask(remove_nans, data_type='train')

        # Read data for the specific epoch/batch.
        tensors = {}
        for name in names:
            if batch_idx is None:
                # Collect data for entire epoch.
                tensors[name] = self._loaded_train_data[name][mask]
            else:
                # Collect data for a single batch.
                first = self.batch_size * batch_idx
                last = first + self.batch_size
                tensors[name] = self._loaded_train_data[name][first:last][mask[first:last]]

        # Convert from numpy to Tensors.
        if not as_numpy:
            tensors = {k: torch.tensor(v) for k, v in tensors.items()}

        return tensors

    def save_eval_tensors(
            self,
            tensors: dict,
            step_idx: Optional[int] = None,
            epoch_idx: Optional[int] = None,
            batch_idx: Optional[int] = None,
            update: bool = False,
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
            All tensors must have shape ``(batch_size,)`` unless only ``epoch_idx``
            is provided, in which case they must have shape ``(n_samples_per_epoch,)``.
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
            check is not performed and all tensors are simply added to the logger.

        """
        # Warn the user about missing dataset_sample_index nor trajectory_sample_index.
        self._warn_if_no_indices(tensors)

        # Validate input arguments.
        step_idx, _, _ = self._validate_indices(
            step_idx, epoch_idx, batch_idx, need_batch=True)

        # Load in memory the data of this NN.
        self._load_data(step_idx, data_type='eval')

        # Make sure all known tensors are updated.
        if len(self._loaded_eval_data) == 0:
            names = tensors.keys()
        else:
            names = self._loaded_eval_data.keys()

        # Convert everything to numpy arrays.
        try:
            tensors = {n: tensors[n].detach().cpu().numpy() for n in names}
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

        # Update file on disk.
        self._dump_data(data_type='eval')

    def save_train_tensors(
            self,
            tensors: dict,
            step_idx: Optional[int] = None,
            epoch_idx: Optional[int] = None,
            batch_idx: Optional[int] = None,
    ):
        """Save the tensors generated during the given epoch/batch/step of training.

        At least one between ``step_idx`` and ``epoch_idx``/``batch_idx`` must
        be passed.

        Parameters
        ----------
        tensors : Dict[str, torch.Tensor]
            A dictionary mapping the name of the saved tensors to their values.
            All tensors must have shape ``(batch_size,)`` unless only ``epoch_idx``
            is provided, in which case they must have shape ``(n_samples_per_epoch,)``.
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
        self._load_data(epoch_idx, data_type='train')

        # Update all tensors.
        mask = self._loaded_train_data[self.MASK_NAME]
        for name, value in tensors.items():
            # Convert from tensor to array.
            value = value.detach().cpu().numpy()

            if batch_idx is None:
                # Assume data is for the entire epoch.
                self._loaded_train_data[name] = value
                mask[:] = True
            else:
                # Initialize the tensor if this is the first time we see it.
                try:
                    saved_array = self._loaded_train_data[name]
                except KeyError:
                    saved_array = np.empty(self.n_samples_per_epoch, dtype=value.dtype)
                    self._loaded_train_data[name] = saved_array

                # Save the data. This is the first index pointing to the start of the batch.
                first = self.batch_size*batch_idx
                saved_array[first:first+len(value)] = value
                mask[first:first+len(value)] = True

        # Update file on disk.
        self._dump_data(data_type='train')

    # --------------------- #
    # Private class members #
    # --------------------- #


    def _build_mask(self, remove_nans, data_type):
        """Return a boolean mask to select the data elements to mask.

        If no mask must be used, returns None.

        data_type can be 'train' or 'eval'.
        """
        # Shortcut.
        is_eval = data_type == 'eval'

        # Load the data to build the NaN mask.
        data_attr = '_loaded_' + data_type + '_data'  # e.g., _loaded_train_data
        loaded_data = getattr(self, data_attr)

        # No need to use a mask for eval data if remove_nans is False.
        if remove_nans is False:
            if is_eval:
                return None
            else:
                return loaded_data[self.MASK_NAME]

        # Build the NaN mask.
        if remove_nans is True:
            mask = None
            for name, value in loaded_data.items():
                if name != self.MASK_NAME:
                    if mask is None:
                        mask = ~np.isnan(value)
                    else:
                        mask &= ~np.isnan(value)
        else:
            # Then assume it is a key.
            mask = ~np.isnan(loaded_data[remove_nans])

        # Add the invalid training data mask.
        if not is_eval:
            mask &= loaded_data[self.MASK_NAME]

        return mask

    def _dump_data(self, data_type):
        """Dump on disk the currently loaded training data."""
        data_attr = '_loaded_' + data_type + '_data'  # e.g., _loaded_train_data
        file_path = self._get_data_file_path(data_type)
        np.savez_compressed(file_path, **getattr(self, data_attr))

    def _get_data_file_path(self, data_type):
        """The file path where the currently loaded training/evaluation data is stored.

        data_type can be 'train' or 'eval'.
        """
        idx_attr = '_loaded_' + data_type + '_idx'  # e.g., _loaded_train_idx
        idx_str = str(getattr(self, idx_attr))

        if data_type == 'eval':
            file_name = 'step-' + idx_str
            dir_path = self._eval_dir_path
        else:  # train
            file_name = 'epoch-' + idx_str
            dir_path = self._train_dir_path

        return os.path.join(dir_path, file_name + '.npz')

    def _load_data(self, idx, data_type):
        """Load/initialize the training/evaluation data in memory.

        data_type can be 'train' or 'eval'.
        """
        # Check if we have already loaded the data.
        idx_attr = '_loaded_' + data_type + '_idx'  # e.g., _loaded_train_idx
        if getattr(self, idx_attr) == idx:
            return

        # Point to new data file.
        setattr(self, idx_attr, idx)

        # Check if there is data on disk.
        data_attr = '_loaded_' + data_type + '_data'  # e.g., _loaded_train_data
        file_path = self._get_data_file_path(data_type)
        if os.path.isfile(file_path):
            # NpzFile offers lazy loading, but we load everything into memory
            # for now to avoid having to deal with correctly closing the file.
            npz_file = np.load(file_path)
            setattr(self, data_attr, {k: v for k, v in npz_file.items()})
        else:
            # Initialize the data for this epoch.
            if data_type == 'eval':
                data = {}
            else:
                # Training data requires a mask.
                data = {self.MASK_NAME: np.full(self.n_samples_per_epoch, fill_value=False)}
            setattr(self, data_attr, data)


class TMBARLogger(BaseLogger):
    """Store and retrieve potential energies and CVs during training/evaluations.

    The user can use this to easily store and retrieve arbitrary per-sample
    quantities such as potential energies and CV values by epoch, batch, or step.

    .. warning::

        Currently, this class is not multi-process or thread safe.

    Current database format
    -----------------------

    The data is stored in HDF5 format with one file per epoch/step, containing groups
    for different state mappings. The data is organized in the following structure:

    ```
    tfep_logs/
    ├── train/
    │   ├── epoch-0.h5
    │   │   ├── state_0_state_1/  # Mapping from state 0 to state 1
    │   │   │   ├── potential_energy
    │   │   │   ├── cv_values
    │   │   │   └── ...
    │   │   ├── state_0_state_2/  # Mapping from state 0 to state 2
    │   │   └── ...
    │   ├── epoch-1.h5
    │   └── ...
    ├── eval/
    │   ├── step-0.h5
    │   │   ├── state_0_state_1/
    │   │   ├── state_0_state_2/
    │   │   └── ...
    │   └── ...
    └── metadata.json
    ```

    Each HDF5 file contains groups for each state mapping, with datasets for the
    various quantities. The data is organized hierarchically:
    - First level: state mapping groups (e.g., "state_0_state_1" for mapping from state 0 to state 1)
    - Second level: datasets (potential_energy, cv_values, etc.)

    Finally, a JSON file is used to store metadata about the experiment such as
    batch and epoch sizes.

    """

    def __init__(
            self,
            save_dir_path='tfep_logs',
            data_loader=None,
            train_subdir_name='train',
            eval_subdir_name='eval',
            n_states=2,
            state_names: Optional[list[str]] = None,
    ):
        """Constructor.

        Parameters
        ----------
        save_dir_path : str, optional
            The main directory where to save the training and evaluation data.
        data_loader : torch.utils.data.DataLoader, optional
            The data loader used for training wrapping a :class:``tfep.io.dataset.TrajectoryDataset``.
            This must be passed when a new logger is created as it is used to
            determine epoch, batch, and trajectory dimensions. If ``save_dir_path``
            points to an existing logger, then this is ignored.
        train_subdir_name : str, optional
            The name of the subdirectory where the training data is stored.
        eval_subdir_name : str, optional
            The name of the subdirectory where the evaluation data is stored.
        n_states : int, optional
            The number of states in the system. Default is 2.
        state_names : list[str], optional
            Names for each state. If None, defaults to ['state_0', 'state_1', ...].

        """
        # Initialize state parameters before calling parent constructor
        self._n_states = n_states
        self._state_names = state_names if state_names is not None else [f'state_{i}' for i in range(n_states)]
        
        # Generate all possible state mappings
        self._state_mappings = []
        for i in range(n_states):
            for j in range(n_states):
                if i != j:  # Exclude self-mappings
                    self._state_mappings.append((i, j))
        
        # Call parent constructor
        super().__init__(save_dir_path, data_loader, train_subdir_name, eval_subdir_name)

    @property
    def n_states(self):
        """The number of states in the system."""
        return self._n_states

    @property
    def state_names(self):
        """The list of state names."""
        return self._state_names

    @property
    def state_mappings(self):
        """List of all possible state mappings as tuples (from_state, to_state)."""
        return self._state_mappings

    def read_eval_tensors(
            self,
            names: Optional[list[str]] = None,
            step_idx: Optional[int] = None,
            epoch_idx: Optional[int] = None,
            batch_idx: Optional[int] = None,
            remove_nans: Union[bool, str] = False,
            sort_by: Optional[str] = None,
            as_numpy: bool = False,
            state_mapping: Optional[tuple[int, int]] = None,
    ):
        """Read the tensors generated with the NN model trained for the given number of epoch/batch/step.

        Either ``step_idx`` or both ``epoch_idx`` and ``batch_idx`` must be passed.

        Parameters
        ----------
        names : list[str], optional
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
        remove_nans : bool or str, optional
            If ``True`` only the indices corresponding to non NaN entries are
            returned. If a string, only the indices corresponding to NaN values
            of ``tensors[remove_nans]`` are returned.
        sort_by : str, optional
            If given, all the returned tensors will be sorted based on the tensor
            with this name (useful if ``sort_by`` is ``'trajectory_sample_index'``).
            The new data order is also stored on disk thus subsequent calls without
            saving new data are guaranteed to follow in the same order.
        as_numpy : bool, optional
            If ``True``, the tensors are returned as a numpy array rather than
            PyTorch ``Tensors``.
        state_mapping : tuple[int, int], optional
            A tuple (from_state, to_state) indicating the state mapping.
            If None, defaults to the first available mapping.

        Returns
        -------
        tensors : Dict[str, torch.Tensor]
            A dictionary mapping the name of the saved tensors to their values.
        """
        # Validate input arguments.
        step_idx, _, _ = self._validate_indices(
            step_idx, epoch_idx, batch_idx, need_batch=True)

        # Use default state mapping if none provided
        if state_mapping is None:
            state_mapping = self._state_mappings[0]
        elif state_mapping not in self._state_mappings:
            raise ValueError(f"Invalid state mapping {state_mapping}. Must be one of {self._state_mappings}")

        # Read data directly from HDF5
        data = self._read_data_from_hdf5(step_idx, data_type='eval', state_mapping=state_mapping)
        
        if data is None:
            return {}

        # Sort the data if requested.
        if sort_by is not None and sort_by in data:
            sort_indices = np.argsort(data[sort_by])
            data = {k: v[sort_indices] for k, v in data.items()}
            # Update the data on disk with sorted version
            self._write_data_to_hdf5(step_idx, data_type='eval', state_mapping=state_mapping, data=data)

        # Remove undesired names.
        if names is not None:
            data = {name: data[name] for name in names if name in data}

        # Build NaN mask.
        mask = self._build_mask(remove_nans, data)
        if mask is not None:
            data = {k: v[mask] for k, v in data.items()}

        # Convert to tensors.
        if not as_numpy:
            data = {k: torch.tensor(v) for k, v in data.items()}

        return data

    def read_train_tensors(
            self,
            names: Optional[list[str]] = None,
            step_idx: Optional[int] = None,
            epoch_idx: Optional[int] = None,
            batch_idx: Optional[int] = None,
            remove_nans: Union[bool, str] = False,
            as_numpy: bool = False,
            state_mapping: Optional[tuple[int, int]] = None,
    ):
        """Read the tensors saved with ``save_train_tensors``.

        Parameters
        ----------
        names : list[str], optional
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
        remove_nans : bool or str, optional
            If ``True`` only the indices corresponding to non NaN entries are
            returned. If a string, only the indices corresponding to NaN values
            of ``tensors[remove_nans]`` are returned.
        as_numpy : bool, optional
            If ``True``, the tensors are returned as a numpy array rather than
            PyTorch ``Tensors``.
        state_mapping : tuple[int, int], optional
            A tuple (from_state, to_state) indicating the state mapping.
            If None, defaults to the first available mapping.

        Returns
        -------
        tensors : Dict[str, torch.Tensor]
            A dictionary mapping the name of the saved tensors to their values.
        """
        # Check input arguments.
        _, epoch_idx, batch_idx = self._validate_indices(
            step_idx, epoch_idx, batch_idx, need_batch=False)

        # Use default state mapping if none provided
        if state_mapping is None:
            state_mapping = self._state_mappings[0]
        elif state_mapping not in self._state_mappings:
            raise ValueError(f"Invalid state mapping {state_mapping}. Must be one of {self._state_mappings}")

        # Read data directly from HDF5
        data = self._read_data_from_hdf5(epoch_idx, data_type='train', state_mapping=state_mapping)
        
        if data is None:
            return {}

        # Determine which names have to be returned.
        if names is None:
            names = [k for k in data.keys() if k != self.MASK_NAME]
        else:
            names = [k for k in names if k in data and k != self.MASK_NAME]

        # Build mask.
        mask = self._build_mask(remove_nans, data)

        # Read data for the specific epoch/batch.
        tensors = {}
        for name in names:
            if batch_idx is None:
                # Collect data for entire epoch.
                tensors[name] = data[name][mask]
            else:
                # Collect data for a single batch.
                first = self.batch_size * batch_idx
                last = first + self.batch_size
                tensors[name] = data[name][first:last][mask[first:last]]

        # Convert from numpy to Tensors.
        if not as_numpy:
            tensors = {k: torch.tensor(v) for k, v in tensors.items()}

        return tensors

    def save_eval_tensors(
            self,
            tensors: dict,
            step_idx: Optional[int] = None,
            epoch_idx: Optional[int] = None,
            batch_idx: Optional[int] = None,
            update: bool = False,
            state_mapping: Optional[tuple[int, int]] = None,
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
            All tensors must have shape ``(batch_size,)`` unless only ``epoch_idx``
            is provided, in which case they must have shape ``(n_samples_per_epoch,)``.
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
            check is not performed and all tensors are simply added to the logger.
        state_mapping : tuple[int, int], optional
            A tuple (from_state, to_state) indicating the state mapping.
            If None, defaults to the first available mapping.
        """
        # Warn the user about missing dataset_sample_index nor trajectory_sample_index.
        self._warn_if_no_indices(tensors)

        # Validate input arguments.
        step_idx, _, _ = self._validate_indices(
            step_idx, epoch_idx, batch_idx, need_batch=True)

        # Use default state mapping if none provided
        if state_mapping is None:
            state_mapping = self._state_mappings[0]
        elif state_mapping not in self._state_mappings:
            raise ValueError(f"Invalid state mapping {state_mapping}. Must be one of {self._state_mappings}")

        # Convert tensors to numpy arrays
        tensors_np = {name: tensor.detach().cpu().numpy() for name, tensor in tensors.items()}

        # Read existing data if update is requested
        if update:
            existing_data = self._read_data_from_hdf5(step_idx, data_type='eval', state_mapping=state_mapping)
            if existing_data:
                # Update logic for existing data
                for index_name in self.INDEX_NAMES:
                    if index_name in tensors_np and index_name in existing_data:
                        tensor_sample_indices = tensors_np[index_name]
                        _, tensor_indices, loaded_indices = np.intersect1d(
                            tensor_sample_indices, existing_data[index_name],
                            assume_unique=True, return_indices=True
                        )
                        if len(tensor_indices) > 0:
                            for name in tensors_np:
                                existing_data[name][loaded_indices] = tensors_np[name][tensor_indices]
                                tensors_np[name] = np.delete(tensors_np[name], tensor_indices)
                        break

                # Append remaining data
                for name, value in tensors_np.items():
                    if name in existing_data:
                        existing_data[name] = np.concatenate((existing_data[name], value))
                    else:
                        existing_data[name] = value
                
                # Write updated data
                self._write_data_to_hdf5(step_idx, data_type='eval', state_mapping=state_mapping, data=existing_data)
            else:
                # No existing data, just write new data
                self._write_data_to_hdf5(step_idx, data_type='eval', state_mapping=state_mapping, data=tensors_np)
        else:
            # Just write new data (append mode)
            existing_data = self._read_data_from_hdf5(step_idx, data_type='eval', state_mapping=state_mapping)
            if existing_data:
                for name, value in tensors_np.items():
                    if name in existing_data:
                        existing_data[name] = np.concatenate((existing_data[name], value))
                    else:
                        existing_data[name] = value
                self._write_data_to_hdf5(step_idx, data_type='eval', state_mapping=state_mapping, data=existing_data)
            else:
                self._write_data_to_hdf5(step_idx, data_type='eval', state_mapping=state_mapping, data=tensors_np)

    def save_train_tensors(
            self,
            tensors: dict,
            step_idx: Optional[int] = None,
            epoch_idx: Optional[int] = None,
            batch_idx: Optional[int] = None,
            state_mapping: Optional[tuple[int, int]] = None,
    ):
        """Save the tensors generated during the given epoch/batch/step of training.

        Parameters
        ----------
        tensors : Dict[str, torch.Tensor]
            A dictionary mapping the name of the saved tensors to their values.
            All tensors must have shape ``(batch_size,)`` unless only ``epoch_idx``
            is provided, in which case they must have shape ``(n_samples_per_epoch,)``.
        step_idx : int or None
            If given, the tensors for this optimization step are saved.
        epoch_idx : int or None
            If given, the tensors for this epoch are saved.
        batch_idx : int or None
            If given together with ``epoch_idx``, the tensors for this epoch/batch
            are saved. Otherwise, the data is assumed to be for the entire epoch.
        state_mapping : tuple[int, int], optional
            A tuple (from_state, to_state) indicating the state mapping.
            If None, defaults to the first available mapping.
        """
        # Warn the user about missing dataset_sample_index nor trajectory_sample_index.
        self._warn_if_no_indices(tensors)

        # Validate input arguments.
        _, epoch_idx, batch_idx = self._validate_indices(
            step_idx, epoch_idx, batch_idx, need_batch=False)
        
        # Use default state mapping if none provided
        if state_mapping is None:
            state_mapping = self._state_mappings[0]
        elif state_mapping not in self._state_mappings:
            raise ValueError(f"Invalid state mapping {state_mapping}. Must be one of {self._state_mappings}")

        # Convert tensors to numpy arrays
        tensors_np = {name: tensor.detach().cpu().numpy() for name, tensor in tensors.items()}

        # Read existing data
        existing_data = self._read_data_from_hdf5(epoch_idx, data_type='train', state_mapping=state_mapping)
        
        if existing_data is None:
            # Initialize new data
            existing_data = {self.MASK_NAME: np.full(self.n_samples_per_epoch, fill_value=False)}

        # Update data
        mask = existing_data[self.MASK_NAME]
        for name, value in tensors_np.items():
            if batch_idx is None:
                # Assume data is for the entire epoch.
                existing_data[name] = value
                mask[:] = True
            else:
                # Initialize the tensor if this is the first time we see it.
                if name not in existing_data:
                    existing_data[name] = np.empty(self.n_samples_per_epoch, dtype=value.dtype)

                # Save the data. This is the first index pointing to the start of the batch.
                first = self.batch_size*batch_idx
                existing_data[name][first:first+len(value)] = value
                mask[first:first+len(value)] = True

        # Write data to HDF5
        self._write_data_to_hdf5(epoch_idx, data_type='train', state_mapping=state_mapping, data=existing_data)

    def _read_data_from_hdf5(self, idx: int, data_type: str, state_mapping: tuple[int, int]) -> Optional[dict]:
        """Read data directly from HDF5 file.

        Parameters
        ----------
        idx : int
            The index of the data to load
        data_type : str
            Either 'train' or 'eval'
        state_mapping : tuple[int, int]
            The state mapping (from_state, to_state)

        Returns
        -------
        Optional[dict]
            The data dictionary, or None if file/group doesn't exist
        """
        file_path = self._get_data_file_path(data_type, idx)
        group_name = self._get_group_name(state_mapping)
        
        if not os.path.isfile(file_path):
            return None
        
        try:
            with h5py.File(file_path, 'r') as f:
                if group_name not in f:
                    return None
                
                # Load all datasets from the group
                data = {}
                for key, dataset in f[group_name].items():
                    try:
                        data[key] = dataset[()]
                    except Exception as e:
                        warnings.warn(f"Failed to load dataset '{key}' from {file_path}:{group_name}: {e}")
                        continue
                
                return data if data else None
                
        except OSError as e:
            warnings.warn(f"Failed to open HDF5 file {file_path}: {e}")
            return None
        except Exception as e:
            warnings.warn(f"Unexpected error reading from {file_path}:{group_name}: {e}")
            return None

    def _write_data_to_hdf5(self, idx: int, data_type: str, state_mapping: tuple[int, int], data: dict):
        """Write data directly to HDF5 file.

        Parameters
        ----------
        idx : int
            The index of the data to write
        data_type : str
            Either 'train' or 'eval'
        state_mapping : tuple[int, int]
            The state mapping (from_state, to_state)
        data : dict
            The data to write
        """
        file_path = self._get_data_file_path(data_type, idx)
        group_name = self._get_group_name(state_mapping)
        
        with h5py.File(file_path, 'a') as f:
            # Create group if it doesn't exist
            if group_name not in f:
                group = f.create_group(group_name)
                # Store state mapping as attributes
                from_state, to_state = state_mapping
                group.attrs['from_state'] = from_state
                group.attrs['to_state'] = to_state
                group.attrs['from_state_name'] = self._state_names[from_state]
                group.attrs['to_state_name'] = self._state_names[to_state]
            else:
                group = f[group_name]
            
            # Store the data
            for name, data_array in data.items():
                if name in group:
                    del group[name]  # Delete existing dataset if it exists
                group.create_dataset(name, data=data_array, compression='gzip')

    def _get_data_file_path(self, data_type: str, idx: int) -> str:
        """Get the path to the HDF5 file for the given data type and index.

        Parameters
        ----------
        data_type : str
            Either 'train' or 'eval'
        idx : int
            The epoch or step index

        Returns
        -------
        str
            The path to the HDF5 file
        """
        if data_type == 'eval':
            file_name = f'step-{idx}.h5'
            dir_path = self._eval_dir_path
        else:  # train
            file_name = f'epoch-{idx}.h5'
            dir_path = self._train_dir_path

        return os.path.join(dir_path, file_name)

    def _get_group_name(self, state_mapping: tuple[int, int]) -> str:
        """Get the HDF5 group name for the given state mapping.

        Parameters
        ----------
            state_mapping : tuple[int, int]
            The state mapping (from_state, to_state)

        Returns
        -------
        str
            The group name in format "state_X_state_Y"
        """
        from_state, to_state = state_mapping
        return f"{self._state_names[from_state]}_{self._state_names[to_state]}"

    def _build_mask(self, remove_nans: Union[bool, str], data: dict) -> Optional[np.ndarray]:
        """Return a boolean mask to select the data elements to mask.

        If no mask must be used, returns None.

        Parameters
        ----------
        remove_nans : bool or str
            If True, mask out NaN values. If string, mask out NaN values in that specific tensor.
        data : dict
            The data dictionary

        Returns
        -------
        Optional[np.ndarray]
            Boolean mask, or None if no mask needed
        """
        if remove_nans is False:
            return None

        # Build the NaN mask.
        if remove_nans is True:
            mask = None
            for name, value in data.items():
                if name != self.MASK_NAME:
                    if mask is None:
                        mask = ~np.isnan(value)
                    else:
                        mask &= ~np.isnan(value)
        else:
            # Then assume it is a key.
            if remove_nans in data:
                mask = ~np.isnan(data[remove_nans])
            else:
                return None

        # Add the training data mask if it exists
        if self.MASK_NAME in data:
            mask &= data[self.MASK_NAME]

        return mask

    def _metadata_from_file(self, file_path):
        """Load metadata from disk, including state-related metadata.

        Parameters
        ----------
        file_path : str
            Path to the metadata file.
        """
        with open(file_path, 'r') as f:
            metadata = json.load(f)
        
        # Load basic metadata
        self._batch_size = metadata['batch_size']
        self._n_samples_per_epoch = metadata['n_samples_per_epoch']
        
        # Load state-related metadata
        self._n_states = metadata.get('n_states', 2)
        self._state_names = metadata.get('state_names', [f'state_{i}' for i in range(self._n_states)])
        
        # Load state mappings
        state_mappings = metadata.get('state_mappings', [])
        if state_mappings:
            # Convert string mappings back to tuples
            self._state_mappings = [tuple(map(int, mapping.split('_'))) for mapping in state_mappings]
        else:
            # Generate default mappings if not found
            self._state_mappings = []
            for i in range(self._n_states):
                for j in range(self._n_states):
                    if i != j:  # Exclude self-mappings
                        self._state_mappings.append((i, j))

    def _save_metadata(self, file_path):
        """Save metadata to disk, including state-related metadata.

        Parameters
        ----------
        file_path : str
            Path to the metadata file.
        """
        metadata = {
            'batch_size': self.batch_size,
            'n_samples_per_epoch': self.n_samples_per_epoch,
            'version': self.VERSION,
            'n_states': self._n_states,
            'state_names': self._state_names,
            'state_mappings': [f"{i}_{j}" for i, j in self._state_mappings]
        }
        with open(file_path, 'w') as f:
            json.dump(metadata, f)