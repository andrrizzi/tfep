#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Utility classes to create PyTorch ``Dataset``s from MDAnalysis trajectories.

The module provides a class :class:`.TrajectoryDataset` that wraps an
MDAnalysis ``Universe`` object (i.e., an object tying a topology and a
trajectory) and implements PyTorch's ``Dataset`` interface. This can be used,
for example, to specify the training dataset for the neural network
implementing the mapping function of targeted free energy perturbation.

The :class:`.TrajectoryDataset` can be subsampled at constant time interval,
while arbitrary subsets of the trajectory can be instead be created using the
:class:`.TrajectorySubset` class.

For usage examples see the documentation of
- :class:`.TrajectoryDataset`
- :class:`.TrajectorySubset`

"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import copy

import numpy as np
import pint
import torch.utils.data


# =============================================================================
# TRAJECTORY DATASET
# =============================================================================

class TrajectoryDataset(torch.utils.data.Dataset):
    """PyTorch ``Dataset`` wrapping an MDAnalysis trajectory.

    The class wraps an ``MDAnalysis.Universe`` object and provide the interface
    of a Pytorch ``Dataset`` to enable the iteration of the trajectory in batches.

    When iterating each batch sample is a dictionary including the following
    keys:

    - ``"positions"``: The coordinates of the system in MDAnalysis units as a
          ``torch.Tensor`` of shape ``(batch_size, n_atoms * 3)``.
    - ``"dataset_sample_index"`` (optional): The index in the dataset if
          ``return_dataset_sample_index`` is ``True``. This is useful to match
          the frame index when the dataset is shuffled.
    - ``"trajectory_sample_index"`` (optional): The index in the trajectory if
          ``return_trajectory_sample_index`` is ``True``. This is useful to match
          the data point to the trajectory frame index.
    - ``"aux1"`` (optional): The name of eventual auxiliary information found
          in the ``universe.trajectory.aux`` dictionary.
    - ``"aux2"`` ...

    Parameters
    ----------
    universe : MDAnalysis.Universe
        An MDAnalysis ``Universe`` object encapsulating both the topology and
        the trajectory.
    return_dataset_sample_index : bool, optional
        If ``True``, the keyword ``"dataset_sample_index"`` is included in the
        batch sample when iterating over the dataset.
    return_trajectory_sample_index : bool, optional
        If ``True``, the keyword ``"trajectory_sample_index"`` is included in
        the batch sample when iterating over the dataset.

    Attributes
    ----------
    universe : MDAnalysis.Universe
        The MDAnalysis ``Universe`` object encapsulated by the dataset.
    return_dataset_sample_index : bool, optional
        Whether to return the keyword ``"dataset_sample_index"`` in the batch
        sample.
    return_trajectory_sample_index : bool, optional
        Whether to return the keyword ``"trajectory_sample_index"`` in the batch
        sample.

    Examples
    --------

    First, you need to create an MDAnalysis ``Universe`` (see the
    `MDAnalysis documentation <https://userguide.mdanalysis.org/stable/>`_).
    In this case, we load a short trajectory with a timestep of 5 ps.

    >>> import os
    >>> import MDAnalysis
    >>> test_data_dir_path = os.path.join(os.path.dirname(__file__), '..', '..', 'tests', 'data')
    >>> pdb_file_path = os.path.join(test_data_dir_path, 'chloro-fluoromethane.pdb')
    >>> universe = MDAnalysis.Universe(pdb_file_path, dt=5)  # ps

    ``TrajectoryDataset`` objects can be used as a normal PyTorch ``Dataset``.

    >>> import torch.utils.data
    >>> trajectory_dataset = TrajectoryDataset(universe)
    >>> data_loader = torch.utils.data.DataLoader(trajectory_dataset, batch_size=2, drop_last=True)
    >>> for batch in data_loader:
    ...     batch_positions = batch['positions']

    By default, ``TrajectoryDataset`` flattens the coordinates of the trajectory
    frames so that each batch has shape ``(batch_size, n_atoms * 3)``.

    >>> trajectory_dataset.n_atoms
    6
    >>> batch_positions.shape
    torch.Size([2, 18])

    Only a subset of frames in the trajectories can be included in the dataset
    by subsampling the trajectory at regular intervals (which can be given in
    number of frames or in units of time). The following overwrites the previous
    selection and discards the first and every other frame until the trajectory
    reaches 20 ps of length (limits are included).

    >>> import pint
    >>> ureg = pint.UnitRegistry()
    >>> trajectory_dataset.subsample(
    ...     start=1, stop=20*ureg.picoseconds, step=2)
    >>> len(trajectory_dataset)
    2

    It is also possible to select a subgroup of atoms to include in the dataset.
    The string must follow `MDAnalysis selection syntax <https://docs.mdanalysis.org/stable/documentation_pages/selections.html>`_.
    This selects the first 5 atoms.

    >>> trajectory_dataset.select_atoms('index 0:4')
    >>> trajectory_dataset.n_atoms
    5

    Auxiliary information in the MDAnalysis ``Trajectory`` is also automatically
    discovered and returned while iterating.

    >>> trajectory_dataset.universe.trajectory.add_auxiliary(
    ...    'my_aux_name', os.path.join(test_data_dir_path, 'auxiliary.xvg'))
    >>> data_loader = torch.utils.data.DataLoader(trajectory_dataset, batch_size=2)
    >>> for batch in data_loader:
    ...     aux_info = batch['my_aux_name']

    """
    def __init__(
            self,
            universe,
            return_dataset_sample_index=True,
            return_trajectory_sample_index=True,
    ):
        super().__init__()

        self.universe = universe
        self.return_dataset_sample_index = return_dataset_sample_index
        self.return_trajectory_sample_index = return_trajectory_sample_index

        # The indexes of the selected trajectory frames. None means all frames.
        self.trajectory_sample_indices = None

        # The MDAnalysis.core.groups.AtomGroup object encapsulating the atom
        # selection. None means all atoms.
        self._selected_atom_group = None

    @property
    def n_atoms(self):
        """Number of selected atoms in the dataset."""
        if self._selected_atom_group is None:
            return self.universe.atoms.n_atoms
        return self._selected_atom_group.n_atoms

    def __copy__(self):
        copied_dataset = self.__class__(
            self.universe.copy(),
            self.return_dataset_sample_index,
            self.return_trajectory_sample_index,
        )
        copied_dataset.trajectory_sample_indices = copy.copy(self.trajectory_sample_indices)
        copied_dataset._selected_atom_group = copy.copy(self._selected_atom_group)
        return copied_dataset

    def __getitem__(self, idx):
        """Implement the ``__getitem__()`` method required for a PyTorch dataset.

        Parameters
        ----------
        index : int
            The frame index of the timestep. This must always be comprised
            between ``0`` and ``len(TrajectoryDataset)``. Note that the
            dataset might contain a smaller of frames than the full trajectory
            if a subset of frames was selected (for example, with
            :func:`~tfep.io.dataset.TrajectoryDataset.subsample`).

        Returns
        -------
        sample : dict
            A dictionary including positions (as 1D ``Tensor`` of length
            ``n_atoms * 3``, and optionally the sample index and the auxiliary
            information for the sample

        """
        ts = self.get_timestep(idx)
        sample = {}

        # MDAnalysis loads coordinates with np.float32 dtype. We convert
        # it to the default torch dtype and return them in flattened shape.
        sample['positions'] =  torch.tensor(np.ravel(ts.positions),
                                            dtype=torch.get_default_dtype())
        if ts.dimensions is not None:
            sample['dimensions'] = torch.tensor(ts.dimensions, dtype=torch.get_default_dtype())

        # Return the configurations and the auxiliary information. If an
        # atom group is selected, this may have lost the auxiliary information
        # so we go back to reading the main Trajectory Timestep for this.
        for aux_name, aux_info in self.universe.trajectory.ts.aux.items():
            sample[aux_name] = torch.tensor(aux_info)

        # Return the requested indices.
        if self.return_dataset_sample_index:
            sample['dataset_sample_index'] = idx
        if self.return_trajectory_sample_index:
            if self.trajectory_sample_indices is None:
                # We have selected all frames. Trajectory and dataset indices are the same.
                sample['trajectory_sample_index'] = idx
            else:
                sample['trajectory_sample_index'] = self.trajectory_sample_indices[idx]
        return sample

    def __len__(self):
        """Number of samples in the dataset (i.e., selected trajectory frames)."""
        if self.trajectory_sample_indices is None:
            return len(self.universe.trajectory)
        return len(self.trajectory_sample_indices)

    def get_timestep(self, idx):
        """Return the MDAnalysis ``Timestep`` object for the given index.

        Parameters
        ----------
        index : int
            The frame index of the timestep. This must always be comprised
            between ``0`` and ``len(TrajectoryDataset)``. Note that the
            dataset might contain a smaller of frames than the full trajectory
            if a subset of frames was selected (for example, with
            :func:`~tfep.io.dataset.TrajectoryDataset.subsample`).

        Returns
        -------
        ts : MDAnalysis.coordinates.base.Timestep
            The MDAnalysis ``Timestep`` object with coordinate information of
            the ``index``-th frame. If a subset of atoms was selected (for
            example with :func:`~tfep.io.dataset.TrajectoryDataset.select_atoms`)
            only the coordinates of those atoms are returned.

        """
        # First check if index refers to a subset of selected trajectory frames
        # or to the full trajectory.
        if self.trajectory_sample_indices is None:
            ts = self.universe.trajectory[idx]
        else:
            ts = self.universe.trajectory[self.trajectory_sample_indices[idx]]

        # If a subset of atoms was selected. Return the Timestep only for those.
        if self._selected_atom_group is None:
            return ts
        else:
            return self._selected_atom_group.ts

    def iterate_as_timestep(self):
        """Iterate over the selected frames/atoms as MDAnalysis ``Timestep`` objects.

        Iterating over a ``TrajectoryDataset`` returns the trajectory information
        in ``torch.Tensor`` format. This method enables iterating the samples in
        the dataset as MDAnalysis ``Timestep`` objects.

        Note that it is still possible to iterate over ``Timestep`` objects using
        the MDAnalysis API and ``TrajectoryDataset.universe.trajectory``. However,
        in this case, the selections of frames/atoms performed at the
        ``TrajectoryDataset`` level are ignored and all frames/atoms are returned.

        Yields
        ------
        ts : MDAnalysis.coordinates.base.Timestep
            The current ``Timestep`` object.

        """
        for i in range(len(self)):
            yield self.get_timestep(i)

    def select_atoms(self, selection):
        """Select a subset of atoms.

        Iterating over the dataset after selecting a subset of atoms yields only
        the coordinates of these atoms.

        For more information about the selection syntax consult the
        `MDAnalysis documentation <https://docs.mdanalysis.org/stable/documentation_pages/selections.html>`_.

        Parameters
        ----------
        selection : str
            The selection string following the MDAnalysis selection syntax.

        """
        self._selected_atom_group = self.universe.select_atoms(selection)

    def subsample(self, start=None, stop=None, step=None, n_frames=None):
        """Select a subset of trajectory frames by subsampling it at regular intervals.

        This function does not modify the trajectory. Thus
        ``TrajectoryDataset.universe.trajectory`` still have the same number of
        frames. However, when iterating over the ``TrajectoryDataset`` only the
        subsampled frames are returned.

        ``start``, ``stop``, and ``step`` can be given either as number of frames
        or in units of time as ``pint.Quantity``. If the latter, the initial time
        of the simulation t0 is taken into account. Note that this might not be
        zero if, for example, the simulation was resumed. For example, if
        ``start`` is 2 ns and the simulation starts at 1 ns, only the first ns
        of data is discarded.

        Parameters
        ----------
        start : int or pint.Quantity, optional
            The first frame to include in the dataset specified either as a
            frame index or in simulation time. If not provided, the subsampling
            starts from the first frame in the trajectory.
        stop : int or pint.Quantity, optional
            The last frame to include in the dataset specified either as a
            frame index or in simulation time. If not provided, the subsampling
            ends at the last frame in the trajectory.
        step : int or pint.Quantity, optional
            The step used for subsampling specified either as a frame index or
            or in simulation time. Only one between ``step`` and ``n_frames``
            may be passed.
        n_frames : int, optional
            The total number of frames to include in the dataset. If this is
            passed, the ``step`` will automatically be determined to satisfy this
            requirement. Note that in this case the obtained samples in the
            dataset might not be equally spaced if ``n_frames`` is not an exact
            divisor of the number of frames. Only one between ``step`` and
            ``n_frames`` may be passed.

        """
        # If all are None, there's no need to subsample.
        if all([x is None for x in [start, stop, step, n_frames]]):
            return

        # Handle default arguments. step and n_frames are handled by get_subsampled_indices.
        if start is None:
            start = 0
        if stop is None:
            # Stop is the last index included in the subsampled trajectory.
            stop = len(self.universe.trajectory)-1

        # Look for a compatible unit registry, if given.
        ureg = None
        for quantity in [start, stop, step]:
            if isinstance(quantity, pint.Quantity):
                ureg = quantity._REGISTRY
                break
        if ureg is None:
            ureg = pint.UnitRegistry()

        # All time quantities in MDAnalysis are in picoseconds.
        ps = ureg.picoseconds
        self.trajectory_sample_indices = get_subsampled_indices(
            dt=self.universe.trajectory.dt * ps,
            stop=stop, start=start, step=step, n_frames=n_frames,
            t0=self.universe.trajectory[0].time * ps)


# =============================================================================
# TRAJECTORY SUBSET
# =============================================================================

class TrajectorySubset:
    """A subset of a ``TrajectoryDataset``.

    Provides the same functionality of the PyTorch class ``torch.utils.data.Subset``,
    which is to provided a subset of the main dataset, but for :class:`.TrajectoryDataset`.

    Contrarily to ``torch.utils.data.Subset``, ``TrajectorySubset`` can also be
    constructed from a filter function rather than only a list of indices.

    The class exposes the same interface as :class:`.TrajectoryDataset`, with
    the exception of :func:`.TrajectoryDataset.subsample`. The reason for this
    exception is to avoid users to inadvertantly leave an object in an undersired
    state since the indices of ``TrajectorySubset`` might be meaningless after
    the subsampling.

    Parameters
    ----------
    dataset : TrajectoryDataset or TrajectorySubset
        The trajectory dataset.
    indices : array_like
        A list of indices of the ``dataset`` elements forming the subset.

    Examples
    --------

    First we create the main ``TrajectoryDataset``

    >>> import os
    >>> import MDAnalysis
    >>> test_data_dir_path = os.path.join(os.path.dirname(__file__), '..', '..', 'tests', 'data')
    >>> pdb_file_path = os.path.join(test_data_dir_path, 'chloro-fluoromethane.pdb')
    >>> universe = MDAnalysis.Universe(pdb_file_path, dt=5)  # ps
    >>> trajectory_dataset = TrajectoryDataset(universe)

    We can then create a subset of the indices.

    >>> len(trajectory_dataset)
    5
    >>> trajectory_subset = TrajectorySubset(trajectory_dataset, indices=[0, 2, 4])
    >>> len(trajectory_subset)
    3

    Or alternatively from a filter function taking as input an MDAnalysis
    ``Timestep`` object and returning ``True`` or ``False`` whether the sample
    must be included in the subset or not. The following trivial example takes
    all samples for which the distance between two atoms is greater than 3
    Angstrom.

    >>> filter_func = lambda idx, ts: np.linalg.norm(ts.positions[1] - ts.positions[0]) > 3
    >>> trajectory_subset = TrajectorySubset.from_filter(trajectory_dataset, filter_func)
    >>> len(trajectory_subset)
    2

    The ``TrajectorySubset`` can be used as a normal ``TrajectoryDataset``.

    >>> trajectory_subset.n_atoms
    6
    >>> trajectory_subset.select_atoms('index 0:2 or index 4')

    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

        # Make sure indices is an array or the subset search won't work.
        if not isinstance(self.indices, np.ndarray):
            self.indices = np.array(self.indices)

    @classmethod
    def from_filter(cls, dataset, filter_func):
        """Static constructor creating a subset based on a boolean filter function.

        Parameters
        ----------
        dataset : TrajectoryDataset
            The trajectory dataset.
        filter_func : Callable
            A function taking as input (in this order) the index of the sample
            in the original dataset and the MDAnalysis ``Timestep`` object and
            returning ``True`` or ``False`` if the sample must be included in
            the subset or not.

        Returns
        -------
        subset : TrajectorySubset
            A new ``TrajectorySubset`` object.

        """
        indices = []
        for idx, ts in enumerate(dataset.iterate_as_timestep()):
            if filter_func(idx, ts):
                indices.append(idx)
        return cls(dataset, indices)

    @property
    def universe(self):
        """The MDAnalysis ``Universe`` object encapsulated by the dataset."""
        return self.dataset.universe

    @property
    def return_dataset_sample_index(self):
        """Whether to return the keyword ``"dataset_sample_index"`` in the batch sample."""
        return self.dataset.return_dataset_sample_index

    @property
    def return_trajectory_sample_index(self):
        """Whether to return the keyword ``"trajectory_sample_index"`` in the batch sample."""
        return self.dataset.return_trajectory_sample_index

    @property
    def n_atoms(self):
        """Number of selected atoms in the dataset."""
        return self.dataset.n_atoms

    @property
    def trajectory_sample_indices(self):
        """Indices of the dataset semples in the trajectory (before subsampling).

        ``trajectory_sample_indices[i]`` is the index of the ``i``-th sample in
        ``self.dataset.trajectory``.
        """
        trajectory_sample_indices = self.dataset.trajectory_sample_indices
        return trajectory_sample_indices[self.indices]

    def __getitem__(self, idx):
        """Implement the ``__getitem__()`` method required for a PyTorch dataset."""
        sample = self.dataset[self.indices[idx]]

        # Update the index if return_dataset_sample_index is True.
        # The trajectory index should already be correct.
        if self.return_dataset_sample_index:
            sample['dataset_sample_index'] = idx

        return sample

    def __len__(self):
        """Number of samples in the dataset (i.e., selected trajectory frames)."""
        return len(self.indices)

    def get_timestep(self, item):
        """Return the MDAnalysis ``Timestep`` object of the frame with the given index.

        See also :func:`.TrajectoryDataset.get_timestep`.
        """
        return self.dataset.get_timestep(self.indices[item])

    def iterate_as_timestep(self):
        """Iterate over MDAnalysis ``Timestep`` objects.

        See also :func:`.TrajectoryDataset.iterate_as_timestep`.
        """
        for idx in range(len(self)):
            yield self.get_timestep(idx)

    def select_atoms(self, selection):
        """Select a subset of atoms.

        See also :func:`.TrajectoryDataset.iterate_as_timestep.select_atoms`.
        """
        self.dataset.select_atoms(selection)


# =============================================================================
# SUBSAMPLING UTILITIES
# =============================================================================

def get_subsampled_indices(
        dt,
        stop,
        start=0,
        step=None,
        n_frames=None,
        t0=0.0,
):
    """Subsamples the trajectory at a constant time interval after discarding an initial equilibration.

    This function returns the indices of the trajectory frames that must be
    selected for subsampling.

    ``start``, ``stop``, and ``step`` can be given either as number of frames
    or in units of time as ``pint.Quantity``. If the latter, the initial time
    of the simulation t0 is taken into account. Note that this might not be
    zero if, for example, the simulation was resumed. For example, if
    ``start`` is 2 ns and the simulation starts at 1 ns, only the first ns
    of data is discarded.

    Parameters
    ----------
    start : int or pint.Quantity
        The first frame to include in the dataset specified either as a
        frame index or in simulation time. If not provided, the subsampling
        starts from the first frame in the trajectory.
    stop : int or pint.Quantity
        The last frame to include in the dataset specified either as a
        frame index or in simulation time.
    step : int or pint.Quantity, optional
        The step used for subsampling specified either as a frame index or
        or in simulation time. Only one between ``step`` and ``n_frames`` may
        be passed.
    n_frames : int, optional
        The total number of frames to include in the dataset. If this is
        passed, the ``step`` will automatically be determined to satisfy this
        requirement. Note that in this case the obtained samples in the
        dataset might not be equally spaced if ``n_frames`` is not an exact
        divisor of the number of frames. Only one between ``step`` and ``n_frames``
        may be passed.
    t0 : pint.Quantity, optional
        The time of the first frame in the trajectory to subsamples. This might
        not be 0.0 if, for example, the simulation was resumed.

    Returns
    -------
    trajectory_indices : numpy.ndarray
        The indices of the trajectory frames to use for subsampling.

    """
    # Check that only one between step and n_frames is given.
    if (step is not None) and (n_frames is not None):
        raise ValueError("Only one between 'step' and 'n_frames' may be passed.")

    # Make time quantities unitless.
    ureg = dt._REGISTRY
    unit = ureg.picoseconds
    dt = dt.to(unit).magnitude
    if t0 is None:
        t0 = 0.0
    else:
        t0 = t0.to(unit).magnitude

    # Convert start, stop, and step to frame indices.
    times = [start, stop, step]
    for i, (t, label) in enumerate(zip(times, ['start', 'stop', 'step'])):
        if isinstance(t, pint.Quantity):
            if label == 'step':
                # No need to subtract t0.
                frame_idx = t.to(unit).magnitude / dt
            else:
                frame_idx = (t.to(unit).magnitude - t0) / dt

            if not np.isclose(frame_idx, np.round(frame_idx)):
                closest_times = dt * np.array([np.floor(frame_idx), np.ceil(frame_idx)]) * unit
                raise ValueError(f'The time step {dt} is not compatible with {label} time {t}. '
                                 f'The closest possible start times are {closest_times[0]} or '
                                 f'{closest_times[1]}')
            times[i] = int(round(frame_idx))
    start, stop, step = times

    # Check if the step must be instead determined by the number of frames.
    if n_frames is not None:
        # Check that there are enough frames.
        if n_frames > stop - start + 1:
            raise ValueError(f"There are not enough frames to select {n_frames} "
                             f"from a trajectory with the start time {start*dt} ps"
                             f" and stop time {stop*dt} ps")

        # Create the frames with a constant number of frames.
        return np.linspace(start, stop, n_frames).astype(int)

    # Create the frames with a constant step. We include "stop" in the dataset.
    if (step is None) and (n_frames is None):
        step = 1
    return np.arange(start, stop+1, step, dtype=int)

