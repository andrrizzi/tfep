#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test objects and function in the module ``tfep.dataset.dataset``.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import copy
import os

import MDAnalysis
import numpy as np
import pint
import pytest
import torch
import torch.utils.data

from tfep.dataset import TrajectoryDataset, TrajectorySubset


# =============================================================================
# GLOBAL VARIABLES
# =============================================================================

SCRIPT_DIR_PATH = os.path.dirname(__file__)
CHLOROMETHANE_PDB_FILE_PATH = os.path.join(SCRIPT_DIR_PATH, 'data', 'chloro-fluoromethane.pdb')
AUXILIARY_DATA_FILE_PATH = os.path.join(SCRIPT_DIR_PATH, 'data', 'auxiliary.xvg')

_U = pint.UnitRegistry()


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope='module')
def read_only_trajectory_dataset():
    """If you want to modify this TrajectoryDataset, copy it first!

    This has return_batch_index = True and no subsampling or atoms selected.
    The timestep of the PDB trajectory is 2 ps.
    """
    universe = MDAnalysis.Universe(CHLOROMETHANE_PDB_FILE_PATH, dt=2)
    return TrajectoryDataset(universe, return_batch_index=True)


# =============================================================================
# TEST UTILITY FUNCTIONS
# =============================================================================

def _check_correct_subset_positions(
        dataset, expected_frame_indices=None, expected_atom_indices=None):
    """Check that the selected frames in the dataset correspond to the expected ones.

    The function checks that the positions in the dataset subsamples are the same
    as those in the dataset.universe.trajectory.

    """
    if expected_frame_indices is None:
        expected_frame_indices = list(range(len(dataset)))

    for i, ts in enumerate(dataset.iterate_as_timestep()):
        dataset_positions = np.array(ts.positions)
        frame_idx = expected_frame_indices[i]
        expected_positions = np.array(dataset.universe.trajectory[frame_idx])

        # Take subset of positions for the selected atoms.
        if expected_atom_indices is not None:
            expected_positions = expected_positions[expected_atom_indices]

        assert np.allclose(dataset_positions, expected_positions)


# =============================================================================
# TESTS
# =============================================================================

@pytest.mark.parametrize('return_batch_index', [False, True])
def test_trajectory_dataset_dataloader_iteration(return_batch_index):
    """Test that TrajectoryDataset interacts well with PyTorch.

    This tests that:
    - A ``TrajectoryDataset`` works well with ``DataLoader`` when iterating for
      batches.
    - Iterating over positions as ``Tensor``s returns the coordinates in flattened
      format.
    - When ``return_batch_index`` is ``True``/``False``, the sample index is also
      returned.
    - Auxiliary information is automatically discovered and returned as well.

    """
    # Create dataset.
    universe = MDAnalysis.Universe(CHLOROMETHANE_PDB_FILE_PATH)
    trajectory_dataset = TrajectoryDataset(universe, return_batch_index=return_batch_index)

    # Create DataLoader.
    batch_size = 2
    data_loader = torch.utils.data.DataLoader(
        trajectory_dataset, batch_size=batch_size, drop_last=True)

    # Test iteration in batches.
    for batch_idx, batch in enumerate(data_loader):
        # Check that positions have been flattened.
        batch_positions = batch['positions']
        assert batch_positions.shape == (2, 18)

        # Check that the index is added.
        if return_batch_index:
            start_index = batch_idx * batch_size
            assert torch.all(batch['index'] == torch.Tensor(range(start_index, start_index+batch_size)))
        else:
            assert 'index' not in batch


@pytest.mark.parametrize('start,stop,step,n_frames,expected_frame_indices', [
    # Test step.
    (0, 4, 1, None, None),
    (1, 3, 2, None, None),
    (0, 3, 2, None, None),
    (2*_U.ps, 4, 2, None, [1, 3]),
    (1, 6*_U.ps, 2, None, [1, 3]),
    (1, 4, 4*_U.ps, None, [1, 3]),
    (2*_U.ps, 6*_U.ps, 4*_U.ps, None, [1, 3]),

    # Test n_frames.
    (0, 4, None, 2, [0, 4]),
    (1, 3, None, 3, [1, 2, 3]),
    (0, 4, None, 3, [0, 2, 4]),
    (2*_U.ps, 4, None, 3, [1, 2, 4]),
    (1, 6*_U.ps, None, 3, [1, 2, 3]),
    (2*_U.ps, 6*_U.ps, None, 2, [1, 3]),
])
def test_trajectory_dataset_subsampling(
        start, stop, step, n_frames, expected_frame_indices, read_only_trajectory_dataset):
    """Test the function TrajectoryDataset.subsample.

    This test that:
    - The __len__ method works as expected.
    - The start/stop/step arguments can be given either in number of frames
      or in units of time.
    - The correct frames are selected and returned when iterating over the
      dataset.

    """
    # Copy dataset to subsample.
    trajectory_dataset = copy.copy(read_only_trajectory_dataset)
    universe = trajectory_dataset.universe  # Shortcut.

    # Check that the correct number of frames have been loaded.
    assert len(trajectory_dataset) == len(universe.trajectory)

    # Subsample.
    trajectory_dataset.subsample(start, stop, step, n_frames)

    # Test that there are the correct number of frames.
    if expected_frame_indices is None:
        expected_frame_indices = list(range(start, stop+1, step))
    assert len(trajectory_dataset) == len(expected_frame_indices)
    assert np.all(trajectory_dataset._subsampled_frame_indices == expected_frame_indices)

    # Check that subsampling selected the correct positions.
    _check_correct_subset_positions(trajectory_dataset, expected_frame_indices)


@pytest.mark.parametrize('selection,expected_atom_indices', [
    ('index 0:4', np.array([0, 1, 2, 3, 4])),
    ('index 1:2 or index 5', np.array([1, 2, 5])),
])
def test_trajectory_dataset_select_atoms(selection, expected_atom_indices, read_only_trajectory_dataset):
    """Test the function TrajectoryDataset.select_atoms.

    This test that:
    - The n_atoms property works as expected.
    - The correct atom positions are selected and returned when iterating over
      the dataset.

    """
    # Copy dataset used for selection.
    trajectory_dataset = copy.copy(read_only_trajectory_dataset)
    universe = trajectory_dataset.universe  # Shortcut.

    # Check that before selection the n_atoms property works as expected.
    assert trajectory_dataset.n_atoms == universe.atoms.n_atoms

    # Select atoms.
    trajectory_dataset.select_atoms(selection)
    assert trajectory_dataset.n_atoms == len(expected_atom_indices)

    # Check that iterating over the dataset returns the correct positions.
    _check_correct_subset_positions(
        trajectory_dataset, expected_atom_indices=expected_atom_indices)


@pytest.mark.parametrize('dt', [1, 5])
def test_trajectory_dataset_auxiliary(dt):
    """Test that TrajectoryDataset returns the auxiliary information correctly."""
    # Create dataset with auxiliary information.
    universe = MDAnalysis.Universe(CHLOROMETHANE_PDB_FILE_PATH, dt=dt)
    universe.trajectory.add_auxiliary('myaux', AUXILIARY_DATA_FILE_PATH)
    trajectory_dataset = TrajectoryDataset(universe)

    # Create DataLoader.
    n_frames = len(trajectory_dataset)
    data_loader = torch.utils.data.DataLoader(trajectory_dataset, batch_size=n_frames)

    # Read all auxiliary information.
    for batch in data_loader:
        aux = batch['myaux']

    # The first column of the test auxiliary data is time+1, while the second
    # column is 25.5-time.
    time = torch.Tensor(list(range(0, n_frames))) * dt
    expected_first_column = time + 1
    expected_second_column = 25.5 - time

    assert torch.all(aux[:, 0] == time)
    assert torch.all(aux[:, 1] == expected_first_column)
    assert torch.all(aux[:, 2] == expected_second_column)


def test_trajectory_subset_nested(read_only_trajectory_dataset):
    """Check that the indices of nested subsets are handled correctly in TrajectorySubset.

    This function tests
    - The indices- and filter-based constructors.
    - That nested subset play nice with each other.
    - That the TrajectoryDataset interface works as expected.

    """
    # Filter the TrajectoryDataset so that we consider only the atoms for
    # which the distance between C and Cl is greater than 2.5 angstrom.
    nested_subset = TrajectorySubset.from_filter(
        read_only_trajectory_dataset,
        filter_func=lambda i, ts: np.linalg.norm(ts.positions[1] - ts.positions[0]) > 2.5
    )
    assert len(nested_subset) == 3

    # Further reduce the dataset using indices.
    subset = TrajectorySubset(nested_subset, indices=[0, 2])
    assert len(subset) == 2

    # Create a subset of atoms to test the TrajectoryDataset interface.
    subset.select_atoms('index 0:2')
    assert subset.n_atoms == 3

    # Check the positions.
    _check_correct_subset_positions(
        subset,
        expected_frame_indices=[0, 4],
        expected_atom_indices=[0, 1, 2]
    )

    # Check that the returned batch index reflects
    # the total number of samples in the subset.
    data_loader = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False)
    for batch_idx, batch in enumerate(data_loader):
        assert batch['index'][0] == batch_idx
