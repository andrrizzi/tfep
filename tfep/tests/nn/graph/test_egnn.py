#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test classes and functions in tfep.nn.modules.egnn.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import numpy as np
import pytest
import scipy.spatial.transform
import torch

from tfep.nn.graph.egnn import EGNNDynamics, _EGLayer, _unsorted_segment_sum
from tfep.utils.geometry import batchwise_rotate
from tfep.utils.misc import atom_to_flattened, flattened_to_atom


# =============================================================================
# TEST MODULE CONFIGURATION
# =============================================================================

_old_default_dtype = None

def setup_module(module):
    global _old_default_dtype
    _old_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.double)


def teardown_module(module):
    torch.set_default_dtype(_old_default_dtype)


# =============================================================================
# TESTS EQUIVARIANT GRAPH NEURAL NETWORK DYNAMICS
# =============================================================================

@pytest.mark.parametrize('batch_size', [1, 2])
def test_egnn_dynamics_node_embedding(batch_size):
    """Test the embedding of the atom types and time into the node features."""
    particle_types = torch.tensor([0, 0, 1])  # Three particles.
    time_feat_dim = 2
    node_feat_dim = 4

    egnn_dynamics = EGNNDynamics(
        particle_types=particle_types,
        r_cutoff=6.0,
        time_feat_dim=time_feat_dim,
        node_feat_dim=node_feat_dim,
    )

    h = egnn_dynamics._create_node_embedding(t=torch.tensor(1.0), batch_size=batch_size)

    # The features result from the concatenation of the atom type and time expansions.
    assert h.shape == (batch_size*len(particle_types), node_feat_dim)

    # Identical atom types must have identical embeddings.
    if batch_size == 1:
        zero_type = [1]
        one_type = []
    else:
        zero_type = [1, 3, 4]
        one_type = [5]

    for i in zero_type:
        assert torch.allclose(h[0], h[i])
    for i in one_type:
        assert torch.allclose(h[2], h[i])
    assert not torch.allclose(h[0], h[2])


@pytest.mark.parametrize('batch_size', [1, 2])
def test_egnn_dynamics_edges(batch_size):
    """EGNNDynamics removes edges between particles more distant than the cutoff."""
    particle_types = torch.tensor([1, 0, 0]) # Three types.
    n_particles = len(particle_types)

    egnn_dynamics = EGNNDynamics(
        particle_types=particle_types,
        r_cutoff=6.0,
    )
    edges = egnn_dynamics._get_edges(batch_size)

    for i in range(2):
        assert edges[i].shape == (n_particles*(n_particles - 1)*batch_size,)

    # There should be no edge between samples within the same batch.
    for edge_src, edge_dest in zip(*edges):
        if edge_src < n_particles:
            assert edge_dest < n_particles
        elif edge_src >= n_particles:
            assert edge_dest >= n_particles

    # EGNNDynamics can internally cache the edges for a fixed batch_size but in
    # the last batch the batch_size can be different. Check that it handles this
    # case.
    if batch_size == 2:
        edges = egnn_dynamics._get_edges(1)
        for i in range(2):
            assert edges[i].shape == (n_particles*(n_particles - 1),)


@pytest.mark.parametrize('batch_size', [1, 4])
@pytest.mark.parametrize('r_cutoff', [1.5, 10.0])
@pytest.mark.parametrize('seed', [0, 5, 10])
def test_egnn_dynamics_equivariance_property(batch_size, r_cutoff, seed):
    """Test that the output of the EGNN has the correct equivariance properties."""
    particle_types = torch.tensor([0, 0, 1])
    n_particles = len(particle_types)

    # Create the equivariant graph.
    egnn_dynamics = EGNNDynamics(
        particle_types=particle_types,
        r_cutoff=r_cutoff,
        time_feat_dim=2,
        node_feat_dim=5,
        distance_feat_dim=4,
        n_layers=2,
        initialize_identity=False,
    )

    # Create a random input.
    generator = torch.Generator()
    generator.manual_seed(seed)
    t = torch.rand(1, generator=generator)
    x = torch.randn(batch_size, n_particles*3, generator=generator)

    # Run the dynamics.
    vel = egnn_dynamics(t, x)

    # Now re-run the dynamics with a randomly rotated input.
    random_state = np.random.RandomState(seed)
    rotation_matrices = scipy.spatial.transform.Rotation.random(
        num=batch_size, random_state=random_state)
    rotation_matrices = torch.tensor(rotation_matrices.as_matrix())

    # Rotate the input.
    x_rotated = batchwise_rotate(flattened_to_atom(x), rotation_matrices)
    x_rotated = atom_to_flattened(x_rotated)

    # The velocity should rotate accordingly.
    vel_rotated = egnn_dynamics(t, x_rotated)
    ref_vel = atom_to_flattened(batchwise_rotate(flattened_to_atom(vel), rotation_matrices))
    assert torch.allclose(ref_vel, vel_rotated)

    # Translate the input with shape (batch_size, n_particles*3).
    translation = torch.randn(batch_size, 3)
    translation = translation.repeat(1, n_particles)
    x_translated = x + translation

    # The velocity should be left invariant.
    vel_translated = egnn_dynamics(t, x_translated)
    assert torch.allclose(vel, vel_translated)

    # Velocity is equivariant w.r.t. permuting two particles of the same type.
    def permute(_x, idx1, idx2):
        _x_permuted = _x.clone()
        _x_permuted[:,idx1*3:(idx1+1)*3] = _x[:,idx2*3:(idx2+1)*3]
        _x_permuted[:,idx2*3:(idx2+1)*3] = _x[:,idx1*3:(idx1+1)*3]
        return _x_permuted

    x_permuted = permute(x, 0, 1)
    vel_permuted = egnn_dynamics(t, x_permuted)
    assert torch.allclose(permute(vel, 0, 1), vel_permuted)

    # Permuting two particles with different type changes the velocity.
    x_permuted = permute(x, 0, 2)
    vel_permuted = egnn_dynamics(t, x_permuted)
    assert not torch.allclose(permute(vel, 0, 2), vel_permuted)


def test_egnn_dynamics_identity():
    """Test that EGNNDynamics can be initialized to output zero velocity."""
    batch_size = 10
    particle_types = torch.tensor([0, 0, 1])
    n_particles = len(particle_types)

    # Create the equivariant graph.
    egnn_dynamics = EGNNDynamics(
        particle_types=particle_types,
        r_cutoff=10.0,
        time_feat_dim=2,
        node_feat_dim=5,
        distance_feat_dim=4,
        n_layers=2,
        initialize_identity=True,
    )

    # Create a random input.
    generator = torch.Generator()
    generator.manual_seed(0)
    t = torch.rand(1, generator=generator)
    x = torch.randn(batch_size, n_particles*3, generator=generator)

    # Run the dynamics.
    vel = egnn_dynamics(t, x)

    # The output is identically 0.0.
    assert torch.allclose(vel, torch.zeros_like(vel))


# =============================================================================
# TESTS EQUIVARIANT GRAPH LAYER
# =============================================================================

def test_eg_layer_compute_distances():
    """Test _EGLayer._compute_distances method."""
    batch_size = 2
    n_particles = 3

    # We compute distances only for a subset of the possible edges.
    edges = [
        torch.tensor([0, 0, 2, 3, 3, 5]),
        torch.tensor([1, 2, 1, 4, 5, 4]),
    ]
    n_edges = len(edges[0])

    # Create random (but reproducible) input and compute distances with torch.
    generator = torch.Generator()
    generator.manual_seed(0)
    x = torch.randn(batch_size, n_particles, 3, generator=generator)
    ref_distances = torch.cdist(x, x)

    # Create graph layer.
    eg_layer = _EGLayer(
        r_cutoff=2.0,
        node_feat_dim=4,
        distance_feat_dim=4,
        speed_factor=1.,
    )

    # Compute the distances and direction with our implementation.
    x_batch_fmt = x.view(batch_size*n_particles, 3)
    distances, directions = eg_layer._compute_distances(x_batch_fmt, edges)
    assert distances.shape == (n_edges,)
    assert directions.shape == (n_edges, 3)

    # Compare distances with pytorch-computed reference.
    for edge_idx, (dest, src) in enumerate(zip(*edges)):
        batch_idx = dest % batch_size
        ref_dest, ref_src = dest % n_particles, src % n_particles
        ref_dist = ref_distances[batch_idx, ref_dest, ref_src]
        assert torch.allclose(ref_dist, distances[edge_idx])

        ref_dir = x[batch_idx, ref_dest] - x[batch_idx, ref_src]
        ref_dir = torch.nn.functional.normalize(ref_dir.unsqueeze(0))[0]
        assert torch.allclose(ref_dir, directions[edge_idx])


def test_eg_layer_prune_edges():
    """Test that _EGLayer._prune_edges discards edges with distance greater than the cutoff."""
    r_cutoff = 1.0

    # We compute distances only for a subset of the possible edges.
    edges = [
        torch.tensor([0, 0, 2, 3, 3, 5]),
        torch.tensor([1, 2, 1, 4, 5, 4]),
    ]

    # Create an input.
    x = torch.tensor([
        # In the first batch sample, consecutive particles are within the
        # cutoff, but non consecutive ones aren't.
        [0, 0, 0.0],
        [0, 0, r_cutoff],
        [0, 0, 2*r_cutoff],
        # In the second batch sample only the first and third atoms are within
        # the cutoff.
        [0.0, 0, 0],
        [5*r_cutoff, 0, 0],
        [r_cutoff, 0, 0],
    ])

    # We expect edges[expected_edges_indices[i]] to be NOT pruned.
    expected_edges_indices = torch.tensor([0, 2, 4])
    n_expected_edges = len(expected_edges_indices)

    # Compute distances and prune them.
    eg_layer = _EGLayer(
        r_cutoff=r_cutoff,
        node_feat_dim=4,
        distance_feat_dim=4,
        speed_factor=1.,
    )
    distances, directions = eg_layer._compute_distances(x, edges)
    p_distances, p_directions, p_edges = eg_layer._prune_edges(distances, directions, edges)

    # Check result of the pruning.
    for arr, ref_arr in zip([p_edges[0], p_edges[1], p_distances, p_directions],
                            [edges[0], edges[1], distances, directions]):
        assert arr.shape[0] == n_expected_edges
        assert torch.allclose(arr, ref_arr[expected_edges_indices])


def test_unsorted_segment_sum():
    """Test the _unsorted_segment_sum function."""
    data = torch.tensor([
        [1.0, 1, 1],
        [2.0, 2, 2],
        [3.0, 3, 3],
        [4.0, 4, 4],
        [5.0, 5, 5],
    ])
    segment_ids = torch.tensor([0, 1, 0, 2, 2])
    n_segments = 3

    expected_result = torch.tensor([
        [4.0, 4, 4],
        [2.0, 2, 2],
        [9.0, 9, 9],
    ])

    result = _unsorted_segment_sum(data, segment_ids, n_segments)
    assert torch.all(result == expected_result)
