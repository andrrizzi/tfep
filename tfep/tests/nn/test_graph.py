#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test classes and functions in tfep.nn.graph.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import pytest
import torch

from tfep.nn.masked import create_autoregressive_mask
from tfep.nn.graph import (
    FixedGraph,
    get_all_edges,
    fix_node_indices_batch_size,
    compute_edge_distances,
    prune_long_edges,
    unsorted_segment_sum,
)


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
# UTILS
# =============================================================================

def fixed_graph_get_edges(batch_size, n_nodes, mask):
    """A utility function wrapping a call to the method FixedGraph.get_edges()."""
    graph = FixedGraph(node_types=[0]*n_nodes, mask=mask)
    return graph.get_edges(batch_size)


# =============================================================================
# TESTS
# =============================================================================

@pytest.mark.parametrize('node_types,expected_encoding', [
    ([0, 1, 0], [[1, 0], [0, 1], [1, 0]]),
    ([3, 1], [[0, 0, 0, 1], [0, 1, 0, 0]])
])
def test_graph_node_type_encoding(node_types, expected_encoding):
    """Test the one hot encoding of the node types."""
    graph = FixedGraph(node_types)
    assert torch.allclose(
        graph._node_types_one_hot,
        torch.tensor(expected_encoding).to(graph._node_types_one_hot)
    )


@pytest.mark.parametrize('batch_size', [1, 5])
@pytest.mark.parametrize('n_nodes', [1, 3])
@pytest.mark.parametrize('autoregressive_mask', [False, True])
@pytest.mark.parametrize('func', [get_all_edges, fixed_graph_get_edges])
def test_get_edges(batch_size, n_nodes, autoregressive_mask, func):
    """Test that get_all_edges() and FixedGraph.get_edges() determine all the masked edges of the graph."""
    # Create the mask.
    if autoregressive_mask is True:
        degrees_nodes = torch.arange(0, n_nodes)
        mask = create_autoregressive_mask(degrees_nodes, degrees_nodes)
    else:
        mask = None

    # Compute all masked edges.
    edges = func(batch_size, n_nodes, mask)

    # Check that no mask and full mask are the same thing.
    if mask is None:
        mask = -(torch.eye(n_nodes)-1)
        edges2 = func(batch_size, n_nodes, mask=mask)
        assert torch.all(edges == edges2)

    # Check that we have the expected number of edges.
    n_edges = int(mask.sum())
    assert edges.shape == (2, n_edges * batch_size)

    # Check that the edges are correct.
    for edge_src, edge_dst in edges.t():
        batch_idx  = edge_src.tolist() // n_nodes
        assert batch_idx < batch_size

        edge_src = edge_src - batch_idx * n_nodes
        edge_dst = edge_dst - batch_idx * n_nodes

        assert mask[edge_src, edge_dst] == 1.0

    # Check that fix_node_indices_batch_size correctly fixes the number of batches.
    new_batch_size = 2
    expected_edges = func(new_batch_size, n_nodes, mask)
    fixed_edges = fix_node_indices_batch_size(
        node_indices=edges[:, :n_edges],
        batch_size=new_batch_size,
        n_nodes=n_nodes,
    )
    assert fixed_edges.shape == (2, new_batch_size*n_edges)
    assert torch.all(expected_edges == fixed_edges)


def test_graph_get_edges_caching():
    """Test that the caching in FixedGraph.get_edges() works correctly."""
    node_types = [0, 2, 1, 0]

    n_nodes = len(node_types)
    degrees_nodes = torch.arange(0, n_nodes)

    # Create the graph.
    graph = FixedGraph(
        node_types=node_types,
        mask=create_autoregressive_mask(degrees_nodes, degrees_nodes),
    )

    # On init, graph caches the edges for batch_size 1.
    expected_n_edges = n_nodes * (n_nodes-1) // 2
    assert graph._n_edges == expected_n_edges
    assert graph._last_batch_edges.shape[1] == expected_n_edges

    # Get first edges.
    batch_size = 3
    edges1 = graph.get_edges(batch_size=batch_size)

    # Calls with the same batch size return the same edges.
    edges = graph.get_edges(batch_size=batch_size)
    assert edges1.shape[1] == batch_size * expected_n_edges
    assert torch.all(edges1 == edges)

    # The last batch might be shorter than the previous one.
    edges = graph.get_edges(batch_size=batch_size-1)
    assert edges.shape[1] == (batch_size-1) * expected_n_edges
    assert torch.all(edges1[:, :edges.shape[1]] == edges)

    # The first batch of the new epoch might be longer than the previous one.
    edges = graph.get_edges(batch_size=batch_size)
    assert edges1.shape[1] == batch_size * expected_n_edges
    assert torch.all(edges1 == edges)


def test_compute_edge_distances():
    """Test compute_edge_distances() function."""
    batch_size = 2
    n_particles = 3

    # We compute distances only for a subset of the possible edges.
    edges = torch.tensor([
        [0, 0, 2, 3, 3, 5],
        [1, 2, 1, 4, 5, 4],
    ])
    n_edges = len(edges[0])

    # Create random (but reproducible) input and compute distances with torch.
    generator = torch.Generator()
    generator.manual_seed(0)
    x = torch.randn(batch_size, n_particles, 3, generator=generator)
    ref_distances = torch.cdist(x, x)

    # Compute the distances and direction with our implementation.
    x_batch_fmt = x.view(batch_size*n_particles, 3)
    distances, directions = compute_edge_distances(
        x_batch_fmt, edges, normalize_directions=True)
    assert distances.shape == (n_edges,)
    assert directions.shape == (n_edges, 3)

    # Compare distances with pytorch-computed reference.
    for edge_idx, (src, dest) in enumerate(edges.t()):
        batch_idx = int(src) // n_particles
        ref_dest, ref_src = dest % n_particles, src % n_particles
        ref_dist = ref_distances[batch_idx, ref_src, ref_dest]
        assert torch.allclose(ref_dist, distances[edge_idx])

        ref_dir = x[batch_idx, ref_dest] - x[batch_idx, ref_src]
        ref_dir = torch.nn.functional.normalize(ref_dir.unsqueeze(0))[0]
        assert torch.allclose(ref_dir, directions[edge_idx])

    # Check normalization and inverse direction.
    distances2, directions2 = compute_edge_distances(
        x_batch_fmt, edges, normalize_directions=False, inverse_directions=True)
    directions2 = directions2 / distances2.unsqueeze(-1)
    assert torch.allclose(directions, -directions2)


def test_prune_long_edges():
    """Test that prune_long_edges() discards edges with distance greater than the cutoff."""
    r_cutoff = 1.0

    # We compute distances only for a subset of the possible edges.
    edges = torch.tensor([
        [0, 0, 2, 3, 3, 5],
        [1, 2, 1, 4, 5, 4],
    ])

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
    distances, directions = compute_edge_distances(x, edges)
    p_edges, p_distances, p_directions = prune_long_edges(r_cutoff, edges, distances, directions)

    # Check result of the pruning.
    for arr, ref_arr in zip([p_edges[0], p_edges[1], p_distances, p_directions],
                            [edges[0], edges[1], distances, directions]):
        assert arr.shape[0] == n_expected_edges
        assert torch.allclose(arr, ref_arr[expected_edges_indices])


def test_unsorted_segment_sum():
    """Test the unsorted_segment_sum function."""
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

    result = unsorted_segment_sum(data, segment_ids, n_segments)
    assert torch.all(result == expected_result)
