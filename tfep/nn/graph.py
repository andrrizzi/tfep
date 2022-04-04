#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Utility functions for graphs.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import itertools

import torch


# =============================================================================
# UTILITIES
# =============================================================================

def get_all_edges(batch_size, n_nodes, mask=None):
    """Return all possible edges between nodes after applying the mask.

    The function caches the returned edges in the self._edges attribute.

    Parameters
    ----------
    batch_size : int
        The batch size.
    n_nodes : int
        The number of nodes in the graph.
    mask : torch.Tensor, optional
        Shape ``(n_nodes, n_nodes)``. A (directional) edge from node ``i`` to
        node ``j`` is created only if ``mask[i, j] != 0``. If ``mask`` is not
        provided, all nodes are connected to all nodes (excluding self interactions).

    Returns
    -------
    edges : torch.Tensor
        Shape ``(2, n_edges*batch_size)``. The ``i``-th edge is created from node
        ``edges[0][i]`` to ``edges[1][i]``, where ``edges[0][i]`` is a node index
        in the range ``[0, batch_size*n_nodes]``.

        Edges are directional so if a message must be passed in both directions,
        two entries connecting the nodes with inverse order are present.

    """
    # Determine the edges for a single batch. edges has shape (2, n_edges).
    if mask is None:
        if n_nodes == 1:
            return torch.empty(2, 0)

        edges = itertools.permutations(range(n_nodes), 2)
        edges = torch.tensor(list(zip(*edges)))
    else:
        if mask.shape != (n_nodes, n_nodes):
            raise ValueError('mask must have shape (n_nodes, n_nodes)')

        edges = mask.nonzero()
        edges = edges.t()

    # To shape (2, batch_size, n_edges).
    edges = edges.unsqueeze(1).expand(-1, batch_size, -1)

    # Now multiply repeated node indices by the batch index. shift has shape (1, batch_size, 1).
    shift = torch.arange(0, batch_size*n_nodes, n_nodes).unsqueeze(-1)
    edges = edges + shift

    # To shape (2, batch_size*n_edges)
    n_edges = edges.shape[-1]
    edges = edges.reshape((2, batch_size*n_edges))

    return edges


def reduce_edges_batch_size(edges, new_batch_size, n_edges):
    """Reduce the number of edges for the given batch size.

    As the last batch could be smaller than the one used to create the edges with
    :func:`.get_all_edges()`, this takes care of fixing the edges to remove the
    extra batches.

    Parameters
    ----------
    edges : torch.Tensor
        Shape ``(2, n_edges)``. The edges are returned by :func:`.get_all_edges()`
    new_batch_size : int
        The output batch size.
    n_edges : int
        The number of edges between nodes in the graph.

    Returns
    -------
    edges : torch.Tensor
        Shape ``(2, new_batch_size*n_edges)``. The new edges after removing the
        extra batches.

    """
    n_returned_edges = new_batch_size * n_edges
    if edges.shape[1] > n_returned_edges:
        return edges[:, :n_returned_edges]
    return edges


# TODO: MERGE THIS WITH tfep.utils.geometry.pdist?
def compute_edge_distances(x, edges, normalize_directions=False):
    """Return distances between nodes across edges.

    Parameters
    ----------
    x : torch.Tensor
        Positions of particles with shape (batch_size*n_nodes, 3).
    edges : torch.Tensor
        Shape ``(2, batch_size*n_edges)``. The ``i``-th edge is created from node
        ``edges[0][i]`` to ``edges[1][i]``, where ``edges[0][i]`` is a node index
        in the range ``[0, batch_size*n_nodes]``.
    normalize_directions : bool, optional
        If ``True``, the returned directions are normalized by their norms.

    Returns
    -------
    distances : torch.Tensor
        Shape ``(batch_size*n_edges,)``. ``distances[i]`` is the distance between
        the nodes of the ``i``-th edge.
    direction : torch.Tensor
        Shape ``(batch_size*n_edges, 3)``. ``direction[i]`` is the vector connecting
        the nodes of the ``i``-th edge. The vectors are normalized by their norms
        if ``normalize_direction`` is ``True``.

    """
    directions = x[edges[1]] - x[edges[0]]
    distances = torch.sqrt(torch.sum(directions**2, dim=-1))
    if normalize_directions:
        directions = directions / distances.unsqueeze(-1)
    return distances, directions


def prune_long_edges(r_cutoff, edges, distances, directions):
    """Detect which edges have distances larger than the cutoff and remove them.

    Parameters
    ----------
    r_cutoff : float
        The radial cutoff. All edges connecting nodes at distance greater than
        this cutoff will be pruned. This must be in the same units as the nodes
        positions.
    edges : torch.Tensor
        Shape ``(2, batch_size*n_edges)``. The ``i``-th edge is created from node
        ``edges[0][i]`` to ``edges[1][i]``, where ``edges[0][i]`` is a node index
        in the range ``[0, batch_size*n_nodes]``.
    distances : torch.Tensor
        Shape ``(batch_size*n_edges,)``. ``distances[i]`` is the distance between
        the nodes of the ``i``-th edge.
    direction : torch.Tensor
        Shape ``(batch_size*n_edges, 3)``. ``direction[i]`` is the direction
        connecting the nodes of the ``i``-th edge.

    Returns
    -------
    edges : torch.Tensor
        Shape ``(2, batch_size*n_pruned_edges)``. The edges after the pruning.
    distances : torch.Tensor, optional
        Shape ``(batch_size*n_pruned_edges,)``. The distances of the nodes across
        the edges after the pruning.
    direction : torch.Tensor, optional
        Shape ``(batch_size*n_pruned_edges, 3)``. The edge directions after the
        pruning.

    """
    mask = distances <= r_cutoff
    edges = edges[:, mask]
    distances = distances[mask]
    directions = directions[mask]
    return edges, distances, directions


def unsorted_segment_sum(data, segment_ids, n_segments):
    """Replicates TensorFlow's tf.math.unsorted_segment_sum in PyTorch."""
    segment_sum = data.new_full((n_segments, data.shape[1]), fill_value=0.0)
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.shape[1])
    segment_sum.scatter_add_(0, segment_ids, data)
    return segment_sum
