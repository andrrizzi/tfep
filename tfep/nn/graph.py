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

class FixedGraph(torch.nn.Module):
    """Graph base class with a fixed topology.

    The class provides utilities to determine the edges between nodes (optionally
    based on a mask) and cache them. It is assumed that the edges do not change
    after constructions.

    The edges must be accessed by the parent class through the ``get_edges()``
    method, which takes care of determining the edges compatible with features
    in the shape ``(batch_size*n_nodes, n_feats_per_node)``.

    Parameters
    ----------
    n_nodes : int
        The number of nodes in the graph.
    mask : torch.Tensor, optional
        Shape ``(n_nodes, n_nodes)``. A (directional) edge from node ``i`` to
        node ``j`` is created only if ``mask[i, j] != 0``. If ``mask`` is not
        provided, all nodes are connected to all nodes (excluding self interactions).

    """

    def __init__(self, n_nodes, mask=None):
        super().__init__()
        # We initially build the edges for a batch of size 1 so at this point
        # self._last_batch_edges will have shape (2, n_edges). However, we'll
        # use this to cache the number of edges for the latest batch size.
        self._last_batch_edges = get_all_edges(1, n_nodes, mask=mask)
        # The number of nodes and edges (for a single batch) is needed for get_edges().
        self._n_nodes = n_nodes
        self._n_edges = int(self._last_batch_edges.shape[1])

    @property
    def n_nodes(self):
        """int: Number of nodes in the graph."""
        return self._n_nodes

    def get_edges(self, batch_size):
        """Return the edges between nodes for the given batch size.

        Parameters
        ----------
        batch_size : int
            The size of the current batch.

        Returns
        -------
        edges : torch.Tensor
            Shape ``(2, n_edges*batch_size)``. The ``i``-th edge is created from
            node ``edges[0][i]`` to ``edges[1][i]``, where ``edges[0][i]`` is a
            node index in the range ``[0, batch_size*n_nodes]``.

            Edges are directional so if a message must be passed in both directions,
            two entries connecting the nodes with inverse order are present.

        """
        # Store new edges so that if the next batch is the same this will be faster.
        self._last_batch_edges = fix_edges_batch_size(
            self._last_batch_edges, batch_size, self._n_edges, n_nodes=self._n_nodes)
        return self._last_batch_edges


def get_all_edges(batch_size, n_nodes, mask=None):
    """Return all possible edges between nodes after applying the mask.

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
        Shape ``(2, batch_size*n_edges)``. The ``i``-th edge is created from node
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

    # Determine the edges for the given shape.
    n_edges = edges.shape[1]
    edges = fix_edges_batch_size(edges, batch_size, n_edges, n_nodes)

    return edges


def fix_edges_batch_size(edges, new_batch_size, n_edges, n_nodes=None):
    """Return edges compatible with with features in shape ``(new_batch_size*n_nodes, n_feats)``.

    This takes previously cached edges for batch size ``old_batch_size`` and
    reformat them to be compatible with ``new_batch_size`. This happens, for
    example, for the last batch, which could be smaller than the one used to
    create the edges cached during the first forward pass.

    Parameters
    ----------
    edges : torch.Tensor
        Shape ``(2, old_batch_size*n_edges)``, where ``n_edges`` are the number
        of edges between nodes in the graph. This is the output of :func:`.get_all_edges()`.
    new_batch_size : int
        The output batch size.
    n_edges : int
        The number of edges between nodes in the graph.
    n_nodes : int, optional
        The number of nodes in the graph. This is only required if
        ``new_batch_size > old_batch_size``.

    Returns
    -------
    edges : torch.Tensor
        Shape ``(2, new_batch_size*n_edges)``. The new edges after fixing the
        batch size.

    """
    n_returned_edges = new_batch_size * n_edges

    if edges.shape[1] == n_returned_edges:
        return edges

    if edges.shape[1] > n_returned_edges:
        # Cut extra batches.
        edges = edges[:, :n_returned_edges]
    else:
        # Add more batches. First determine the edges for the first batch.
        if edges.shape[1] > n_edges:
            edges = edges[:, :n_edges]

        # To shape (2, batch_size, n_edges).
        edges = edges.unsqueeze(1).expand(-1, new_batch_size, -1)

        # Now multiply repeated node indices by the batch index. shift has shape (1, batch_size, 1).
        shift = torch.arange(0, new_batch_size*n_nodes, n_nodes).unsqueeze(-1)
        edges = edges + shift

        # To shape (2, batch_size*n_edges)
        n_edges = edges.shape[-1]
        edges = edges.reshape((2, new_batch_size*n_edges))

    return edges


# TODO: MERGE THIS WITH tfep.utils.geometry.pdist?
def compute_edge_distances(x, edges, normalize_directions=False, inverse_directions=False):
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
    inverse_directions : bool, optional
        If ``True``, the direction vectors go from the destination node to the
        source node rather than the opposite.

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
    if inverse_directions:
        directions = x[edges[0]] - x[edges[1]]
    else:
        directions = x[edges[1]] - x[edges[0]]

    distances = torch.sqrt(torch.sum(directions**2, dim=-1))

    if normalize_directions:
        directions = directions / distances.unsqueeze(-1)
    return distances, directions


def prune_long_edges(r_cutoff, edges, distances, *args):
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
    *args : Sequence[torch.Tensor]
        Other tensors of shape ``(batch_size*n_edges, *)`` to prune in the same
        way.

    Returns
    -------
    edges : torch.Tensor
        Shape ``(2, batch_size*n_pruned_edges)``. The edges after the pruning.
    distances : torch.Tensor, optional
        Shape ``(batch_size*n_pruned_edges,)``. The distances of the nodes across
        the edges after the pruning.
    *other : torch.Tensor, optional
        Other pruned tensors of shape ``(batch_size*n_pruned_edges, *)``.

    """
    mask = distances <= r_cutoff
    edges = edges[:, mask]
    distances = distances[mask]
    pruned_args = [arg[mask] for arg in args]
    return edges, distances, *pruned_args


def unsorted_segment_sum(data, segment_ids, n_segments):
    """Replicates TensorFlow's tf.math.unsorted_segment_sum in PyTorch."""
    segment_sum = data.new_full((n_segments, data.shape[1]), fill_value=0.0)
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.shape[1])
    segment_sum.scatter_add_(0, segment_ids, data)
    return segment_sum
