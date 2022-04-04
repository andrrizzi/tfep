#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
E(n) equivariant graph neural network for continuous normalizing flows.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import torch

import tfep.nn.graph
from tfep.nn.encoders import GaussianBasisExpansion, BehlerParrinelloRadialExpansion
from tfep.utils.misc import flattened_to_atom, atom_to_flattened


# =============================================================================
# LAYERS
# =============================================================================

class EGNNDynamics(torch.nn.Module):
    """
    E(n) equivariant graph neural network for continuous normalizing flows.

    This implements a variation of the E(n) equivariant graph neural network
    used in [1] as the dynamics in a continuous normalizing flow. First, messages
    are passed only between particles whose distance is smaller than a user-provided
    cutoff.

    Second, the message for each edge is built as an input of the two node features
    (which is initialized as a function of the particle types), their distance,
    and the integration time. Each of these inputs is first encoded into a vector
    of fixed dimension (see parameters ``node_feat_dim``, ``distance_feat_dim``,
    and ``time_feat_dim``), and then concatenated. Specifically:

    - Node features are initialized with a random vector as a function of the
      particle type similarly to SchNet [2].
    - Particle distances are projected onto a Gaussian basis with a cutoff
      switching function using Behler-Parrinello symmetry functions [3].
    - Time is also projected onto a Gaussian basis similarly to a Kernel flow [4].
      For this the time of integration is assumed to have limits 0.0 and 1.0.

    For distances and time, the bandwidth (i.e., standard deviation) of each
    Gaussian is optimized during training.

    Parameters
    ----------
    particle_types : torch.Tensor
        A tensor of integers of shape ``(n_particles,)`` where ``particle_types[i]``
        is the ID of the particle type for the i-th particle. The IDs must start
        from 0 and be consecutive (i.e., ``0 <= particle_types[i] < n_particle_types``).
    r_cutoff : float
        The radial cutoff in the same units used for the coordinate positions
        in the forward pass.
    time_feat_dim : int, optional
        The number of Gaussians used for expanding the time input, which is
        assumed to be in the interval [0, 1].
    node_feat_dim : int, optional
        The dimension of the node feature.
    distance_feat_dim : int, optional
        The number of Gaussians used for expanding the distance input.
    n_layers : int, optional
        The number of message passing layers.
    speed_factor : float, optional
        The output of the dynamics is such that each neighbor contributes to the
        particle velocity with an additive vector with magnitude in ``[-speed_factor,speed_factor]``.
    initialize_identity : bool, optional
        If ``True`` (default), the layers are initialized so that the dynamics
        performs the identity function, which in this context means that outputs
        zero velocities.

    References
    ----------
    [1] Satorras VG, Hoogeboom E, Fuchs FB, Posner I, Welling M. E(n)
        Equivariant Normalizing Flows for Molecule Generation in 3D.
        arXiv preprint arXiv:2105.09016. 2021 May 19.
    [2] Schütt KT, Sauceda HE, Kindermans PJ, Tkatchenko A, Müller KR.
        Schnet–a deep learning architecture for molecules and materials.
        The Journal of Chemical Physics. 2018 Jun 28;148(24):241722.
    [3] Behler J, Parrinello M. Generalized neural-network representation of
        high-dimensional potential-energy surfaces. Physical review letters.
        2007 Apr 2;98(14):146401.
    [4] Köhler J, Klein L, Noé F. Equivariant flows: exact likelihood generative
        learning for symmetric densities. InInternational Conference on Machine
        Learning 2020 Nov 21 (pp. 5361-5370). PMLR.

    """

    def __init__(
        self,
        particle_types,
        r_cutoff,
        time_feat_dim=16,
        node_feat_dim=64,
        distance_feat_dim=64,
        n_layers=4,
        speed_factor=1.0,
        initialize_identity=True,
    ):
        super().__init__()

        # Store the random vectors associated to each particle type as the one-hot
        # encoding h of the particle types and a matrix of parameters W so that
        # the embedding can be computed as dot(W, [h, time]).
        self.particle_types_one_hot = torch.nn.functional.one_hot(particle_types)
        self.time_embedding = GaussianBasisExpansion.from_range(
            n_gaussians=time_feat_dim,
            max_mean=1.0,
            trainable_stds=True,
        )
        self.h_embedding = torch.nn.Linear(
            in_features=len(self.particle_types_one_hot[1]) + time_feat_dim,
            out_features=node_feat_dim,
        )

        # Create the graph layers.
        self._n_layers = n_layers
        for layer_idx in range(n_layers):
            eg_layer = _EGLayer(
                r_cutoff=r_cutoff,
                node_feat_dim=node_feat_dim,
                distance_feat_dim=distance_feat_dim,
                speed_factor=speed_factor
            )

            # Force the identity function if requested.
            if initialize_identity:
                eg_layer.update_x_mlp[-2].weight.data.fill_(0.0)

            self.add_module('graph_layer_'+str(layer_idx), eg_layer)

        # Initializing this requires knowing the number of particles and it is
        # thus initialized lazily and cached on the first forward().
        self._edges = None
        self._n_edges = None

    @property
    def n_particles(self):
        return len(self.particle_types_one_hot)

    def forward(self, t, x):
        """Output the velocity in the continuous normalizing flow.

        Parameters
        ----------
        t : torch.Tensor
            A tensor of shape ``(1,)`` with the time of the integration.
        x : torch.Tensor
            A tensor of shape ``(batch_size, n_particles*3)`` where ``x[b, 3*a+i]``
            is the ``i``-th component of the position vector of the ``a``-th
            particle for batch ``b``.

        Returns
        -------
        v : torch.Tensor
            A tensor of shape ``(batch_size, n_particles*3)`` where ``v[b, 3*a+i]``
            is the ``i``-th component of the velocity vector of the ``a``-th
            particle for batch ``b``.

        """
        batch_size = len(x)

        # x from shape (batch_size, n_particles*3) to (batch_size*n_particles, 3).
        # Internally, we cast everything to a single system of batch_size*n_particles
        # particles and we don't draw edges between samples in the batch to
        # simplify the code.
        vel = x.view(batch_size*self.n_particles, 3)

        # Node features of size (batch_size*n_particles, node_feat_dim).
        h = self._create_node_embedding(t, batch_size)

        # Get the edges.
        if self._edges is None:
            # Generate all the possible edges between the nodes and cache them.
            edges = tfep.nn.graph.get_all_edges(batch_size, self.n_particles)
            self._edges = edges
            self._n_edges = int(edges.shape[1]) // batch_size
        else:
            # The last batch might have a smaller batch size than the one cached.
            edges = tfep.nn.graph.reduce_edges_batch_size(self._edges, batch_size, self._n_edges)

        # Run through the graph.
        for layer_idx in range(self._n_layers):
            eg_layer = self._modules['graph_layer_'+str(layer_idx)]
            h, vel = eg_layer(h, vel, edges)

        # Reshape the velocities back in batch format.
        vel = vel.view(batch_size, self.n_particles*3)

        # To obtain a translational invariant velocity, we subtract the initial
        # positions.
        vel = vel - x

        # Remove the mean of the velocity so that the center of geometry is
        # preserved and the transformation is regularized.
        vel_atom_fmt = flattened_to_atom(vel)
        vel_mean = torch.mean(vel_atom_fmt, dim=1, keepdim=True)
        vel = atom_to_flattened(vel_atom_fmt - vel_mean)
        return vel

    def _create_node_embedding(self, t, batch_size):
        """Return the node features.

        Returns
        -------
        h : torch.Tensor
            A torch of shape ``(batch_size*n_particles, node_feature_dimension)``.

        """
        # Initialize node invariant features by concatenating the one-hot features
        # encoding particle types with the soft one-hot encoding representing time.
        # GaussianBasisExpansion requires a tensor with at least 1 dimension.
        t_embedded = self.time_embedding(t.unsqueeze(0))
        # t_embedded from shape (time_feat_dim,) to (n_particles, time_feat_dim).
        t_embedded = t_embedded.expand(self.n_particles, -1)
        # h has shape (n_particles, n_particle_types+time_feat_dim).
        h = torch.cat([self.particle_types_one_hot, t_embedded], dim=-1)

        # Now assign to the one-hot representations the embedding parameters.
        h = self.h_embedding(h)
        # h from shape (n_particles, node_feat_dim) to (batch_size*n_particles, node_feat_dim).
        h = h.repeat(batch_size, 1)

        return h


class _EGLayer(torch.nn.Module):
    """Equivariant graph neural network layer."""

    def __init__(self, r_cutoff, node_feat_dim, distance_feat_dim, speed_factor):
        super().__init__()

        self.speed_factor = speed_factor

        # Embedding layer used to expand distances in vector features.
        self.distance_embedding = BehlerParrinelloRadialExpansion.from_range(
            r_cutoff=r_cutoff,
            n_gaussians=distance_feat_dim,
            max_mean=r_cutoff,
            trainable_stds=True,
            # We'll remove radii > r_cutoff before passing the input to this so
            # we can set force_zero_after_cutoff to False.
            force_zero_after_cutoff=False,
        )

        # Layer used to create the message for each edge.
        self.message_mlp = torch.nn.Sequential(
            torch.nn.Linear(2*node_feat_dim + distance_feat_dim, node_feat_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(node_feat_dim, node_feat_dim),
            torch.nn.SiLU(),
        )

        # Attention layer used to weight messages.
        self.attention_mlp = torch.nn.Sequential(
            torch.nn.Linear(node_feat_dim, 1),
            torch.nn.Sigmoid(),
        )

        # Network used to compute the displacement magnitudes to update the positions.
        self.update_x_mlp = torch.nn.Sequential(
            torch.nn.Linear(node_feat_dim, node_feat_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(node_feat_dim, 1, bias=False),
            torch.nn.Tanh(),
            # The original implementation in [1] also multiplied the results of this by
            # a constant so that the output range was [0, constant) instead of [0,1).
            # In this application, the perturbation is very small so [0,1) should suffice.
        )

        # Network used to compute the update of the node features.
        self.update_h_mlp = torch.nn.Sequential(
            torch.nn.Linear(2*node_feat_dim, node_feat_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(node_feat_dim, node_feat_dim),
        )

    def forward(self, h, x, edges):
        """Propagate the equivariant graph layer.

        Parameters
        ----------
        h : torch.Tensor
            Scalar node features with shape ``(batch_size*n_particles, node_feat_dim)``.
        x : torch.Tensor
            Vector node features with shape ``(batch_size*n_particles, 3)``.
        edges : List[torch.Tensor]
            A list of two tensors, both of shape ``(n_edges,)``. The i-th edge
            is created from node ``x[edges[0][i]]`` to ``x[edges[1][i]]``. The edge
            is directional so if a message must be passed in both directions, two
            entries are needed.

        """
        # Compute distances and unit diff vectors between particles positions.
        # distances has shape (n_edges,) and directions (n_edges, 3).
        # The original implementation in [1] normalized the directions by
        # distance + 1e-8 to avoid division by 0.0. In this application, it is
        # unlikely that two atoms will overlap in the middle of the training and
        # if it happens the SCF won't converge anyway.
        distances, directions = tfep.nn.graph.compute_edge_distances(
            x, edges, normalize_directions=True)

        # Identify the edges that are within the cutoff and discard the others.
        edges, distances, directions = tfep.nn.graph.prune_long_edges(
            self.distance_embedding.r_cutoff, edges, distances, directions)

        # Create the messages between edges.
        edge_messages = self._create_edge_messages(h, distances, edges)

        # Update node scalar and vector features.
        h  = self._update_h(h, edge_messages, edges)
        x = self._update_x(x, edge_messages, directions, edges)

        return h, x

    def _create_edge_messages(self, h, distances, edges):
        """Create a message for each edge.

        Parameters
        ----------
        h : torch.Tensor
            Node features of shape (batch_size*n_particles, node_feat_dim).
        distances : torch.Tensor
            Distances between particles of shape (n_edges,)

        Returns
        -------
        edge_messages : torch.Tensor
            A message for each edge of shape (n_edges, node_feat_dim).
        """
        # Embed distances. dist_embedding has shape (n_edges, distance_feat_dim).
        dist_embedding = self.distance_embedding(distances)

        # Concatenate node features and distance information and create the messages.
        input = torch.cat([h[edges[0]], h[edges[1]], dist_embedding], dim=1)
        edge_messages = self.message_mlp(input)

        # Create the attention coefficient.
        attention = self.attention_mlp(edge_messages)
        edge_messages = edge_messages * attention

        return edge_messages

    def _update_h(self, h, edge_messages, edges):
        # Aggregate all messages with destination edges[1]. Like h,
        # node_messages has shape (batch_size*n_particles, node_feat_dim).
        src, dest = edges
        node_messages = tfep.nn.graph.unsorted_segment_sum(edge_messages, dest, h.shape[0])

        # Concatenate the current h and the aggregated message and feed them
        # to the node-update MLP. out has shape (batch_size*n_particles, node_feat_dim).
        input = torch.cat([h, node_messages], dim=1)
        out = self.update_h_mlp(input)

        # As in [1], the output of the MLP is a residual.
        return h + out

    def _update_x(self, x, edge_messages, directions, edges):
        # Compute the magnitude of the displacements. edge_messages has shape
        # (n_edges, node_feat_dim) and disp_magnitude is (n_edges, 1).
        disp_magnitude = self.update_x_mlp(edge_messages)
        # The original implementation in [1] also multiplied the results of this by
        # a constant so that the magnitude was in [0, constant) instead of [0,1).
        # In this application, the perturbation is very small so [0,1) should suffice.

        # Compute displacements. directions has shape (n_edges, 3).
        disp = self.speed_factor * directions * disp_magnitude

        # Aggregate displacement. aggregate_disp has shape (batch_size*n_particles, 3).
        src, dest = edges
        aggregate_disp = tfep.nn.graph.unsorted_segment_sum(disp, dest, x.shape[0])

        # Add the displacement.
        return x + aggregate_disp
