#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Embedding layers for masked autoregressive flows (MAF).
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import abc
from collections.abc import Sequence
from typing import Optional

import torch

from tfep.utils.misc import ensure_tensor_sequence, remove_and_shift_sorted_indices


# =============================================================================
# BASE CLASS
# =============================================================================

class MAFEmbedding(abc.ABC, torch.nn.Module):
    """An embedding layer for a masked autoregressive flow.

    This class documents the API of an embedding layer compatible with
    :class:`tfep.nn.flows.maf.MAF`.

    """

    @abc.abstractmethod
    def get_degrees_out(self, degrees_in: torch.Tensor) -> torch.Tensor:
        """Return the degrees of the features after the forward pass.

        These are the degrees that will be passed as input to the conditioner.

        The periodic features after the forward are represented as 2 features
        (cosine and sine) that both are assigned the same degree as the input
        feature.

        Parameters
        ----------
        degrees_in : torch.Tensor
            The degrees of the input features.

        Returns
        -------
        degrees_out : torch.Tensor
            The degrees of the features after the forward pass.
        """
        pass


# =============================================================================
# PERIODIC EMBEDDING
# =============================================================================

class PeriodicEmbedding(MAFEmbedding):
    """Lift periodic degrees of freedom into a periodic representation (cos, sin)."""

    def __init__(
            self,
            n_features_in : int,
            limits : torch.Tensor,
            periodic_indices : Optional[Sequence[int]] = None,
    ):
        """Constructor.

        Parameters
        ----------
        n_features_in : int
            Number of input features.
        limits : torch.Tensor[float]
            A pair ``(lower, upper)`` defining the limits of the periodic
            variables. The period is given by ``upper - lower``.
        periodic_indices : Sequence[int] or None, optional
            Shape (n_periodic,). The (ordered) indices of the input features
            that are periodic and must be lifted to the (cos, sin)
            representation. If ``None``, all features are embedded.

        """
        super().__init__()

        # Limits.
        self.register_buffer('limits', ensure_tensor_sequence(limits))

        # Input periodic feature indices.
        if periodic_indices is None:
            periodic_indices = torch.arange(n_features_in)
        else:
            periodic_indices = ensure_tensor_sequence(periodic_indices)
        self.register_buffer('_periodic_indices_in', periodic_indices)

        # We cache both periodic and nonperiodic indices in input and output.
        degrees_in = torch.zeros(n_features_in).to(periodic_indices)
        degrees_in[self._periodic_indices_in] = 1
        degrees_out = self.get_degrees_out(degrees_in)
        self.register_buffer('_periodic_indices_out', (degrees_out == 1).nonzero().flatten())
        self.register_buffer('_nonperiodic_indices_in', (degrees_in == 0).nonzero().flatten())
        self.register_buffer('_nonperiodic_indices_out', (degrees_out == 0).nonzero().flatten())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Lift each periodic degree of freedom x into a periodic representation (cosx, sinx).

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch_size, n_features)``. Input tensor.

        Returns
        -------
        y : torch.Tensor
            Shape ``(batch_size, n_features + n_periodic)``. The input with the
            periodic DOFs transformed. The cosx, sinx representation is placed
            contiguously where the original DOF was. E.g., if ``2`` is the first
            element in ``periodic_indices``, then cos and sin will be placed at
            ``y[:, 2]`` and ``y[:, 3]`` respectively.
        """
        batch_size, n_features = x.shape

        # Transform periodic interval to [0, 2pi].
        period_scale = 2*torch.pi / (self.limits[1] - self.limits[0])
        x_periodic = (x[:, self._periodic_indices_in] - self.limits[0]) * period_scale
        x_embedded = torch.stack([
            torch.cos(x_periodic),
            torch.sin(x_periodic),
        ], dim=2).reshape(batch_size, -1)

        # Fill output.
        n_periodic = len(self._periodic_indices_in)
        y = torch.empty((batch_size, n_features+n_periodic)).to(x)
        y[:, self._periodic_indices_out] = x_embedded
        y[:, self._nonperiodic_indices_out] = x[:, self._nonperiodic_indices_in]

        return y

    def get_degrees_out(self, degrees_in: torch.Tensor) -> torch.Tensor:
        """Return the degrees of the features after the forward pass.

        These are the degrees that will be passed as input to the conditioner.

        The periodic features after the forward are represented as 2 features
        (cosine and sine) that both are assigned the same degree as the input
        feature.

        Parameters
        ----------
        degrees_in : torch.Tensor
            The degrees of the input features.

        Returns
        -------
        degrees_out : torch.Tensor
            The degrees of the features after the forward pass.

        """
        repeats = torch.ones_like(degrees_in)
        repeats[self._periodic_indices_in] = 2
        return torch.repeat_interleave(degrees_in, repeats)


# =============================================================================
# FLIP-INVARIANT EMBEDDING
# =============================================================================

class FlipInvariantEmbedding(MAFEmbedding):
    """Embeds vector features into a representation invariant to sign flips.

    This implements the embedding proposed in [1] (Equation 46 in the SI).

    References
    ----------
    [1] Köhler J, Invernizzi M, De Haan P, Noé F. Rigid body flows for sampling
        molecular crystal structures. In International Conference on Machine
        Learning 2023 Jul 3 (pp. 17301-17326). PMLR.

    """

    def __init__(
            self,
            n_features_in : int,
            embedding_dimension : int,
            embedded_indices : Optional[Sequence[int]] = None,
            vector_dimension : int = 4,
            hidden_layer_width : int = 32,
    ):
        """Constructor.

        Parameters
        ----------
        n_features_in : int
            Number of input features (embedded and not).
        embedding_dimension : int
            The embedding dimension of each vector.
        embedded_indices : Sequence[int] or None, optional
            A sequence of length ``n_vectors*vector_dimension`` with the
            (ordered) indices of the input features corresponding to the
            vectors to embed. Vectors are assumed to be represented by
            consecutive elements. If ``None``, all features are embedded.
        vector_dimension : int, optional
            The dimension of the embedded vectors. Default is 4.
        hidden_layer_width : int, optional
            The width of the hidden layer of the fully-connected neural
            networks used to embed the vectors. Default is 32.

        """
        super().__init__()

        if embedded_indices is None:
            embedded_indices = torch.arange(n_features_in)
        else:
            embedded_indices = ensure_tensor_sequence(embedded_indices)
        self.register_buffer('_embedded_indices_in', embedded_indices)

        # We need to cache both embedded and non-embedded indices
        # before and after the embedding.
        all_indices = torch.arange(n_features_in)
        self.register_buffer('_nonembedded_indices_in', remove_and_shift_sorted_indices(
            all_indices, removed_indices=embedded_indices, shift=False))

        # Indices of the first component of the embedded input vectors.
        vectors_first_indices_in = embedded_indices.reshape(-1, vector_dimension)[:, 0].tolist()

        # Find embedded vectors indices after the embedding.
        embedded_indices_out = []
        dimension_diff = embedding_dimension - vector_dimension
        shift_idx = 0
        for first_idx_in in vectors_first_indices_in:
            first_idx_out = shift_idx + first_idx_in
            vector_indices_out = range(first_idx_out, first_idx_out+embedding_dimension)
            embedded_indices_out.extend(list(vector_indices_out))
            shift_idx += dimension_diff
        self.register_buffer('_embedded_indices_out', torch.tensor(embedded_indices_out))

        # Find the non-embedded feature indices after the embedding.
        n_features_diff = len(vectors_first_indices_in) * dimension_diff
        all_indices = torch.arange(n_features_in + n_features_diff)
        self.register_buffer('_nonembedded_indices_out', remove_and_shift_sorted_indices(
            all_indices, removed_indices=self._embedded_indices_out, shift=False))

        # Create NNP layers for the embedding.
        self.embedding_layer = torch.nn.Sequential(
            torch.nn.Linear(vector_dimension, hidden_layer_width),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_layer_width, embedding_dimension),
        )
        self.weight_layer = torch.nn.Sequential(
            torch.nn.Linear(vector_dimension, hidden_layer_width),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_layer_width, 1),
            torch.nn.Softmax(),
        )

    @property
    def vector_dimension(self) -> int:
        """int: The input vector dimensionality."""
        return self.embedding_layer[0].in_features

    @property
    def embedding_dimension(self) -> int:
        """int: The embedding dimension for each vector."""
        return self.embedding_layer[-1].out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, n_features)``. Input tensor.

        Returns
        -------
        y : torch.Tensor
            Shape ``(batch, n_features + n_vectors*(embedded_dim - vector_dim))``.
            The transformed features with the ``n_vectors`` vectors embedded
            from the ``vector_dim`` to an ``embedded_dim`` space.

        """
        batch_size = x.shape[0]

        # Extract vectors.
        vectors = x[:, self._embedded_indices_in]

        # From (batch, n_vectors*vector_dim) to (batch*n_vectors, vector_dim).
        vectors = vectors.reshape(-1, self.vector_dimension)

        # Find the positive and negative embeddings.
        # Out shape is (batch*n_vectors, 2, embedding_dim).
        embedded_vectors = torch.stack([
            self.embedding_layer(vectors),
            self.embedding_layer(-vectors)
        ], dim=1)

        # Find the softmax weights for the positive and negative embeddings.
        # Out shape is (batch*n_vectors, 2, 1).
        weights = torch.stack([
            self.weight_layer(vectors),
            self.weight_layer(-vectors),
        ], dim=1)

        # Weighted sum the positive and negative embeddings.
        # Out shape is (batch*n_vectors, embedding_dim)
        embedded_vectors = (weights * embedded_vectors).sum(dim=1)

        # From (batch*n_vectors, embedding_dim) to (batch, n_vectors*embedding_dim).
        embedded_vectors = embedded_vectors.reshape(batch_size, -1)

        # Construct output tensor.
        n_features_out = len(self._nonembedded_indices_out) + len(self._embedded_indices_out)
        out = torch.empty((batch_size, n_features_out)).to(x)
        out[:, self._embedded_indices_out] = embedded_vectors
        out[:, self._nonembedded_indices_out] = x[:, self._nonembedded_indices_in]

        return out

    def get_degrees_out(self, degrees_in: torch.Tensor) -> torch.Tensor:
        """Return the degrees of the features after the forward pass.

        This requires that all components of each vector is assigned a single
        degree.

        Parameters
        ----------
        degrees_in : torch.Tensor
            Shape ``(n_features_in,)``. The degrees of the input features.

        Returns
        -------
        degrees_out : torch.Tensor
            Shape ``(n_features_out,)``. The degrees of the features after the
            forward pass.

        Raises
        ------
        ValueError
            If there are some embedded vectors whose components have been
            assigned different degrees.

        """
        # Get the degrees of the input vectors. Shape (n_vectors, vector_dimension).
        vec_degrees_in = degrees_in[self._embedded_indices_in].reshape(-1, self.vector_dimension)

        # All components of each vector must be assigned to the same degree.
        if not torch.all(vec_degrees_in == vec_degrees_in[:, [0]]):
            raise ValueError('The same degree must be assigned to all '
                             'components of each embedded vectors.')

        # Initialize the number of output features.
        n_features_out = len(self._embedded_indices_out) + len(self._nonembedded_indices_out)
        degrees_out = torch.empty(n_features_out, dtype=degrees_in.dtype).to(degrees_in)

        # The degrees of the nonembedded features remain the same.
        degrees_out[self._nonembedded_indices_out] = degrees_in[self._nonembedded_indices_in]

        # The degrees of the embedded vectors are maintained but the change shape.
        vec_degrees_out = vec_degrees_in[:, [0]].expand(-1, self.embedding_dimension)
        degrees_out[self._embedded_indices_out] = vec_degrees_out.flatten()

        return degrees_out
