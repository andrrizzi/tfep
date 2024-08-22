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

import torch

from tfep.utils.misc import ensure_tensor_sequence


# =============================================================================
# BASE CLASS
# =============================================================================

class MAFEmbedding(torch.nn.Module):
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

class PeriodicEmbedding(torch.nn.Module):
    """Lift periodic DOFs into a periodic representation (cos, sin).

    Parameters
    ----------
    n_features_in : int
        Number of input features.
    periodic_indices : torch.Tensor[int]
        Shape (n_periodic,). The (ordered) indices of the input features that
        are periodic and must be lifted to the (cos, sin) representation.
    limits : torch.Tensor[float]
        A pair ``(lower, upper)`` defining the limits of the periodic variables.
        The period is given by ``upper - lower``.

    """

    def __init__(
            self,
            n_features_in : int,
            periodic_indices : torch.Tensor,
            limits : torch.Tensor,
    ):
        super().__init__()

        # Convert all sequences to Tensors to simplify the code.
        periodic_indices = ensure_tensor_sequence(periodic_indices)
        limits = ensure_tensor_sequence(limits)

        self.register_buffer('limits', limits)

        # Cache a set of periodic/nonperiodic indices BEFORE and AFTER the input has been lifted.
        periodic_indices_set = set(periodic_indices.tolist())
        self.register_buffer('_periodic_indices', periodic_indices)
        self.register_buffer('_nonperiodic_indices', torch.tensor([i for i in range(n_features_in)
                                                                   if i not in periodic_indices_set]))

        periodic_indices_lifted = []  # Shape (n_periodic,).
        nonperiodic_indices_lifted = []  # Shape (n_non_periodic,).

        shift_idx = 0
        for i in range(n_features_in):
            if i in periodic_indices_set:
                periodic_indices_lifted.append(i+shift_idx)
                shift_idx += 1
            else:
                nonperiodic_indices_lifted.append(i + shift_idx)

        # Cache as Tensor.
        self.register_buffer('_periodic_indices_lifted', torch.tensor(periodic_indices_lifted))
        self.register_buffer('_nonperiodic_indices_lifted', torch.tensor(nonperiodic_indices_lifted))

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
        x_periodic = (x[:, self._periodic_indices] - self.limits[0]) * period_scale
        cosx = torch.cos(x_periodic)
        sinx = torch.sin(x_periodic)

        # Fill output.
        n_periodic = len(self._periodic_indices)
        y = torch.empty((batch_size, n_features+n_periodic)).to(x)
        y[:, self._periodic_indices_lifted] = cosx
        y[:, self._periodic_indices_lifted+1] = sinx
        y[:, self._nonperiodic_indices_lifted] = x[:, self._nonperiodic_indices]

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
        repeats[self._periodic_indices] = 2
        return torch.repeat_interleave(degrees_in, repeats)
