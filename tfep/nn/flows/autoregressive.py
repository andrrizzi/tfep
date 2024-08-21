#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Base autoregressive flow layer for PyTorch.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

from collections.abc import Sequence
from typing import Optional

import torch

from tfep.utils.misc import ensure_tensor_sequence


# =============================================================================
# AUTOREGRESSIVE FLOW
# =============================================================================

class AutoregressiveFlow(torch.nn.Module):
    """Autoregressive flow.

    This implements a generic autoregressive flow based on the framework described
    in [1] in which the features are transformed by a transformer parametrized
    by a conditioner layer.

    See Also
    --------
    :class:`.Conditioner` : Documents the API of a conditioner layer.
    :class:`.Transformer` : Documents the API of a transformer.

    References
    ----------
    [1] Papamakarios G, Pavlakou T, Murray I. Masked autoregressive flow for
        density estimation. In Advances in Neural Information Processing
        Systems 2017 (pp. 2338-2347).

    """

    def __init__(
            self,
            n_features_in : int,
            transformer_indices: Sequence[Sequence[int]],
            conditioner: torch.nn.Module,
            transformer: torch.nn.Module,
            conditioner_indices: Optional[Sequence[int]] = None,
            embedding: Optional[torch.nn.Module] = None,
            initialize_identity: bool = True,
    ):
        """Constructor.

        Parameters
        ----------
        n_features_in : int, optional
            Total number of input features.
        transformer_indices : Sequence[Sequence[int]]
            The feature indices (possibly a subset of the input features) passed
            to the transformer grouped by their order in the autoregressive model.
            This information is required to evaluate the inverse which generally
            requires multiple passes (see ref. [1]). Any feature not in this
            sequence is considered fixed and propagated without changes by the
            flow.

            For example, ``[[0, 2], [3], [1, 4]`` specifies an autoregressive
            model in which the features ``0,2`` do not depend on any other
            transformed feature (although it might depend on other fixed features
            entering the conditioner), ``3`` depends on ``0,2``, and ``1,4``
            depend on ``0,2,3``.
        conditioner : :class:`.Conditioner`
            The conditioner layer generating parameters for the ``transformer``.
        transformer : :class:`.Transformer`
            The transformer used to map the input features.
        conditioner_indices : Sequence[int], optional
            The subset of features can be passed to the conditioner. By default,
            all input features are passed. When present, these features are passed
            to the ``embedding`` layer instead of directly to the ``conditioner``.
        embedding : torch.nn.Module, optional
            If present, the conditioner input features are first passed to this
            layer whose output is then fed to the ``conditioner``.
        initialize_identity : bool, optional
            If ``True``, the flow is initialized to perform the identity function.

        """
        super().__init__()

        # Turn sequences to tensors.
        transformer_indices = [ensure_tensor_sequence(x) for x in transformer_indices]

        # There is currently no way in pytorch to load a buffer initialized to
        # None so we store an empty buffer instead.
        if conditioner_indices is None:
            conditioner_indices = torch.tensor([], dtype=int)
        else:
            conditioner_indices = ensure_tensor_sequence(conditioner_indices)

        # Check that all indices are withing 0 and n_features_in.
        for indices in (conditioner_indices, *transformer_indices):
            if (indices is not None) and torch.any((indices < 0) | (n_features_in <= indices)):
                raise ValueError("All indices must be 0 <= i < n_features_in.")

        # Build the masks used to evaluate the inverse.
        n_inverse_iterations = len(transformer_indices)
        inverse_masks = torch.full((n_inverse_iterations, n_features_in), False)
        for idx, indices in enumerate(transformer_indices):
            inverse_masks[idx, indices] = True

        # Now build a 1D tensor of feature indices to pass to the transformer.
        transformer_indices = torch.cat(transformer_indices).sort().values

        # Determine the features that must be simply propagated without mapping.
        fixed_indices = torch.arange(n_features_in)
        fixed_indices = fixed_indices[~torch.isin(fixed_indices, transformer_indices)]

        # If fixed_indices is empty, we don't need to store transformer_indices
        # in memory since all features are passed to the transformer.
        n_transformer_indices = len(transformer_indices)
        if len(fixed_indices) == 0:
            transformer_indices = torch.empty_like(fixed_indices)

        # Store everything.
        self._conditioner = conditioner
        self._transformer = transformer
        self._embedding = embedding
        self.register_buffer('_transformer_indices', transformer_indices)
        self.register_buffer('_inverse_masks', inverse_masks)
        self.register_buffer('_fixed_indices', fixed_indices)
        self.register_buffer('_conditioner_indices', conditioner_indices)

        # Initialize the flow to the identity function.
        if initialize_identity:
            # Determine the parameters that the conditioner needs to output for
            # the flow to be the identity function.
            identity_parameters = self._transformer.get_identity_parameters(n_transformer_indices)
            self._conditioner.set_output(identity_parameters)

    @property
    def has_fixed_indices(self):
        """bool: True if some of the features are not transformed by the flow."""
        return len(self._fixed_indices) > 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Push forward.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch_size, n_features)``. Input tensor.

        Returns
        -------
        y : torch.Tensor
            Shape ``(batch_size, n_features)``. Mapped features.
        log_det_J : torch.Tensor
            Shape ``(batch_size,)``. The log absolute value of the Jacobian of
            the transformation.

        """
        parameters = self.get_transformer_parameters(x)

        # Make sure the conditioning dimensions are not altered.
        if self.has_fixed_indices:
            # Avoid in-place operations.
            y = torch.empty_like(x)

            # Fixed indices are not modify.
            y[:, self._fixed_indices] = x[:, self._fixed_indices]

            # Modify only a subset of the features.
            y[:, self._transformer_indices], log_det_J = self._transformer(
                x[:, self._transformer_indices], parameters)
        else:
            y, log_det_J = self._transformer(x, parameters)

        return y, log_det_J

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Inverse.

        This is in general slower than the forward pass as it may require
        multiple passes (see ref. [1] for the algorithm).

        Parameters
        ----------
        y : torch.Tensor
            Shape ``(batch_size, n_features)``. Input tensor.

        Returns
        -------
        x : torch.Tensor
            Shape ``(batch_size, n_features)``. Mapped features.
        log_det_J : torch.Tensor
            Shape ``(batch_size,)``. The log absolute value of the Jacobian of
            the transformation.

        """
        # Initialize the output to an arbitrary value.
        x = torch.zeros_like(y)

        # Initialize the features that are not transformed by the flow.
        if self.has_fixed_indices:
            x[:, self._fixed_indices] = y[:, self._fixed_indices]

            # Only this subset of features is fed to the transformer.
            y = y[:, self._transformer_indices]

            # self._inverse_masks also has shape (n_iterations, n_features). We
            # need a version of the mask with shape (n_iterations, n_transformer_features).
            inverse_masks_transformer = self._inverse_masks[:, self._transformer_indices]
        else:
            inverse_masks_transformer = self._inverse_masks

        # Compute the inverse.
        for mask, mask_transformer in zip(self._inverse_masks, inverse_masks_transformer):
            # Compute the inversion with the current x. Cloning at each step is
            # necessary to compute gradients on inverse.
            parameters = self.get_transformer_parameters(x.clone())

            # The log_det_J that we compute with the last pass is the total log_det_J.
            # x_temp has shape (batch_size, n_mapped_features).
            x_temp, log_det_J = self._transformer.inverse(y, parameters)

            # No need to update all the features (which can complicate the
            # computational graph) but only those we can update at this point.
            x[:, mask] = x_temp[:, mask_transformer]

        return x, log_det_J

    def get_transformer_parameters(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the parameters for the transformer.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch_size, n_features)``. The input tensor.

        Returns
        -------
        Parameters : torch.Tensor
            Shape ``(batch_size, n_parameters)``. The transformer parameters.

        """
        if self._conditioner_indices is not None:
            x = x[:, self._conditioner_indices]
        if self.embedding is not None:
            x = self.embedding(x)
        return self._conditioner(x)
