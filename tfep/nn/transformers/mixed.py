#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
A transformer applying different transformers to different features.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

from collections.abc import Sequence

import torch

from tfep.nn.transformers.transformer import MAFTransformer
from tfep.utils.misc import ensure_tensor_sequence


# =============================================================================
# MIXED TRANSFORMER
# =============================================================================

class MixedTransformer(MAFTransformer):
    """A transformer applying different transformers to different features."""

    def __init__(
            self,
            transformers : Sequence[MAFTransformer],
            indices : Sequence[Sequence[int]],
    ):
        """Constructor.

        Parameters
        ----------
        transformers : Sequence[MAFTransformer].
            The transformers to mix.
        indices : Sequence[Sequence[int]]
            A list of length ``len(transformers)``. ``indices[i]`` is another
            list containing the indices of the input features for the ``i``-th
            transformer. The sum of all the lengths must equal the number of
            features.

        """
        super().__init__()

        # Input checking.
        if len(transformers) < 2:
            raise ValueError('The number of transformers must be greater than 1.')
        if len(transformers) != len(indices):
            raise ValueError('The number of elements in indices must equal that in transformers.')

        self._transformers = transformers

        # Save the indices into buffers.
        for idx, ind in enumerate(indices):
            self.register_buffer(f'indices{idx}', ensure_tensor_sequence(ind))

        # Cache the starting and ending indices to split the parameters.
        par_lengths = [len(transformer.get_identity_parameters(len(ind)))
                       for transformer, ind in zip(transformers, indices)]
        split_indices = torch.cumsum(torch.tensor(par_lengths[:-1]), dim=0)
        self.register_buffer('_parameters_split_indices', split_indices)

    def forward(self, x: torch.Tensor, parameters: torch.Tensor) -> tuple[torch.Tensor]:
        """Apply the transformation.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch_size, n_features)``. The input features.
        parameters : torch.Tensor
            Shape ``(batch_size, n_parameters)``. The parameters for the
            transformers expected grouped by transformer (i.e., first all
            parameters for the first transformer, then those of the second one
            etc.).

        Returns
        -------
        y : torch.Tensor
            Shape ``(batch_size, n_features)``. The transformed vectors.
        log_det_J : torch.Tensor
            Shape ``(batch_size,)``. The logarithm of the absolute value of the Jacobian
            determinant ``dy / dx``.

        """
        return self._run(x, parameters, inverse=False)

    def inverse(self, y: torch.Tensor, parameters: torch.Tensor) -> tuple[torch.Tensor]:
        """Reverse the transformation.

        Parameters
        ----------
        y : torch.Tensor
            Shape ``(batch_size, n_features)``. The input features.
        parameters : torch.Tensor
            Shape ``(batch_size, n_parameters)``. The parameters for the
            transformers expected grouped by transformer (i.e., first all
            parameters for the first transformer, then those of the second one
            etc.).

        Returns
        -------
        x : torch.Tensor
            Shape ``(batch_size, n_features)``. The transformed vectors.
        log_det_J : torch.Tensor
            Shape ``(batch_size,)``. The logarithm of the absolute value of the Jacobian
            determinant ``dx / dy``.

        """
        return self._run(y, parameters, inverse=True)

    def get_identity_parameters(self, n_features: int) -> torch.Tensor:
        """Return the value of the parameters that makes this the identity function.

        This can be used to initialize the normalizing flow to perform the identity
        transformation.

        Parameters
        ----------
        n_features : int
            The dimension of the input vector passed to the transformer.

        Returns
        -------
        parameters : torch.Tensor
            Shape ``(n_parameters,)``. The parameters for the identity function.

        """
        parameters = [transformer.get_identity_parameters(len(indices))
                      for transformer, indices in zip(self._transformers, self._indices)]
        return torch.cat(parameters, dim=-1)

    def get_degrees_out(self, degrees_in: torch.Tensor) -> torch.Tensor:
        """Returns the degrees associated to the conditioner's output.

        Parameters
        ----------
        degrees_in : torch.Tensor
            Shape ``(n_transformed_features,)``. The autoregressive degrees
            associated to the features provided as input to the transformer.

        Returns
        -------
        degrees_out : torch.Tensor
            Shape ``(n_parameters,)``. The autoregressive degrees associated
            to each output of the conditioner that will be fed to the
            transformer as parameters.

        """
        degrees_out = [transformer.get_degrees_out(degrees_in[indices])
                       for transformer, indices in zip(self._transformers, self._indices)]
        return torch.cat(degrees_out, dim=-1)

    @property
    def _indices(self):
        """Construct a list of buffers."""
        indices = []
        for idx, transformer in enumerate(self._transformers):
            indices.append(getattr(self, f'indices{idx}'))
        return indices

    def _run(self, x, parameters, inverse):
        """Execute the transformation."""
        # Avoid in place modification for the result.
        y = torch.empty_like(x)
        cumulative_log_det_J = 0.0

        # Split the parameters by transformer.
        parameters = torch.tensor_split(parameters, self._parameters_split_indices, dim=1)

        # Run transformers.
        for idx, (transformer, par) in enumerate(zip(self._transformers, parameters)):
            indices = getattr(self, f'indices{idx}')
            if inverse:
                y[:, indices], log_det_J = transformer.inverse(x[:, indices], par)
            else:
                y[:, indices], log_det_J = transformer(x[:, indices], par)
            cumulative_log_det_J = cumulative_log_det_J + log_det_J

        return y, cumulative_log_det_J
