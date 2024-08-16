#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Base transformer classes for autoregressive flows.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import abc

import torch


# =============================================================================
# BASE CLASSES
# =============================================================================

class Transformer(torch.nn.Module):
    """A transformer for an autoregressive flow.

    This class documents the API of a transformer layer compatible with an
    :class:`.AutoregressiveFlow`.

    """

    def forward(self, x: torch.Tensor, parameters: torch.Tensor) -> tuple[torch.Tensor]:
        """Apply the transformation.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch_size, n_features)``. The input features.
        parameters : torch.Tensor
            Shape ``(batch_size, n_parameters)``. The parameters for the
            transformation generated by a conditioner.

        Returns
        -------
        y : torch.Tensor
            Shape ``(batch_size, n_features)``. The transformed features.
        log_det_J : torch.Tensor
            Shape ``(batch_size,)``. The log absolute value of the Jacobian
            determinant of the transformation.

        """
        return super().forward(x)  # Raises NotImplementedError.

    @abc.abstractmethod
    def inverse(self, y: torch.Tensor, parameters: torch.Tensor) -> tuple[torch.Tensor]:
        """Reverse the transformation.

        Parameters
        ----------
        y : torch.Tensor
            Shape ``(batch_size, n_features)``. The input features.
        parameters : torch.Tensor
            Shape ``(batch_size, n_parameters)``. The parameters for the
            transformation generated by a conditioner.

        Returns
        -------
        x : torch.Tensor
            Shape ``(batch_size, n_features)``. The transformed features.
        log_det_J : torch.Tensor
            Shape ``(batch_size,)``. The log absolute value of the Jacobian
            determinant of the transformation.

        """
        pass

    @abc.abstractmethod
    def get_parameters_identity(self, n_features: int) -> torch.Tensor:
        """Returns the parameters that would make this transformer the identity function.

        Parameters
        ----------
        n_features : int
            The number of transformed features.

        Returns
        -------
        parameters_identity : torch.Tensor
            Shape ``(n_parameters,)``. The parameters of the transformation that
            makes this transformer the identity function.

        """
        pass