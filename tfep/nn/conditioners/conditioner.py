#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Base conditioner classes for autoregressive flows.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import abc

import torch


# =============================================================================
# BASE CLASSES
# =============================================================================

class Conditioner(abc.ABC, torch.nn.Module):
    """A conditioner for an autoregressive flow.

    This class documents the API of a conditioner layer compatible with an
    :class:`.AutoregressiveFlow`.

    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the parameters for the transformer.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch_size, n_features)``. The input features.

        Returns
        -------
        parameters : torch.Tensor
            Shape ``(batch_size, n_parameters)``. The parameters for the transformer.

        """
        return super().forward(x)  # Raises NotImplementedError.

    @abc.abstractmethod
    def set_output(self, output: torch.Tensor):
        """Sets the parameters of the conditioner to produce a constant output.

        This is used to force the autoregressive flow to implement the identity
        function on initialization.

        Parameters
        ----------
        output : torch.Tensor
            Shape ``(n_parameters,)``. The desired output of the conditioner.

        """
        pass
