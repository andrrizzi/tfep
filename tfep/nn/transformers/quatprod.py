#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Quaternion product transformation for autoregressive normalizing flows.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import torch

from tfep.nn.transformers.transformer import MAFTransformer


# =============================================================================
# QUATERNION PRODUCT TRANSFORMER
# =============================================================================

class QuaternionProductTransformer(MAFTransformer):
    r"""Quaternion product transformer.

    This is a volume-preserving transformation that can be applied to
    quaternions. For each (normalized) quaternion in the input, the conditioner
    must provide a 4-dimensional vector (possibly unnormalized). As quaternions
    typically model the orientation of a molecule, the transformation is
    equivalent to applying a separate rigid rotation to each molecule and thus
    has a unit Jacobian.

    """

    def forward(self, x: torch.Tensor, parameters: torch.Tensor) -> tuple[torch.Tensor]:
        """Apply the transformation.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch_size, n_quaternions*4)``. The quaternions elements
            are contiguous (i.e., the first and second input quaternions are
            ``x[:4]`` and ``x[4:8]``.
        parameters : torch.Tensor
            Shape ``(batch_size, n_quaternions*4)``. The parameters interpreted
            as (unnormalized) quaternions that will multiply those in ``x``.
            These are normalized in the function.

        Returns
        -------
        y : torch.Tensor
            Shape ``(batch_size, n_quaternions*4)``. The transformed normalized
            quaternions.
        log_det_J : torch.Tensor
            Shape ``(batch_size,)``. The logarithm of the absolute value of the Jacobian
            determinant ``dy / dx`` (i.e., always zero).

        """
        # roma is an optional dependency at the moment
        import roma

        # From (batch, n_quaternions*4) to (batch*n_quaternions, 4).
        batch_size = x.shape[0]
        x = x.reshape(-1, 4)
        parameters = parameters.reshape(-1, 4)

        # Transform.
        y = roma.quat_product(roma.quat_normalize(parameters), x)
        log_det_J = torch.zeros(batch_size).to(x)
        return y.reshape(batch_size, -1), log_det_J

    def inverse(self, y: torch.Tensor, parameters: torch.Tensor) -> tuple[torch.Tensor]:
        """Reverse the transformation.

        Parameters
        ----------
        y : torch.Tensor
            Shape ``(batch_size, n_quaternions*4)``. The quaternions elements
            are contiguous (i.e., the first and second input quaternions are
            ``y[:4]`` and ``y[4:8]``.
        parameters : torch.Tensor
            Shape ``(batch_size, n_quaternions*4)``. The parameters interpreted
            as (unnormalized) quaternions that will multiply those in ``y``.
            These are normalized in the function.

        Returns
        -------
        x : torch.Tensor
            Shape ``(batch_size, n_quaternions*4)``. The transformed normalized
            quaternions.
        log_det_J : torch.Tensor
            Shape ``(batch_size,)``. The logarithm of the absolute value of the Jacobian
            determinant ``dy / dx`` (i.e., always zero).

        """
        # roma is an optional dependency at the moment
        import roma

        # From (batch, n_quaternions*4) to (batch*n_quaternions, 4).
        batch_size = y.shape[0]
        y = y.reshape(-1, 4)
        parameters = parameters.reshape(-1, 4)

        # Transform.
        x = roma.quat_product(roma.quat_conjugation(roma.quat_normalize(parameters)), y)
        log_det_J = torch.zeros(batch_size).to(y)
        return x.reshape(batch_size, -1), log_det_J

    def get_identity_parameters(self, n_features: int) -> torch.Tensor:
        """Return the value of the parameters that makes this the identity function.

        This can be used to initialize the normalizing flow to perform the identity
        transformation.

        Parameters
        ----------
        n_features : int
            The dimension of the input vector passed to the transformer. Must
            be divisible by 4.

        Returns
        -------
        parameters : torch.Tensor
            A tensor of shape ``(n_features,)`` representing the parameter
            vector to perform the identity function with a Moebius transformer.

        """
        # roma is an optional dependency at the moment
        import roma
        return roma.identity_quat(n_features//4).flatten()

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
        return degrees_in.detach().clone()
