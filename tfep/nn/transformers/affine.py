#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Affine transformer for autoregressive normalizing flows.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import torch
import torch.autograd
import torch.nn.functional


# =============================================================================
# AFFINE
# =============================================================================

class AffineTransformer(torch.nn.Module):
    """Affine transformer module for autoregressive normalizing flows.

    This is an implementation of the transformation

    :math:`y_i = exp(a_i) * x_i + b_i`

    where :math:`a_i` and :math:`b_i` are the log scale and shift parameters of
    the transformation that are usually generated by a conditioner.

    See Also
    --------
    nets.functions.transformer.affine_transformer

    """
    # Number of parameters needed by the transformer for each input dimension.
    n_parameters_per_input = 2

    def forward(self, x, parameters):
        """Apply the affine transformation to the input.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor x of shape ``(batch_size, n_features)``.
        parameters : torch.Tensor
            Parameters of the transformation with shape ``(batch_size, 2, n_features)``
            where ``parameters[:, 0, i]`` is the shift parameter :math:`b_1`
            and ``parameters[:, 1, i]`` is the log scale :math:`a_1`.

        Returns
        -------
        y : torch.Tensor
            Output tensor of shape ``(batch_size, n_features)``.

        """
        shift, log_scale = self._split_parameters(parameters)
        return affine_transformer(x, shift, log_scale)

    def inverse(self, y, parameters):
        """Reverse the affine transformation.

        Parameters
        ----------
        y : torch.Tensor
            Input tensor x of shape ``(batch_size, n_features)``.
        parameters : torch.Tensor
            Parameters of the transformation with shape ``(batch_size, 2, n_features)``
            where ``parameters[:, 0, i]`` is the shift parameter :math:`b_1`
            and ``parameters[:, 1, i]`` is the log scale :math:`a_1`.

        Returns
        -------
        x : torch.Tensor
            Output tensor of shape ``(batch_size, n_features)``.

        """
        shift, log_scale = self._split_parameters(parameters)
        return affine_transformer_inverse(y, shift, log_scale)

    def get_identity_parameters(self, n_features):
        """Return the value of the parameters that makes this the identity function.

        This can be used to initialize the normalizing flow to perform the identity
        transformation. Both the shift and the log scale must be zero for the affine
        transformation to be the identity.

        Parameters
        ----------
        n_features : int
            The dimension of the input vector of the transformer.

        Returns
        -------
        parameters : torch.Tensor
            A tensor of shape ``(2, n_features)``.

        """
        return torch.zeros(size=(self.n_parameters_per_input, n_features))

    def _split_parameters(self, parameters):
        """Divide shift from log scale."""
        return parameters[:, 0], parameters[:, 1]


# =============================================================================
# FUNCTIONAL API
# =============================================================================

def affine_transformer(x, shift, log_scale):
    r"""Implement an affine transformer for autoregressive normalizing flows.

    This provides a functional API to the ``AffineTransformer`` layer. It
    implements the transformation

    :math:`y_i = exp(a_i) * x_i + b_i`

    where :math:`a_i` and :math:`b_i` are the log scale and shift parameters of
    the transformation that are usually generated by a conditioner.

    The function returns the transformed feature as a ``Tensor`` of shape
    ``(batch_size, n_features)`` and the log absolute determinant of its
    Jacobian as a ``Tensor`` of shape ``(batch_size,)``.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor x of shape ``(batch_size, n_features)``.
    shift : torch.Tensor
        The shift coefficients of shape ``(batch_size, n_features)``
        (i.e. the ``b`` coefficients).
    log_scale : torch.Tensor
        The logarithm of the scale coefficients of shape ``(batch_size, n_features)``
        (i.e. the ``a`` coefficients).

    Returns
    -------
    y : torch.Tensor
        Output tensor of shape ``(batch_size, n_features)``.
    log_det_J : torch.Tensor
        The logarithm of the absolute value of the determinant of the Jacobian
        of the transformation with shape ``(batch_size,)``.

    See Also
    --------
    tfep.nn.transformers.AffineTransformer

    """
    y =  x * torch.exp(log_scale) + shift
    log_det_J = torch.sum(log_scale, dim=1)
    return y, log_det_J


def affine_transformer_inverse(y, shift, log_scale):
    r"""Inverse function of ``affine_transformer``.

    This provides a functional API to the ``AffineTransformer`` layer. It
    implements the inverse of the transformation

    :math:`y_i = exp(a_i) * x_i + b_i`

    where :math:`a_i` and :math:`b_i` are the log scale and shift parameters of
    the transformation that are usually generated by a conditioner.

    The function returns the transformed feature as a ``Tensor`` of shape
    ``(batch_size, n_features)`` and the log absolute determinant of its
    Jacobian as a ``Tensor`` of shape ``(batch_size,)``.

    Parameters
    ----------
    y : torch.Tensor
        Input tensor x of shape ``(batch_size, n_features)``.
    shift : torch.Tensor
        The shift coefficients of shape ``(batch_size, n_features)``
        (i.e. the ``b`` coefficients).
    log_scale : torch.Tensor
        The logarithm of the scale coefficients of shape ``(batch_size, n_features)``
        (i.e. the ``a`` coefficients).

    Returns
    -------
    x : torch.Tensor
        Output tensor of shape ``(batch_size, n_features)``.
    log_det_J : torch.Tensor
        The logarithm of the absolute value of the determinant of the Jacobian
        of the transformation with shape ``(batch_size,)``.

    """
    x = (y - shift) * torch.exp(-log_scale)
    log_det_J = -torch.sum(log_scale, dim=1)
    return x, log_det_J
