#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Normalizing flow transforming to and from PCA-whitened space.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import torch

import tfep.utils.math


# =============================================================================
# PCA WHITENED FLOW
# =============================================================================

class PCAWhitenedFlow(torch.nn.Module):
    """Normalizing flow transforming to and from PCA-whitened space.

    The layer wraps a normalizing flows, passes to it PCA-whitened coordinates,
    and finally (optionally) blacken the output to return to the original space.

    The PCA-whitening matrix is estimated from data that is passed on initialization.

    Parameters
    ----------
    flow : torch.nn.Module
        The wrapped normalizing flow.
    x : torch.Tensor
        A tensor of shape (n_samples, n_features) which is used to estimate mean
        and covariance matrix of the coordinates and compute the PCA matrix.
    blacken : bool, optional
        If ``False``, the output coordinates are not blackened by inverting the
        PCA whitening transformation.

    """

    def __init__(self, flow, x, blacken=True):
        super().__init__()
        self.flow = flow
        self.blacken = True

        # We don't need to keep track of the graph for backpropagation here.
        x = x.detach()

        # Compute mean and covariance.
        cov, mean = tfep.utils.math.cov(x, return_mean=True)

        # Compute eigenvalues/vectors and singular values.
        eigvalues, eigvectors = torch.linalg.eigh(cov)
        if torch.any(eigvalues < 0.0):
            raise ValueError(
                'Cannot determine the PCA whitening matrix since some of the '
                'eigenvalues of the covariance matrix estimate are negative. '
                'Likely, this is due to an insufficient number of samples.')
        singular_values = torch.sqrt(eigvalues)

        # Whitening matrix.
        whitening_matrix = torch.matmul(eigvectors, torch.diag(1. / singular_values))
        blackening_matrix = torch.matmul(torch.diag(singular_values), eigvectors.t())

        # The jacobian determinant of the whitening transformation
        # is just the product inverse singular values.
        whitening_log_det_J = -torch.sum(torch.log(singular_values))

        # Register various tensors as buffers so that PyTorch will automatically
        # save them and restore them with state_dict.
        self.register_buffer('mean', mean)
        self.register_buffer('whitening_matrix', whitening_matrix)
        self.register_buffer('blackening_matrix', blackening_matrix)
        self.register_buffer('whitening_log_det_J', whitening_log_det_J)

    def n_parameters(self):
        """int: The total number of parameters that can be optimized."""
        return self.flow.n_parameters()

    def forward(self, x):
        return self._pass(x, inverse=False)

    def inverse(self, y):
        return self._pass(y, inverse=True)

    def _whiten(self, x):
        return torch.matmul(x - self.mean, self.whitening_matrix)

    def _blacken(self, x):
        return torch.matmul(x, self.blackening_matrix) + self.mean

    def _pass(self, x, inverse):
        # Check wheter we need to whiten and/or blacken the features.
        whiten = not inverse or self.blacken
        blacken = inverse or self.blacken

        # Whiten before going through the encapsulated flow.
        if whiten:
            x = self._whiten(x)

        # Run the encapsulated flow.
        if inverse:
            y, log_det_J = self.flow.inverse(x)
        else:
            y, log_det_J = self.flow(x)

        # Blacken the feature before returning the output.
        if blacken:
            y = self._blacken(y)

        # If we perform only one between whitening/blackening,
        # the two jacobians don't cancel each other out.
        if not (whiten and blacken):
            if whiten:
                log_det_J = log_det_J + self.log_det_J
            else:
                log_det_J = log_det_J - self.log_det_J

        return y, log_det_J
