#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Encoders to build input features.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import torch


# =============================================================================
# LAYERS
# =============================================================================

class GaussianRadialBasisExpansion(torch.nn.Module):
    """
    Expands distance into a soft one-hot encoded vector using a Gaussian basis.

    This is a simple Gaussian radial basis expansion similar to that used in
    Schnet [1]. Note that this does not use an enveloping function that smoothly
    let this decay to 0 at a fixed cutoff.

    Parameters
    ----------
    means : torch.Tensor
        A tensor of shape ``(n_gaussians,)`` where ``means[i]`` is the center
        of the ``i``-th Gaussian.
    stds : torch.Tensor
        A tensor of shape ``(n_gaussians,)`` where ``stds[i]`` is the standard
        deviation of the ``i``-th Gaussian.
    trainable_means : bool, optional
        If ``True``, the means are defined as parameters of the neural network
        and optimized during training.
    trainable_stds : bool, optional
        If ``True``, the standard deviations are defined as parameters of the
        neural network and optimized during training.

    References
    ----------
    [1] Schütt KT, Sauceda HE, Kindermans PJ, Tkatchenko A, Müller KR.
        Schnet–a deep learning architecture for molecules and materials.
        The Journal of Chemical Physics. 2018 Jun 28;148(24):241722.

    """
    def __init__(self, means, stds, trainable_means=False, trainable_stds=False):
        super().__init__()
        self._means = means

        # We store stds as inverse variances in log units so that even if we
        # train them we are sure the stds will remain positive.
        self._log_gammas = torch.log(1 / stds**2)

        # If we need to train them, we need to define them as parameters.
        if trainable_means:
            self._means = torch.nn.Parameter(self._means)
        if trainable_stds:
            self._log_gammas = torch.nn.Parameter(self._log_gammas)

    @classmethod
    def from_range(cls, n_gaussians, max_mean, min_mean=0.0, relative_std=3.0,
                   trainable_means=False, trainable_stds=False):
        """Create a basis of equidistant Gaussians in a given range.

        By default, standard deviations are set equal to three times the
        displacement between two consecutive gaussians.

        Parameters
        ----------
        n_gaussians : int
            The number of equidistant Gaussians.
        max_mean : float
            The largest mean of the Gaussian.
        min_mean : float, optional
            The smallest mean of the Gaussian.
        relative_std : float, optional
            The standard deviation of each Gaussian relative to the displacement
            between two means. I.e., the std of the Gaussians will be set to
            ``relative_std * (means[i] - means[i-1])``.
        trainable_means : bool, optional
            If ``True``, the means are defined as parameters of the neural network
            and optimized during training.
        trainable_stds : bool, optional
            If ``True``, the standard deviations are defined as parameters of the
            neural network and optimized during training.

        """
        spacing = (max_mean - min_mean)/(n_gaussians-1)
        means = torch.linspace(min_mean, max_mean, n_gaussians)
        stds = torch.full((len(means),), fill_value=relative_std*spacing)
        return cls(means, stds, trainable_means=trainable_means, trainable_stds=trainable_stds)

    def forward(self, distances):
        """Expand a matrix of distances into a soft one-hot representation.

        Parameters
        ----------
        distances : torch.Tensor
            Distance matrix of shape ``[batch_size, n_atoms, n_atoms]`` or
            ``[batch_size, n_atoms, n_atoms, 1]`` where ``distances[b, i, j]``
            represent the distance between atoms ``i`` and ``j`` for the ``b``-th
            batch.

        Returns
        -------
        encoding : torch.Tensor
            A matrix of shape ``[batch_size, n_atoms, n_atoms, n_gaussians]``
            where ``n_gaussians`` is the number of Gaussian basis function used
            to expand the distance.

        """
        if len(distances.shape) < 4:
            distances = distances.unsqueeze(-1)
        disp = (distances - self._means).pow(2)
        gammas = self._log_gammas.exp()
        encoding = torch.exp(- gammas * disp)
        return encoding
