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
# GAUSSIAN BASIS EXPANSION
# =============================================================================

class GaussianBasisExpansion(torch.nn.Module):
    """
    Expands a float into a soft one-hot encoded vector using a Gaussian basis.

    This is a simple Gaussian basis expansion similar to that used in Schnet [1]
    for the radial expansion. Note that this does not use an enveloping function
    that smoothly let this decay to 0 at a fixed cutoff.

    The means and bandwidths of the Gaussians can be specified as trainable
    parameters.

    Parameters
    ----------
    means : torch.Tensor
        A tensor of shape ``(n_gaussians,)`` where ``means[i]`` is the center
        of the ``i``-th Gaussian. The units must be the same used for the
        input distances in ``forward()``.
    stds : torch.Tensor
        A tensor of shape ``(n_gaussians,)`` where ``stds[i]`` is the standard
        deviation of the ``i``-th Gaussian. The units must be the same used for
        the input distances in ``forward()``.
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
            The largest mean of the Gaussian in the same units used for the input
            distances in ``forward()``.
        min_mean : float, optional
            The smallest mean of the Gaussian in the same units used for the
            input distances in ``forward()``.
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
        means, stds = cls._get_equidistant_means_and_stds(
            n_gaussians, max_mean, min_mean, relative_std)
        return cls(means, stds, trainable_means=trainable_means,
                   trainable_stds=trainable_stds)

    def forward(self, data):
        """Expand float data into a soft one-hot representation.

        Parameters
        ----------
        data : torch.Tensor
            Data tensor with shape ``(batch_size, *)``. Typically, this is a
            distance matrix of shape ``[batch_size, n_atoms, n_atoms]`` or
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
        if data.shape[-1] != 1:
            data = data.unsqueeze(-1)
        disp = (data - self._means).pow(2)
        gammas = self._log_gammas.exp()
        encoding = torch.exp(- gammas * disp)
        return encoding

    @classmethod
    def _get_equidistant_means_and_stds(cls, n_gaussians, max_mean, min_mean, relative_std):
        spacing = (max_mean - min_mean)/(n_gaussians-1)
        means = torch.linspace(min_mean, max_mean, n_gaussians)
        stds = torch.full((len(means),), fill_value=relative_std*spacing)
        return means, stds


# =============================================================================
# GAUSSIAN RADIAL BASIS EXPANSION
# =============================================================================

def behler_parrinello_cosine_switching_function(r_cutoff, r, force_zero_after_cutoff=True):
    """Compute the value of the Behler-Parrinello switching function.

    Parameters
    ----------
    r_cutoff : float
        The cutoff imposed by the switching function. The units of ``r`` and
        ``r_cutoff`` must be the same.
    r : torch.Tensor
        A tensor of shape ``(batch_size, n_atoms, n_atoms)`` where ``r[b, i, j]``
        is the distance between atoms ``i`` and ``j`` for the ``b``-th batch.
    force_zero_after_cutoff : bool, optional
        If ``False``, the function assumes that values after the cutoff are not
        provided and thus no element of the switching function needs to be
        explicitly set to 0.0. This can save a calculation if you have already
        removed distances greater than ``r_cutoff`` from ``r``.

    Returns
    -------
    switching_value : torch.Tensor
        A tensor of shape ``(batch_size, n_atoms, n_atoms)`` where ``switching_value[b, i, j]``
        is the value of the switching function between atoms ``i`` and ``j`` for
        the ``b``-th batch.

    """
    switching_value = 0.5 * torch.cos(torch.pi / r_cutoff * r) + 0.5
    if force_zero_after_cutoff:
        switching_value[r > r_cutoff] = 0.0
    return switching_value


class BehlerParrinelloRadialExpansion(GaussianBasisExpansion):
    """
    Expands distance into a soft one-hot encoded vector using a Gaussian basis with a cosine switching function.

    This is a Gaussian radial basis expansion multiplied by a switching function
    similar to that used in Behler-Parrinello neural networks [1].

    The means and bandwidths of the Gaussians can be specified as trainable
    parameters.

    Parameters
    ----------
    r_cutoff : float
        The cutoff for the switching function in the same units used for the
        input distances in ``forward()``.
    means : torch.Tensor
        A tensor of shape ``(n_gaussians,)`` where ``means[i]`` is the center
        of the ``i``-th Gaussian. The units must be the same used for the
        input distances in ``forward()``.
    stds : torch.Tensor
        A tensor of shape ``(n_gaussians,)`` where ``stds[i]`` is the standard
        deviation of the ``i``-th Gaussian. The units must be the same used for
        the input distances in ``forward()``.
    trainable_means : bool, optional
        If ``True``, the means are defined as parameters of the neural network
        and optimized during training.
    trainable_stds : bool, optional
        If ``True``, the standard deviations are defined as parameters of the
        neural network and optimized during training.
    force_zero_after_cutoff : bool, optional
        If ``False``, the function assumes that values after the cutoff are not
        provided and thus no element of the switching function needs to be
        explicitly set to 0.0. This can save a calculation if you have already
        removed distances greater than ``r_cutoff`` from the input.

    References
    ----------
    [1] Behler J, Parrinello M. Generalized neural-network representation of
        high-dimensional potential-energy surfaces. Physical review letters.
        2007 Apr 2;98(14):146401.

    """
    def __init__(self, r_cutoff, means, stds, trainable_means=False,
                 trainable_stds=False, force_zero_after_cutoff=True):
        super().__init__(means, stds, trainable_means, trainable_stds)
        self.r_cutoff = r_cutoff
        self.force_zero_after_cutoff = force_zero_after_cutoff

    @classmethod
    def from_range(cls, r_cutoff, n_gaussians, max_mean, min_mean=0.0, relative_std=3.0,
                   trainable_means=False, trainable_stds=False, force_zero_after_cutoff=True):
        """Create a basis of equidistant Gaussians in a given range.

        By default, standard deviations are set equal to three times the
        displacement between two consecutive gaussians.

        Parameters
        ----------
        r_cutoff : float
            The cutoff for the switching function in the same units used for the
            input distances in ``forward()``.
        n_gaussians : int
            The number of equidistant Gaussians.
        max_mean : float
            The largest mean of the Gaussian in the same units used for the input
            distances in ``forward()``.
        min_mean : float, optional
            The smallest mean of the Gaussian in the same units used for the
            input distances in ``forward()``.
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
        force_zero_after_cutoff : bool, optional
            If ``False``, the function assumes that values after the cutoff are not
            provided and thus no element of the switching function needs to be
            explicitly set to 0.0. This can save a calculation if you have already
            removed distances greater than ``r_cutoff`` from the input.

        """
        means, stds = cls._get_equidistant_means_and_stds(
            n_gaussians, max_mean, min_mean, relative_std)
        return cls(r_cutoff, means, stds, trainable_means=trainable_means,
                   trainable_stds=trainable_stds)

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
        encoding = super().forward(distances)
        switching = behler_parrinello_cosine_switching_function(
            self.r_cutoff, distances, self.force_zero_after_cutoff)
        return encoding * switching.unsqueeze(-1)
