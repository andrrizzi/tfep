#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Circular spline transformer for autoregressive normalizing flows.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import numpy as np
import torch
import torch.autograd


# =============================================================================
# NEURAL SPLINE
# =============================================================================

class NeuralSplineTransformer(torch.nn.Module):
    """Neural spline transformer module for autoregressive normalizing flows.

    This is an implementation of the neural spline transformer proposed
    in [1]. Using the therminology in [1], the spline function is defined
    from K+1 knots (x, y) that give rise to K bins.

    Parameters
    ----------
    x0 : torch.Tensor
        Position of the first of the K+1 knots determining the positions of the
        K bins for the input as a tensor of shape ``(n_features,)``. Inputs that
        are equal or below this (in any dimension) are mapped to itself.
    xf : torch.Tensor
        Position of the last of the K+1 knots determining the positions of the
        K bins for the input as a tensor of shape ``(n_features,)``. Inputs that
        are equal or above this (in any dimension) are mapped to itself.
    n_bins : int
        Total number of bins (i.e., K).
    y0 : torch.Tensor, optional
        Position of the first of the K+1 knots determining the positions of the
        K bins for the output as a tensor of shape ``(n_features,)``. If not
        passed, ``x0`` is taken.
    yf : torch.Tensor, optional
        Position of the last of the K+1 knots determining the positions of the
        K bins for the output as a tensor of shape ``(n_features,)``. If not
        passed, ``x0`` is taken.

    See Also
    --------
    nets.functions.transformer.neural_spline_transformer

    References
    ----------
    [1] Durkan C, Bekasov A, Murray I, Papamakarios G. Neural spline flows.
        arXiv preprint arXiv:1906.04032. 2019 Jun 10.

    """
    def __init__(self, x0, xf, n_bins, y0=None, yf=None):
        super().__init__()

        # Handle mutable default arguments y_0 and y_final.
        if y0 is None:
            y0 = x0.detach()
        if yf is None:
            yf = xf.detach()

        self.x0 = x0
        self.xf = xf
        self.y0 = y0
        self.yf = yf
        self.n_bins = n_bins

    @property
    def n_parameters_per_input(self):
        """Number of parameters needed by the transformer for each input dimension."""
        # n_bins widths, n_bins heights and n_bins-1 slopes.
        return 3*self.n_bins - 1

    def forward(self, x, parameters):
        """Apply the transformation to the input.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor x of shape ``(batch_size, n_features)``.
        parameters : torch.Tensor
            Parameters of the transformation with shape ``(batch_size, 3*n_bins-1, n_features)``
            where ``parameters[:, 0:n_bins, i]`` determine the widths,
            ``parameters[:, n_bins:2*n_bins, i]`` determine the heights, and
            ``parameters[:, 2*n_bins:3*n_bins-1, i]`` determine the slopes of
            the bins.

            As in the original paper, the parameters determine the widths and
            heights go through a ``softmax`` function and those determining the
            slopes through a ``softplus`` function to generate widths and heights
            that are positive and slopes that ensures the transformation will be
            monotonic increasing (and thus invertible).

        Returns
        -------
        y : torch.Tensor
            Output tensor of shape ``(batch_size, n_features)``.

        """
        # Divide the parameters in widths, heights and slopes.
        widths = torch.nn.functional.softmax(parameters[:, :self.n_bins], dim=1) * (self.xf - self.x0)
        heights = torch.nn.functional.softmax(parameters[:, self.n_bins:2*self.n_bins], dim=1) * (self.yf - self.y0)
        slopes = torch.nn.functional.softplus(parameters[:, 2*self.n_bins:])
        return neural_spline_transformer(x, self.x0, self.y0, widths, heights, slopes)

    def inverse(self, y, parameters):
        """Currently not implemented."""
        raise NotImplementedError(
            'Inversion of neural spline transformer has not been implemented yet.')

    def get_identity_parameters(self, n_features):
        """Return the value of the parameters that makes this the identity function.

        Note that if ``self.x0 != self.y0`` or ``self.y0 != self.y1`` it is
        impossible to implement the identity using this transformer, and the
        returned parameters will be those to map linearly the input domain of
        ``x`` to the output of ``y``.

        This can be used to initialize the normalizing flow to perform the identity
        transformation.

        Parameters
        ----------
        n_features : int
            The dimension of the input vector of the transformer.

        Returns
        -------
        parameters : torch.Tensor
            A tensor of shape ``(3*n_bins-1, n_features)`` where ``K`` and ``L`` are
            the number and degree of the polynomials.

        """
        # Strictly speaking, this becomes the identity conditioner only if x0 == y0 and xf == yf.
        id_conditioner = torch.empty(size=(self.n_parameters_per_input, n_features))

        # Both the width and the height of each bin must be constant.
        # Remember that the parameters go through the softmax function.
        id_conditioner[:self.n_bins].fill_(1 / self.n_bins)
        id_conditioner[self.n_bins:2*self.n_bins].fill_(1 / self.n_bins)

        # The slope must be one in each knot. Remember that the parameters
        # go through the softplus function.
        id_conditioner[2*self.n_bins:].fill_(np.log(np.e - 1))

        return id_conditioner


# =============================================================================
# FUNCTIONAL API
# =============================================================================

def neural_spline_transformer(x, x0, y0, widths, heights, slopes):
    r"""Implement the neural spline transformer.

    This is an implementation of the neural spline transformer proposed
    in [1]. Using the therminology in [1], the spline function is defined
    from K+1 knots (x, y) that give rise to K bins.

    The difference with :class:`~tfep.nn.transformers.spline.NeuralSplineTransformer`
    is that it takes as parameters directly widths, heights, and slopes of the
    K+1 knots used for interpolation rather than parameters that are then
    passed to ``softmax`` and ``softplus`` functions.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor x of shape ``(batch_size, n_features)``. Currently, this
        must hold: ``x0[i] <= x[i] <= x0[i] + cumsum(widths)`` for all ``i``.
    x0 : torch.Tensor
        Position of the first of the K+1 knots determining the positions of the
        K bins for the input as a tensor of shape ``(n_features,)``. Inputs that
        are equal or below this (in any dimension) are mapped to itself.
    y0 : torch.Tensor
        Position of the first of the K+1 knots determining the positions of the
        K bins for the output as a tensor of shape ``(n_features,)``.
    widths : torch.Tensor
        ``widths[b, k, i]`` is the width of the k-th bin between the k-th and (k+1)-th
         knot for the i-th feature and b-th batch. The tensor has shape
         ``(batch_size, K, n_features)``.
    heights : torch.Tensor
        ``heights[b, k, i]`` is the height of the k-th bin between the k-th and (k+1)-th
         knot for the i-th feature and b-th batch. The tensor has shape
         ``(batch_size, K, n_features)``.
    slopes : torch.Tensor
        ``slopes[b, k, i]`` is the slope at the (k+1)-th knot (the slope of the
        first and last knots are always 1. The tensor has shape
        ``(batch_size, K-1, n_features)``.

    References
    ----------
    [1] Durkan C, Bekasov A, Murray I, Papamakarios G. Neural spline flows.
        arXiv preprint arXiv:1906.04032. 2019 Jun 10.

    """
    dtype = x0.dtype
    batch_size, n_bins, n_features = widths.shape
    n_knots = n_bins + 1

    # knots_x has shape (n_features, K+1).
    knots_x = torch.empty(batch_size, n_knots, n_features, dtype=dtype)
    knots_x[:, 0] = x0
    knots_x[:, 1:] = x0 + torch.cumsum(widths, dim=1)
    knots_y = torch.empty(batch_size, n_knots, n_features, dtype=dtype)
    knots_y[:, 0] = y0
    knots_y[:, 1:] = y0 + torch.cumsum(heights, dim=1)

    # The 0-th and last knots have always slope 1 to avoid discontinuities.
    # After this, slopes has shape (batch_size, n_features, K+1).
    ones = torch.ones(batch_size, 1, n_features, dtype=dtype)
    slopes = torch.cat((ones, slopes, ones), dim=1)

    # For an idea about how the indexing is working in this function, see
    # https://stackoverflow.com/questions/55628014/indexing-a-3d-tensor-using-a-2d-tensor.
    batch_indices = torch.arange(batch_size).unsqueeze(-1)  # Shape: (batch_size, 1).
    feat_indices = torch.arange(n_features).repeat(batch_size, 1)  # Shape: (batch_size, n_features).

    # bin_indices[i][j] is the index of the bin assigned to x[i][j].
    bin_indices = torch.sum((x.unsqueeze(1) > knots_x), dim=1) - 1

    # All the following arrays have shape (batch_size, n_features).
    # widths_b_f[i][j] is the width of the bin assigned to x[i][j].
    widths_b_f = widths[batch_indices, bin_indices, feat_indices]
    heights_b_f = heights[batch_indices, bin_indices, feat_indices]

    # lower_knot_x_b_f[i][j] is the lower bound of the bin assigned to x[i][j].
    lower_knot_x_b_f = knots_x[batch_indices, bin_indices, feat_indices]
    lower_knot_y_b_f = knots_y[batch_indices, bin_indices, feat_indices]

    # slopes_k_b_f[i][j] is the slope of the lower-bound knot of the bin assigned to x[i][j].
    # slopes_k1_b_f[i][j] is the slope of the upper-bound knot of the bin assigned to x[i][j].
    slopes_k_b_f = slopes[batch_indices, bin_indices, feat_indices]
    slopes_k1_b_f = slopes[batch_indices, bin_indices+1, feat_indices]

    # This is s_k = (y^k+1 - y^k)/(x^k+1 - x^k) and epsilon in the
    # paper, both with shape (batch_size, n_features).
    s_b_f = heights_b_f / widths_b_f
    epsilon_b_f = (x - lower_knot_x_b_f) / widths_b_f

    # epsilon * (1 - epsilon)
    epsilon_1mepsilon_b_f = epsilon_b_f * (1 - epsilon_b_f)
    epsilon2_b_f = epsilon_b_f**2

    # Compute the output.
    numerator = heights_b_f * (s_b_f * epsilon2_b_f + slopes_k_b_f * epsilon_1mepsilon_b_f)
    denominator = s_b_f + (slopes_k1_b_f + slopes_k_b_f - 2*s_b_f) * epsilon_1mepsilon_b_f
    y = lower_knot_y_b_f + numerator/denominator

    # Compute the derivative
    numerator = s_b_f**2 * (slopes_k1_b_f*epsilon2_b_f + 2*s_b_f*epsilon_1mepsilon_b_f + slopes_k_b_f*(1 - epsilon_b_f)**2)
    denominator = (s_b_f + (slopes_k1_b_f + slopes_k_b_f - 2 * s_b_f) * epsilon_1mepsilon_b_f)**2
    dy_dx = numerator / denominator

    # Compute the log det J.
    log_det_J = torch.sum(torch.log(dy_dx), dim=1)

    return y, log_det_J
