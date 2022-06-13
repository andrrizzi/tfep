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

    This class can also implement circular splines [2] by setting ``circular``
    to ``True``.

    Parameters
    ----------
    x0 : torch.Tensor
        Shape ``(n_features,)``. Position of the first of the K+1 knots determining
        the positions of the K bins for the input.
    xf : torch.Tensor
        Shape ``(n_features,)``. Position of the last of the K+1 knots determining
        the positions of the K bins for the input.
    n_bins : int
        Total number of bins (i.e., K).
    y0 : torch.Tensor, optional
        Shape ``(n_features,)``. Position of the first of the K+1 knots determining
        the positions of the K bins for the output. If not passed, ``x0`` is taken.
    yf : torch.Tensor, optional
        Shape ``(n_features,)``. Position of the last of the K+1 knots determining
        the positions of the K bins for the output. If not passed, ``xf`` is taken.
    circular : bool, optional
        If ``True``, the slope of the last know is set equal to the last node and
        ``y0`` and ``yf`` are set to ``x0`` and ``xf`` respectively. This effectively
        implements circular splines [2].

    See Also
    --------
    nets.functions.transformer.neural_spline_transformer

    References
    ----------
    [1] Durkan C, Bekasov A, Murray I, Papamakarios G. Neural spline flows.
        arXiv preprint arXiv:1906.04032. 2019 Jun 10.
    [2] Rezende DJ, et al. Normalizing flows on tori and spheres. arXiv preprint
        arXiv:2002.02428. 2020 Feb 6.

    """
    def __init__(self, x0, xf, n_bins, y0=None, yf=None, circular=False):
        super().__init__()

        # Check consistent configuration.
        if circular and not (y0 is None and yf is None):
            raise ValueError('With circular=True, both y0 and yf must be None.')

        # Handle mutable default arguments y_0 and y_final.
        if y0 is None:
            y0 = x0.detach()
        if yf is None:
            yf = xf.detach()

        self.x0 = x0
        self.xf = xf
        self.n_bins = n_bins
        self._y0 = y0
        self._yf = yf
        self._circular = circular

    @property
    def circular(self):
        return self._circular

    @property
    def n_parameters_per_input(self):
        """Number of parameters needed by the transformer for each input dimension."""
        # n_bins widths, n_bins heights. The number of slopes depends on whether
        # the spline is circular (n_bins) or not (n_bins+1).
        if self.circular:
            return 3*self.n_bins
        return 3*self.n_bins + 1

    def forward(self, x, parameters):
        """Apply the transformation to the input.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch_size, n_features)``. Input features.
        parameters : torch.Tensor
            Shape: ``(batch_size, K, n_features)`` where ``K`` equals ``3*n_bins``
            if circular or ``3*n_bins+1`` if not. Parameters of the transformation,
            where ``parameters[b, 0:n_bins, i]`` determine the widths,
            ``parameters[b, n_bins:2*n_bins, i]`` determine the heights,
            and ``parameters[b, 2*n_bins:, i]`` determine the slopes of the bins
            for feature ``x[b, i]``.

            As in the original paper, the passed widths and heights go through
            a ``softmax`` function and the slopes through a ``softplus`` function
            to generate widths/heights that are positive and slopes that make a
            monotonic increasing transformation.

        Returns
        -------
        y : torch.Tensor
            Shape ``(batch_size, n_features)``. Output.

        """
        assert parameters.shape[1] == self.n_parameters_per_input

        # Divide the parameters in widths, heights and slopes.
        widths, heights, slopes = self._get_parameters(parameters)
        return neural_spline_transformer(x, self.x0, self._y0, widths, heights, slopes)

    def inverse(self, y, parameters):
        """Currently not implemented."""
        raise NotImplementedError(
            'Inversion of neural spline transformer has not been implemented yet.')

    def get_identity_parameters(self, n_features):
        """Return the value of the parameters that makes this the identity function.

        Note that if ``x0 != y0`` or ``y0 != y1`` it is
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
            A tensor of shape ``(K, n_features)``  where ``K`` equals ``3*n_bins``
            if circular or ``3*n_bins+1`` if not.

        """
        if not (torch.allclose(self.x0, self._y0) and torch.allclose(self.xf, self._yf)):
            raise ValueError('The identity neural spline transformer can be '
                             'implemented only if x0=y0 and xf=yf.')

        # Both the width and the height of each bin must be constant.
        # Remember that the parameters go through the softmax function.
        id_conditioner = torch.empty(size=(self.n_parameters_per_input, n_features))
        id_conditioner[:self.n_bins].fill_(1 / self.n_bins)
        id_conditioner[self.n_bins:2*self.n_bins].fill_(1 / self.n_bins)

        # The slope must be one in each knot. Remember that the parameters
        # go through the softplus function.
        id_conditioner[2*self.n_bins:].fill_(np.log(np.e - 1))

        return id_conditioner

    def _get_parameters(self, parameters):
        widths = torch.nn.functional.softmax(parameters[:, :self.n_bins], dim=1) * (self.xf - self.x0)
        heights = torch.nn.functional.softmax(parameters[:, self.n_bins:2*self.n_bins], dim=1) * (self._yf - self._y0)
        slopes = torch.nn.functional.softplus(parameters[:, 2*self.n_bins:])

        # If this is a circular spline flow, we set the slope of the last knot
        # to the first. Otherwise, this has already shape (batch_size, K+1, n_features).
        if slopes.shape[1] < self.n_bins+1:
            slopes = torch.cat((slopes, slopes[:, 0:1]), dim=1)

        return widths, heights, slopes


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
        Shape ``(batch_size, n_features)``. Input features. Currently, this
        must hold: ``x0[i] <= x[i] <= x0[i] + cumsum(widths)`` for all ``i``.
    x0 : torch.Tensor
        Shape ``(n_features,)``. Position of the first of the K+1 knots determining
        the positions of the K bins for the input.
    y0 : torch.Tensor
        Shape ``(n_features,)``. Position of the first of the K+1 knots determining
        the positions of the K bins for the output.
    widths : torch.Tensor
        Shape ``(batch_size, K, n_features)``. ``widths[b, k, i]`` is the width
        of the k-th bin between the k-th and (k+1)-th knot for the i-th feature
        and b-th batch.
    heights : torch.Tensor
        Shape ``(batch_size, K, n_features)``. ``heights[b, k, i]`` is the height
        of the k-th bin between the k-th and (k+1)-th knot for the i-th feature
        and b-th batch.
    slopes : torch.Tensor
        Shape ``(batch_size, K+1, n_features)``. ``slopes[b, k, i]`` is the slope
        at the k-th knot.

    References
    ----------
    [1] Durkan C, Bekasov A, Murray I, Papamakarios G. Neural spline flows.
        arXiv preprint arXiv:1906.04032. 2019 Jun 10.

    """
    dtype = x0.dtype
    batch_size, n_bins, n_features = widths.shape
    n_knots = n_bins + 1

    # knots_x has shape (batch, K+1, n_features).
    knots_x = torch.empty(batch_size, n_knots, n_features, dtype=dtype)
    knots_x[:, 0] = x0
    knots_x[:, 1:] = x0 + torch.cumsum(widths, dim=1)
    knots_y = torch.empty(batch_size, n_knots, n_features, dtype=dtype)
    knots_y[:, 0] = y0
    knots_y[:, 1:] = y0 + torch.cumsum(heights, dim=1)

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
