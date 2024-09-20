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

from typing import Optional

import numpy as np
import torch
import torch.autograd

from tfep.nn.transformers.transformer import MAFTransformer


# =============================================================================
# NEURAL SPLINE
# =============================================================================

class NeuralSplineTransformer(MAFTransformer):
    r"""Neural spline transformer module for autoregressive normalizing flows.

    This is an implementation of the neural spline transformer proposed
    in [1]. Using the therminology in [1], the spline function is defined
    from K+1 knots (x, y) that give rise to K bins. The domain of the spline
    for feature ``i`` is in the interval
    ``x0[i] <= x[i] <= x0[i] + cumsum(widths)``. All values outside this
    interval are transformed linearly following the slopes of the first and
    last knot.

    This class can also implement circular splines [2] for (optionally a subset
    of) the degrees of freedom that are periodic. The periodic DOFs do not take
    slopes for the last knot (which is set to be equal to that of the first),
    but take instead a shift parameter so that the final transformation is

    :math:`y = \mathrm{spline}((x - x_0 + \phi) % p + x_0)

    where :math:`x_0` correspond to the ``x0`` parameter for that DOF, :math:`p`
    is the period, and :math:`\phi` is the shift.

    See Also
    --------
    nets.functions.transformer.neural_spline_transformer

    References
    ----------
    [1] Durkan C, Bekasov A, Murray I, Papamakarios G. Neural spline flows.
        arXiv preprint arXiv:1906.04032. 2019 Dec 12.
    [2] Rezende DJ, et al. Normalizing flows on tori and spheres. arXiv preprint
        arXiv:2002.02428. 2020 Feb 6.

    """
    def __init__(
            self,
            x0: torch.Tensor,
            xf: torch.Tensor,
            n_bins: int,
            y0: Optional[torch.Tensor] = None,
            yf: Optional[torch.Tensor] = None,
            circular: bool = False
    ):
        """Constructor.

        Parameters
        ----------
        x0 : torch.Tensor
            Shape ``(n_features,)``. Position of the first of the K+1 knots
            determining the positions of the K bins for the input.
        xf : torch.Tensor
            Shape ``(n_features,)``. Position of the last of the K+1 knots
            determining the positions of the K bins for the input.
        n_bins : int
            Total number of bins (i.e., K).
        y0 : torch.Tensor, optional
            Shape ``(n_features,)``. Position of the first of the K+1 knots
            determining the positions of the K bins for the output. If not passed,
            ``x0`` is taken.
        yf : torch.Tensor, optional
            Shape ``(n_features,)``. Position of the last of the K+1 knots
            determining the positions of the K bins for the output. If not
            passed, ``xf`` is taken.
        circular : bool or torch.Tensor, optional
            If ``True``, all degrees of freedom are treated as periodic. If a
            list of integers, only the features at these indices are periodic.
            For the periodic DOFs, ``y0`` and ``yf`` must correspond to ``x0``
            and ``xf``.

        """
        super().__init__()

        # Handle mutable default arguments y_0 and y_final.
        if y0 is None:
            y0 = x0.detach()
        if yf is None:
            yf = xf.detach()

        # Check consistent configuration of circular.
        if (circular is not False) and not (
                    torch.allclose(x0[circular], y0[circular]) and
                    torch.allclose(xf[circular], yf[circular])):
            raise ValueError('x0==y0 and xf==yf must hold for all periodic degrees of freedom.')

        self.register_buffer('x0', x0)
        self.register_buffer('xf', xf)
        self.register_buffer('n_bins', torch.as_tensor(n_bins))
        self.register_buffer('_y0', y0)
        self.register_buffer('_yf', yf)
        self.register_buffer('_circular', torch.as_tensor(circular))

    @property
    def n_parameters_per_input(self):
        """Number of parameters needed by the transformer for each input dimension."""
        # n_bins widths, n_bins heights, and n_bins slopes. The +1 can be
        # either the slope of the last knot for not circular DOFs or the shift
        # for periodic DOFs.
        return 3*self.n_bins + 1

    def forward(self, x: torch.Tensor, parameters: torch.Tensor) -> tuple[torch.Tensor]:
        """Apply the transformation to the input.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch_size, n_features)``. Input features. Periodic
            features will be mapped within their domain before being
            transformed. Non-periodic features outside their domain will be
            transformed linearly.
        parameters : torch.Tensor
            Shape: ``(batch_size, (3*n_bins+1)*n_features)``. The order of the
            parameters should allow reshaping the array to
            ``(batch_size, 3*n_bins+1, n_features)``, where
            ``parameters[b, :n_bins, i]``, ``parameters[b, n_bins:2*n_bins, i]``,
            and ``parameters[b, 2*n_bins:3*n_bins, i]`` are the widths, heights,
            and slopes for feature ``x[b, i]``. ``parameters[b, 3*n_bins, i]``
            instead depends on whether the ``x[b, i]`` is periodic. If not
            periodic, they are interpreted the slopes of the last knot. Otherwise,
            the slope of the last knot is set equal to the first and the
            parameters are used as shifts.

            As in the original paper, the passed widths and heights go through
            a ``softmax`` function and the slopes through a ``softplus`` function
            to generate widths/heights that are positive and slopes that make a
            monotonic increasing transformation.

        Returns
        -------
        y : torch.Tensor
            Shape ``(batch_size, n_features)``. Output.
        log_det_J : torch.Tensor
            Shape ``(batch_size,)``. The logarithm of the absolute value of the
            Jacobian determinant ``dy / dx``.

        """
        # Divide the parameters in widths, heights and slopes (and shifts).
        # shift has shape (batch, n_features).
        widths, heights, slopes, shifts = self._get_parameters(parameters)

        # First we shift the periodic DOFs so that we can learn the fixed point
        # of the neural spline transformation. Shifting is volume preserving
        # (i.e., log_det_J = 0).
        if shifts is not None:
            x = (x - self.x0 + shifts) % (self.xf - self.x0) + self.x0

        # Run rational quadratic spline.
        return neural_spline_transformer(x, self.x0, self._y0, widths, heights, slopes)

    def inverse(self, y: torch.Tensor, parameters: torch.Tensor) -> tuple[torch.Tensor]:
        """Inverse function.

        See ``forward`` for the parameters description.

        """
        # Divide the parameters in widths, heights and slopes (and shifts).
        # shift has shape (batch, n_features).
        widths, heights, slopes, shifts = self._get_parameters(parameters)

        # Invert rational quadratic spline.
        x, log_det_J = neural_spline_transformer_inverse(y, self.x0, self._y0, widths, heights, slopes)

        # Shifts the periodic DOFs. Shifting is volume preserving.
        if shifts is not None:
            x = (x - self.x0 - shifts) % (self.xf - self.x0) + self.x0

        return x, log_det_J

    def get_identity_parameters(self, n_features: int) -> torch.Tensor:
        """Return the value of the parameters that makes this the identity function.

        Note that if ``x0 != y0`` or ``y0 != y1`` it is impossible to implement
        the identity using this transformer, and the returned parameters will
        be those to map linearly the input domain of ``x`` to the output of
        ``y``.

        This can be used to initialize the normalizing flow to perform the
        identity transformation.

        Parameters
        ----------
        n_features : int
            The dimension of the input vector of the transformer.

        Returns
        -------
        parameters : torch.Tensor
            Shape ``((3*n_bins+1)*n_features,)``. The parameters for the
            identity function.

        """
        if not (torch.allclose(self.x0, self._y0) and torch.allclose(self.xf, self._yf)):
            raise ValueError('The identity neural spline transformer can be '
                             'implemented only if x0=y0 and xf=yf.')

        # Both the width and the height of each bin must be constant.
        # Remember that the parameters go through the softmax function.
        id_conditioner = torch.empty(size=(self.n_parameters_per_input, n_features)).to(self.x0)
        id_conditioner[:self.n_bins].fill_(1 / self.n_bins)
        id_conditioner[self.n_bins:2*self.n_bins].fill_(1 / self.n_bins)

        # The slope must be one in each knot. Remember that the parameters
        # go through the softplus function.
        id_conditioner[2*self.n_bins:].fill_(np.log(np.e - 1))

        # Set the shift to 0.0. If self._circular is False, nothing is set.
        id_conditioner[3*self.n_bins, self._circular] = 0

        return id_conditioner.reshape(-1)

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
        return degrees_in.tile((self.n_parameters_per_input,))

    def _get_parameters(self, parameters):
        # From (batch_size, 3*n_bins+1*n_features) to (batch_size, 3*n_bins+1, n_features).
        batch_size = parameters.shape[0]
        parameters = parameters.reshape(batch_size, self.n_parameters_per_input, -1)

        # Handle slopes and shifts for periodic DOFs.
        slopes = parameters[:, 2*self.n_bins:]
        if (len(self._circular.shape) == 0) and (self._circular == False):
            shifts = None
        else:  # self._circular is either True or an array of indices.
            # Do not modify the original parameters.
            slopes = slopes.clone()
            batch_size, _, n_features = slopes.shape
            shifts = torch.zeros((batch_size, n_features)).to(slopes)
            shifts[:, self._circular] = slopes[:, self.n_bins, self._circular]
            slopes[:, self.n_bins, self._circular] = slopes[:, 0, self._circular]

        # Normalize widths/heights to boundaries and slopes positive.
        widths = torch.nn.functional.softmax(parameters[:, :self.n_bins], dim=1) * (self.xf - self.x0)
        heights = torch.nn.functional.softmax(parameters[:, self.n_bins:2*self.n_bins], dim=1) * (self._yf - self._y0)
        slopes = torch.nn.functional.softplus(slopes)
        return widths, heights, slopes, shifts


# =============================================================================
# FUNCTIONAL API
# =============================================================================

def neural_spline_transformer(x, x0, y0, widths, heights, slopes):
    r"""Implement the neural spline transformer.

    This is an implementation of the neural spline transformer proposed in [1].
    Using the therminology in [1], the spline function is defined from K+1
    knots (x, y) that delimit K bins. The domain of the spline for feature ``i``
    is in the interval ``x0[i] <= x[i] <= x0[i] + cumsum(widths)``. All values
    outside this interval are transformed linearly following the slopes of the
    first and last knot. Note that periodic features must be passed within
    their domain or they will be transformed linearly.

    The difference with :class:`~tfep.nn.transformers.spline.NeuralSplineTransformer`
    is that it takes as parameters directly widths, heights, and slopes of the
    K+1 knots used for interpolation rather than parameters that are then
    passed to ``softmax`` and ``softplus`` functions.

    Parameters
    ----------
    x : torch.Tensor
        Shape ``(batch_size, n_features)``. Input features. Periodic features
        must be within their domain. Non-periodic features outside their domain
        will be transformed linearly.
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
        arXiv preprint arXiv:1906.04032. 2019 Dec 12.

    """
    # Assign inputs to bins. _b_f suffix represents the shape (batch_size, n_features).
    (widths_b_f, heights_b_f,
     lower_knot_x_b_f, lower_knot_y_b_f,
     slopes_k_b_f, slopes_k1_b_f,
     s_b_f) = _assign_bins(x, x0, y0, widths, heights, slopes, inverse=False)

    # epsilon_b_f[i][j] is the epsilon value for x[i][j].
    epsilon_b_f = (x - lower_knot_x_b_f) / widths_b_f

    # epsilon * (1 - epsilon)
    epsilon_1mepsilon_b_f = epsilon_b_f * (1 - epsilon_b_f)
    epsilon2_b_f = epsilon_b_f**2

    # Compute the output.
    numerator = heights_b_f * (s_b_f * epsilon2_b_f + slopes_k_b_f * epsilon_1mepsilon_b_f)
    denominator = s_b_f + (slopes_k1_b_f + slopes_k_b_f - 2*s_b_f) * epsilon_1mepsilon_b_f
    y = lower_knot_y_b_f + numerator/denominator

    # Compute the log_det_J.
    log_det_J = _compute_log_det_J(
        x, widths_b_f, lower_knot_x_b_f, slopes_k_b_f, slopes_k1_b_f, s_b_f,
        epsilon_b_f, epsilon_1mepsilon_b_f, epsilon2_b_f, inverse=False,
    )
    return y, log_det_J


def neural_spline_transformer_inverse(y, x0, y0, widths, heights, slopes):
    r"""Implement the inverse of the neural spline transformer.

    For more details, see the documentation of ``neural_spline_transformer``.

    References
    ----------
    [1] Durkan C, Bekasov A, Murray I, Papamakarios G. Neural spline flows.
        arXiv preprint arXiv:1906.04032. 2019 Dec 12.

    """
    # Assign inputs to bins. _b_f suffix represents the shape (batch_size, n_features).
    (widths_b_f, heights_b_f,
     lower_knot_x_b_f, lower_knot_y_b_f,
     slopes_k_b_f, slopes_k1_b_f,
     s_b_f) = _assign_bins(y, x0, y0, widths, heights, slopes, inverse=True)

    # Common terms for inversion coefficients. All variable refers to the paper [1].
    # y - y^k
    y_myk = y - lower_knot_y_b_f
    # delta^{k+1} + delta^k - 2s^k
    dk1_dk_m2s = slopes_k1_b_f + slopes_k_b_f - 2*s_b_f

    # Inversion coefficients: a, b, c.
    a = heights_b_f*(s_b_f - slopes_k_b_f) + y_myk*dk1_dk_m2s
    b = heights_b_f*slopes_k_b_f - y_myk*dk1_dk_m2s
    c = -s_b_f * y_myk

    # Compute inverse.
    epsilon_b_f = 2 * c.div(-b - torch.sqrt(b**2 - 4*a*c))
    x = epsilon_b_f * widths_b_f + lower_knot_x_b_f

    # Compute the log_det_J.
    epsilon_1mepsilon_b_f = epsilon_b_f * (1 - epsilon_b_f)
    epsilon2_b_f = epsilon_b_f**2
    log_det_J = _compute_log_det_J(
        x, widths_b_f, lower_knot_x_b_f, slopes_k_b_f, slopes_k1_b_f, s_b_f,
        epsilon_b_f, epsilon_1mepsilon_b_f, epsilon2_b_f, inverse=True,
    )
    return x, log_det_J


def _compute_log_det_J(
        x, widths_b_f, lower_knot_x_b_f, slopes_k_b_f, slopes_k1_b_f, s_b_f,
        epsilon_b_f, epsilon_1mepsilon_b_f, epsilon2_b_f, inverse,
):
    """Compute the log det J of the transformation.

    The ``x`` is always the x coordinate. Even for the inverse function.
    If epsilon_1mepsilon_b_f and epsilon2_b_f are not given they are computed here.
    """
    # Compute the derivative
    numerator = s_b_f**2 * (slopes_k1_b_f*epsilon2_b_f + 2*s_b_f*epsilon_1mepsilon_b_f + slopes_k_b_f*(1 - epsilon_b_f)**2)
    denominator = (s_b_f + (slopes_k1_b_f + slopes_k_b_f - 2 * s_b_f) * epsilon_1mepsilon_b_f)**2
    dy_dx = numerator / denominator

    # Compute the log det J of the forward transformation.
    log_det_J = torch.sum(torch.log(dy_dx), dim=1)
    if inverse:
        return -log_det_J
    return log_det_J


def _assign_bins(x, x0, y0, widths, heights, slopes, inverse):
    """Assign the input to the bins and return the values of knots, widths, heights, and slopes."""
    batch_size, n_bins, n_features = widths.shape

    # Compute cumulative width and height.
    cum_width = torch.cumsum(widths, dim=1)
    cum_height = torch.cumsum(heights, dim=1)

    # The knots at the start and end of the array are there to enable the
    # definition (as a linear function) outside the domain of the spline
    # (x0, xf).
    n_knots = n_bins + 3

    # knots_x has shape (batch, K+1+2, n_features).
    knots_x = torch.empty(batch_size, n_knots, n_features).to(x0)
    knots_x[:, 1] = x0
    knots_x[:, 2:-1] = x0 + cum_width
    knots_y = torch.empty(batch_size, n_knots, n_features).to(x0)
    knots_y[:, 1] = y0
    knots_y[:, 2:-1] = y0 + cum_height

    # The extreme knots defining the linear layers must be large. We set them
    # to 2 orders of magnitude larger than the spline domain. cum_width[-1] is
    # the total width of the domain.
    dx = cum_width[:, -1] * 100.
    knots_x[:, 0] = x0 - dx
    knots_x[:, -1] = knots_x[:, -2] + dx

    # The extreme knots in y must be placed along the linear function.
    dy0 = slopes[:, 0] * dx
    knots_y[:, 0] = y0 - dy0
    dyf = slopes[:, -1] * dx
    knots_y[:, -1] = knots_y[:, -2] + dyf

    # Set the slopes of the extreme knots to make the function be linear outside
    # the spline domain.
    slopes = torch.cat([slopes[:, 0:1], slopes, slopes[:, -1:]], dim=1)
    dx = dx.unsqueeze(1)
    widths = torch.cat([dx, widths, dx], dim=1)
    heights = torch.cat([dy0.unsqueeze(1), heights, dyf.unsqueeze(1)], dim=1)

    # For an idea about how the indexing is working in this function, see
    # https://stackoverflow.com/questions/55628014/indexing-a-3d-tensor-using-a-2d-tensor.
    batch_indices = torch.arange(batch_size).unsqueeze(-1)  # Shape: (batch_size, 1).
    feat_indices = torch.arange(n_features).repeat(batch_size, 1)  # Shape: (batch_size, n_features).

    # bin_indices[i][j] is the index of the bin assigned to x[i][j].
    if inverse:
        bin_indices = torch.sum((x.unsqueeze(1) > knots_y), dim=1) - 1
    else:
        bin_indices = torch.sum((x.unsqueeze(1) > knots_x), dim=1) - 1

    # _b_f suffix represents the shape (batch_size, n_features).
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

    return (
        widths_b_f, heights_b_f,
        lower_knot_x_b_f, lower_knot_y_b_f,
        slopes_k_b_f, slopes_k1_b_f,
        s_b_f,
    )
