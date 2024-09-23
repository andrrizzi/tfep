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

    This class can also implement circular splines [2] for periodic degrees of
    freedom. The periodic DOFs do not take slopes for the last knot (which is
    set to be equal to that of the first), but take instead a shift parameter
    so that the final transformation is

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
            circular: bool = False,
            identity_boundary_slopes: bool = False,
            min_bin_size: float = 1e-4,
            min_slope: float = 1e-4,
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
        circular : bool, optional
            If ``True``, the features are treated as periodic. In this case,
            ``y0`` and ``yf`` must correspond to ``x0`` and ``xf``.
        identity_boundary_slopes : bool, optional
            If ``True``, the slopes at the boundaries of the domain are forced
            to 1 so that the spline can implement the identity function outside
            its domain. Note that this will be true only if ``x0 == y0`` and
            ``xf == yf``.
        min_bin_size : float, optional
            The minimum possible bin size (i.e., width and height) for
            numerical stability.
        min_slope : float, optional
            The minimum possible slope for numerical stability.

        """
        super().__init__()

        # Handle mutable default arguments y_0 and y_final.
        if y0 is None:
            y0 = x0.detach()
        if yf is None:
            yf = xf.detach()

        # Check consistent configuration of circular.
        if circular and not (
                    torch.allclose(x0[circular], y0[circular]) and
                    torch.allclose(xf[circular], yf[circular])):
            raise ValueError('x0==y0 and xf==yf must hold for all periodic degrees of freedom.')

        # Check values for minimum bin size/slope.
        if min_bin_size <= 0.:
            raise ValueError('The minimum bin size should be positive.')
        if (min_slope <= 0.) or (min_slope >= 1.):
            raise ValueError('The minimum slope should be between 0 and 1.')

        self.register_buffer('x0', x0)
        self.register_buffer('xf', xf)
        self.register_buffer('n_bins', torch.as_tensor(n_bins))
        self.register_buffer('_y0', y0)
        self.register_buffer('_yf', yf)
        self.register_buffer('_circular', torch.as_tensor(circular))
        self.register_buffer('_identity_boundary_slopes', torch.as_tensor(identity_boundary_slopes))
        self.register_buffer('_min_bin_size', torch.as_tensor(min_bin_size))
        self.register_buffer('_min_slope', torch.as_tensor(min_slope))

    @property
    def n_parameters_per_feature(self) -> int:
        """int: Number of parameters needed by the transformer for each feature."""
        if self._identity_boundary_slopes:
            if self._circular:
                return 3*self.n_bins
            else:
                return 3*self.n_bins - 1
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
            Shape: ``(batch_size, n_parameters_per_feat*n_features)``, where
            ``n_parameters_per_feat`` is ``3*n_bins + 1`` for standard splines.
            If ``identity_boundary_slopes=True``, ``n_parameters_per_feat`` is
            instead ``3*n_bins`` for circular splines and ``3*n_bins - 1`` for
            standard ones since the slopes of the boundary knots are fixed.

            The order of the parameters should allow reshaping the array to
            ``(batch_size, n_parameters_per_feat, n_features)``, where
            ``parameters[b, :n_bins, i]`` and ``parameters[b, n_bins:2*n_bins, i]``
            are the widths and heights of the bins for feature ``x[b, i]``.
            ``parameters[b, 2*n_bins:3*n_bins +- 1, i]`` are the slopes (the
            ``+- 1`` is due to the values of ``identity_boundary_slopes`` and
            ``circular``). For circular splines, ``parameters[b, -1, i]`` is
            the shift parameter.

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
            Shape ``(n_parameters,)``. The parameters for the identity function.

        """
        if not (torch.allclose(self.x0, self._y0) and torch.allclose(self.xf, self._yf)):
            raise ValueError('The identity neural spline transformer can be '
                             'implemented only if x0=y0 and xf=yf.')

        # The slopes parameters in _get_parameters are offset so that the final
        # slope will be 1 when zeros are passed. This also sets the shifts to 0
        # for periodic features.
        id_conditioner = torch.zeros(size=(self.n_parameters_per_feature, n_features)).to(self.x0)

        # Both the width and the height of each bin must be constant.
        # Remember that the parameters go through the softmax function.
        id_conditioner[:self.n_bins].fill_(1 / self.n_bins)
        id_conditioner[self.n_bins:2*self.n_bins].fill_(1 / self.n_bins)

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
        return degrees_in.tile((self.n_parameters_per_feature,))

    def _get_parameters(self, parameters: torch.Tensor) -> tuple[torch.Tensor]:
        """Return the parameters for the functional API.

        The function normalizes widths, heights, and slopes so that they are in
        their correct domain and to enforce ``min_bin_size`` and ``min_slope``.
        It also offsets the slopes parameters so that 0 equals the identity.

        Parameters
        ----------
        parameters : torch.Tensor
            Shape ``(batch_size, n_parameters)``. The spline parameters.

        Returns
        -------
        widths : torch.Tensor
            Shape ``(batch_size, 3*n_bins, n_features)``.
        heights : torch.Tensor
            Shape ``(batch_size, 3*n_bins, n_features)``.
        slopes : torch.Tensor
            Shape ``(batch_size, 3*n_bins+1, n_features)``. The boundary slopes
            are automatically set to 1 if ``identity_boundary_slopes is True``
            or set to be identical for circular splines.
        shifts : torch.Tensor or None
            Shape ``(batch_size, n_features)``. Only returned for circular
            splines.

        """
        # From (batch_size, n_par_per_feat*n_features) to (batch_size, n_par_per_feat, n_features).
        batch_size = parameters.shape[0]
        parameters = parameters.reshape(batch_size, self.n_parameters_per_feature, -1)

        # Extract parameters.
        widths = parameters[:, :self.n_bins]
        heights = parameters[:, self.n_bins:2*self.n_bins]
        slopes = parameters[:, 2*self.n_bins:]

        # Init identity slopes.
        if self._identity_boundary_slopes:
            zeros = torch.zeros_like(widths[:, :1])

        # Handle slopes and shifts for periodic DOFs.
        if self._circular:
            # The last parameter is the shift.
            shifts = slopes[:, -1]

            # Check if we the slope of the boundary knots are fixed.
            if self._identity_boundary_slopes:
                slopes = torch.cat([zeros, slopes], dim=1)
            else: # Do not modify the original parameters.
                slopes = slopes.clone()
            # We set the slope of the last knot equal to the first.
            slopes[:, -1] = slopes[:, 0]
        else:
            shifts = None
            if self._identity_boundary_slopes:
                slopes = torch.cat([zeros, slopes, zeros], dim=1)

        # Normalize widths/heights.
        tot_width = self.xf - self.x0 - self.n_bins*self._min_bin_size
        tot_height = self._yf - self._y0 - self.n_bins*self._min_bin_size
        widths = torch.nn.functional.softmax(widths, dim=1) * tot_width + self._min_bin_size
        heights = torch.nn.functional.softmax(heights, dim=1) * tot_height + self._min_bin_size

        # Normalize slopes. The offset is such that the slope will 1 when the
        # parameters passed will be 0.
        offset = torch.log(torch.exp(1. - self._min_slope) - 1.)
        slopes = torch.nn.functional.softplus(slopes + offset) + self._min_slope

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
    first and last knot.

    Note that this function implements only the neural spline function.
    Compared, to :class:`~tfep.nn.transformers.spline.NeuralSplineTransformer`,
    it has several differences:
    - It takes as parameters directly widths, heights, and slopes of the
      K+1 knots used for interpolation rather than parameters that are then
      passed to ``softmax`` and ``softplus`` functions.
    - Periodic features are not shifted nor mapped to their domain. These are
      expected to be passed in their domain or they will be transformed
      linearly.
    - It does not guarantee a minimum bin size and slope for numerical
      stability.

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
