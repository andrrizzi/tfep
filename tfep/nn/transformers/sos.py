#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Sum-of-squares polynomial transformer for autoregressive normalizing flows.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import numpy as np
import torch
import torch.autograd

from tfep.nn.transformers.transformer import Transformer


# =============================================================================
# SUM-OF-SQUARES POLYNOMIAL TRANSFORMER
# =============================================================================

class SOSPolynomialTransformer(Transformer):
    """Sum-of-squares polynomial transformer module for autoregressive normalizing flows.

    This is an implementation of the polynomial transformer proposed in [1].

    :math:`y_i = a_0 + \int_0^{x_i} \sum_{k=1}^K \left( \sum_{l=0}^L a_{kl} z^l \right)^2 dz`

    where :math:`K` and :math:`L` are the total number and degree of the polynomials
    respectively, and :math:`a_X` represent the parameters of the transformer.

    Only sums of squared first-degree polynomials (i.e., L=1) are currently
    supported as they are the only one with an analytic inverse and sum of
    zeroth degree polynomials (i.e., L=0) are equivalent to affine transformer.

    See Also
    --------
    nets.functions.transformer.sos_polynomial_transformer

    References
    ----------
    [1] Jaini P, Selby KA, Yu Y. Sum-of-Squares Polynomial Flow. arXiv
        preprint arXiv:1905.02325. 2019 May 7.

    """
    def __init__(self, n_polynomials=2):
        """Constructor.

        Parameters
        ----------
        n_polynomials : int
            The functional form of this transformer is a sum of squared polynomials.
            This is the number of such polynomials, which must be greater than
            1. The more polynomials, the greater the number of parameters. Default
            is 2.

        """
        super().__init__()
        if n_polynomials < 2:
            raise ValueError('n_polynomials must be strictly greater than 1.')
        self.n_polynomials = n_polynomials

    @property
    def degree_polynomials(self):
        """The degree of each squared polynomial."""
        return 1

    @property
    def parameters_per_polynomial(self):
        """Numer of parameters needed by the transformer for each squared polynomial."""
        return self.degree_polynomials + 1

    @property
    def n_parameters_per_input(self):
        """Number of parameters needed by the transformer for each input dimension."""
        return self.parameters_per_polynomial * self.n_polynomials + 1

    def forward(self, x: torch.Tensor, parameters: torch.Tensor) -> tuple[torch.Tensor]:
        """Apply the transformation to the input.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch_size, n_features)``. Input tensor.
        parameters : torch.Tensor
            Shape ``(batch_size, (1 + K*L)*n_features)``. The coefficients of the
            squared polynomials obtained from the conditioner. The coefficients
            are ordered by polynomial so that ``parameters[:,0]`` is :math:`a_0`
            followed by :math:`a_{10}, a_{11}, ..., a_{K0}, a_{K1}`.

        Returns
        -------
        y : torch.Tensor
            Shape ``(batch_size, n_vectors*dimension)``. The transformed vectors.
        log_det_J : torch.Tensor
            Shape ``(batch_size,)``. The logarithm of the absolute value of the Jacobian
            determinant ``dy / dx``.

        """
        # From (batch, n_parameters*n_features) to (batch, n_parameters, n_features).
        batch_size = parameters.shape[0]
        parameters = parameters.reshape(batch_size, self.n_parameters_per_input, -1)
        return sos_polynomial_transformer(x, parameters)

    def inverse(self, y: torch.Tensor, parameters: torch.Tensor) -> tuple[torch.Tensor]:
        """Currently not implemented."""
        raise NotImplementedError(
            'Inversion of SOS polynomial transformer has not been implemented yet.')

    def get_parameters_identity(self, n_features: int) -> torch.Tensor:
        """Return the value of the parameters that makes this the identity function.

        This can be used to initialize the normalizing flow to perform the identity
        transformation.

        Parameters
        ----------
        n_features : int
            The dimension of the input vector of the transformer.

        Returns
        -------
        parameters : torch.Tensor
            Shape ``(1+K*L, n_features)`` where ``K`` and ``L`` are the number
            and degree of the polynomials.

        """
        id_conditioner = torch.zeros(size=(self.n_parameters_per_input, n_features))
        # The sum of the squared linear parameters must be 1.
        id_conditioner[1::self.parameters_per_polynomial].fill_(np.sqrt(1 / self.n_polynomials))
        return id_conditioner.flatten()


# =============================================================================
# FUNCTIONAL API
# =============================================================================

class SOSPolynomialTransformerFunc(torch.autograd.Function):
    r"""Implement the sum-of-squares polynomial transformer for triangular maps.

    This provides a functional API for the :class:`~tfep.nn.transformers.SOSPolynomialTransformer`
    layer. It implements the polynomial transformer proposed in [1].

    :math:`y_i = a_0 + \int_0^{x_i} \sum_{k=1}^K \left( \sum_{l=0}^L a_{kl} z^l \right)^2 dz`

    where :math:`K` and :math:`L` are the total number and degree of the polynomials
    respectively, and :math:`a_X` represent the parameters of the transformer.

    The function returns the transformed feature as a ``Tensor`` of shape
    ``(batch_size, n_features)`` and the log absolute determinant of its
    Jacobian as a ``Tensor`` of shape ``(batch_size,)``.

    Only sums of squared first-degree polynomials (i.e., L=1) are currently
    supported as they are the only one with an analytic inverse and sum of
    zeroth degree polynomials (i.e., L=0) are equivalent to affine transformer.

    Parameters
    ----------
    x : torch.Tensor
        Shape ``(batch_size, n_features)``. Input tensor x.
    parameters : torch.Tensor
        Shape ``(batch_size, 1+K*L, n_features)``. The coefficients of the squared
        polynomials obtained from the conditioner. The coefficients are ordered
        by polynomial so that ``parameters[:,0]`` is :math:`a_0` followed by
        :math:`a_{10}, a_{11}, ..., a_{K0}, a_{K1}`.

    Returns
    -------
    y : torch.Tensor
        Output tensor of shape ``(batch_size, n_features)``.
    log_det_J : torch.Tensor
        The logarithm of the absolute value of the determinant of the Jacobian
        of the transformation with shape ``(batch_size,)``.

    References
    ----------
    [1] Jaini P, Selby KA, Yu Y. Sum-of-Squares Polynomial Flow. arXiv
        preprint arXiv:1905.02325. 2019 May 7.

    """

    @staticmethod
    def forward(ctx, x, parameters):
        # Compute the parameters of the sos polynomial.
        sos_degree_coefficients = SOSPolynomialTransformerFunc.get_sos_poly_coefficients(parameters)

        # Compute the power of x.
        x_powers = [x, x*x]

        # Compute y and the gradient of y w.r.t. x.
        y = sos_degree_coefficients[1].clone()
        grad_x = sos_degree_coefficients[1].clone()

        for degree, coef in enumerate(sos_degree_coefficients[2:]):
            term = coef * x_powers[degree]
            y += term
            grad_x += (degree+2) * term

        y *= x
        y += sos_degree_coefficients[0]

        log_det_J = torch.sum(torch.log(grad_x), dim=1)

        # Save tensor used for backward() before returning.
        ctx.save_for_backward(grad_x, parameters, *x_powers)

        # We don't need to compute gradients of log_det_J.
        ctx.mark_non_differentiable(log_det_J)
        return y, log_det_J

    @staticmethod
    def backward(ctx, grad_y, grad_log_det_J):
        saved_grad_x, parameters, x, x2 = ctx.saved_tensors
        grad_x = grad_parameters = None
        batch_size, n_features = saved_grad_x.shape

        # Compute gradients w.r.t. input parameters.
        if ctx.needs_input_grad[0]:
            grad_x = saved_grad_x * grad_y

        if ctx.needs_input_grad[1]:
            grad_parameters = torch.empty_like(parameters)

            # The first coefficient is the constant term.
            grad_parameters[:, 0] = torch.ones(
                size=(batch_size, n_features), dtype=saved_grad_x.dtype)

            # Zeroth and first degree terms of the inner polynomials.
            zeroth_degree_terms = parameters[:, 1::2]
            first_degree_terms = parameters[:, 2::2]

            # We need to add a dimension corresponding to the number of
            # coefficients in the power of x for them to be broadcastable.
            x = x.unsqueeze(1)
            x2 = x2.unsqueeze(1)
            x3 = x2 * x

            grad_parameters[:, 1::2] = first_degree_terms*x2 + 2*zeroth_degree_terms*x
            grad_parameters[:, 2::2] = 2/3*first_degree_terms*x3 + zeroth_degree_terms*x2

            grad_parameters = grad_parameters * grad_y.unsqueeze(1)

        return grad_x, grad_parameters

    @staticmethod
    def get_sos_poly_coefficients(parameters):
        """Compute the coefficient of the SOS polynomial.

        Parameters
        ----------
        parameters : torch.Tensor
            The coefficients of the squared polynomials obtained from the
            conditioner. Each ``Tensor`` has shape ``(batch_size, 1+K*L, n_features)``.
            The coefficients are ordered by polynomial so that ``parameters[:,0]``
            is :math:`a_0` followed by :math:`a_{10}, a_{11}, ..., a_{K0}, a_{K1}`.

        Returns
        -------
        sos_poly_coefficients : List[torch.Tensor]
            ``sos_poly_coefficients[i]`` is a tensor of shape ``(batch_size, n_features)``
            with the coefficients of the term of the SOS polynomial of degree ``i``.

        """
        # We support only L=1 for now. Number of coefficients in
        # each summed polynomials include also the constant term.
        coeff_per_inner_poly = 2
        batch_size, _, n_features = parameters.shape

        # inner_degree_parameters[d][b][p] is the parameter for the term of
        # the p-th inner polynomial of degree d for the b-th batch sample.
        inner_degree_coefficients = []
        for degree in range(coeff_per_inner_poly):
            inner_degree_coefficients.append(parameters[:, 1+degree::coeff_per_inner_poly])

        # Find the coefficients of the integrated polynomial.
        sos_degree_coefficients = [parameters[:, 0]]
        sos_degree_coefficients.append(torch.sum(inner_degree_coefficients[0]**2, dim=1))
        sos_degree_coefficients.append(torch.sum(inner_degree_coefficients[0]*inner_degree_coefficients[1], dim=1))
        sos_degree_coefficients.append(torch.sum(inner_degree_coefficients[1]**2, dim=1) / 3)

        return sos_degree_coefficients


# Functional notation.
sos_polynomial_transformer = SOSPolynomialTransformerFunc.apply
