#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Moebius transformation for autoregressive normalizing flows.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import torch

from tfep.utils.math import batchwise_dot, batchwise_outer


# =============================================================================
# MOEBIUS TRANSFORMERS
# =============================================================================

class MoebiusTransformer(torch.nn.Module):
    r"""Moebius transformer.

    This implements a generalization of the Moebius transformation proposed in
    [1, 2] to non-unit spheres. The transformer will expand/contract the distribution
    on the sphere of radius :math:`r`, where :math:`r` is the norm of the input
    vector.

    The transformation has the form

    :math:`y = \frac{||x||^2 - ||w||^2}{||x - w||^2} (x - w) - w`

    where :math:`y, x, w` are all ``dimension``-dimensional vectors and
    :math:`||w|| < ||x||`. The function automatically rescales the ``w`` argument
    following the same strategy as in [2] to satisfy the condition on the norm.
    Consequently, ``w``s of any norm can be passed.

    The implementation of the transformation on the unit sphere is slightly more
    efficient and can be toggled with the ``unit_sphere`` argument.

    Parameters
    ----------
    dimension : int
        The dimensionality of the vectors in ``x`` and ``w``.
    max_radius : float
        Must be stringly less than 1. Rescaling of the ``w`` vectors will be
        performed so that its maximum norm will be ``max_radius * |x|``.
    unit_sphere : bool
        If ``True``, the input vectors ``x`` are assumed to be on the unit sphere,
        which makes the implementation slightly faster.

    References
    ----------
    [1] Kato S, McCullagh P. Moebius transformation and a Cauchy family
        on the sphere. arXiv preprint arXiv:1510.07679. 2015 Oct 26.
    [2] Rezende DJ, Papamakarios G, Racanière S, Albergo MS, Kanwar G,
        Shanahan PE, Cranmer K. Normalizing Flows on Tori and Spheres.
        arXiv preprint arXiv:2002.02428. 2020 Feb 6.

    """
    # Number of parameters needed by the transformer for each input dimension.
    n_parameters_per_input = 1

    def __init__(self, dimension, max_radius=0.99, unit_sphere=False):
        """Constructor."""
        super().__init__()
        self.dimension = dimension
        self.max_radius = max_radius
        self.unit_sphere = unit_sphere

    def forward(self, x, w):
        """Apply the transformation.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch_size, n_vectors*dimension)``. Contiguous elements of ``x``
            are interpreted as vectors (i.e., the first and second input vectors are
            ``x[:dimension]`` and ``x[dimension:2*dimension]``.
        w : torch.Tensor
            Shape ``(batch_size, n_vectors*dimension)``. The transformation parameters.
            These parameter vectors are automatically rescaled so that ``|w| < |x|``.

        Returns
        -------
        y : torch.Tensor
            Shape ``(batch_size, n_vectors*dimension)``. The transformed vectors.
        log_det_J : torch.Tensor
            Shape ``(batch_size,)``. The logarithm of the absolute value of the Jacobian
            determinant ``dy / dx``.

        """
        return moebius_transformer(
            x,
            w[:, 0],
            dimension=self.dimension,
            max_radius=self.max_radius,
            unit_sphere=self.unit_sphere
        )

    def inverse(self, y, w):
        """Reverse the transformation.

        Parameters
        ----------
        y : torch.Tensor
            Shape ``(batch_size, n_vectors*dimension)``. Contiguous elements of ``y``
            are interpreted as vectors (i.e., the first and second input vectors are
            ``y[:dimension]`` and ``y[dimension:2*dimension]``.
        w : torch.Tensor
            Shape ``(batch_size, n_vectors*dimension)``. The transformation parameters.
            These parameter vectors are automatically rescaled so that ``|w| < |y|``.

        Returns
        -------
        x : torch.Tensor
            Shape ``(batch_size, n_vectors*dimension)``. The transformed vectors.
        log_det_J : torch.Tensor
            Shape ``(batch_size,)``. The logarithm of the absolute value of the Jacobian
            determinant ``dx / dy``.

        """
        return moebius_transformer(
            y,
            -w[:, 0],
            dimension=self.dimension,
            max_radius=self.max_radius,
            unit_sphere=self.unit_sphere
        )

    def get_identity_parameters(self, n_features):
        """Return the value of the parameters that makes this the identity function.

        This can be used to initialize the normalizing flow to perform the identity
        transformation.

        Parameters
        ----------
        n_features : int
            The dimension of the input vector passed to the transformer.

        Returns
        -------
        w : torch.Tensor
            A tensor of shape ``(n_features,)`` representing the parameter
            vector to perform the identity function with a Moebius transformer.

        """
        return torch.zeros(size=(self.n_parameters_per_input, n_features))


class SymmetrizedMoebiusTransformer(torch.nn.Module):
    r"""Symmetrized Moebius transformer.

    This implements a generalization of the symmetrized Moebius transformation
    proposed in [1] to non-unit spheres. The transformer will expand/contract
    the distribution on the sphere of radius :math:`r`, where :math:`r` is the
    norm of the input vector.

    The transformation has the form

    :math:`y = ||f(x; w)|| \frac{f(x; w) + f(x; -w)}{||f(x; w) + f(x; -w)||}`

    where :math:`f` is the Moebius transform (see :class:``.MoebiusTransformer``),
    and :math:`y, x, w` are all ``dimension``-dimensional vectors with
    :math:`||w|| < ||x||`. The function automatically rescales the ``w`` argument
    following the same strategy as in [2] to satisfy the condition on the norm.
    Consequently, ``w``s of any norm can be passed.

    Parameters
    ----------
    dimension : int
        The dimensionality of the ``x`` and ``w`` vectors.
    max_radius : float
        Must be stringly less than 1. Rescaling of the ``w`` vectors will be
        performed so that its maximum norm will be ``max_radius * |x|``.

    References
    ----------
    [1] Köhler J, Invernizzi M, De Haan P, Noé F. Rigid body flows for sampling
        molecular crystal structures. arXiv preprint arXiv:2301.11355. 2023 Jan 26.
    [2] Rezende DJ, Papamakarios G, Racanière S, Albergo MS, Kanwar G,
        Shanahan PE, Cranmer K. Normalizing Flows on Tori and Spheres.
        arXiv preprint arXiv:2002.02428. 2020 Feb 6.

    """
    # Number of parameters needed by the transformer for each input dimension.
    n_parameters_per_input = 1

    def __init__(self, dimension, max_radius=0.99, unit_sphere=False):
        """Constructor."""
        super().__init__()
        self.dimension = dimension
        self.max_radius = max_radius

    def forward(self, x, w):
        """Apply the transformation.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch_size, n_vectors*dimension)``. Contiguous elements of ``x``
            are interpreted as vectors (i.e., the first and second input vectors are
            ``x[:dimension]`` and ``x[dimension:2*dimension]``.
        w : torch.Tensor
            Shape ``(batch_size, n_vectors*dimension)``. The transformation parameters.
            These parameter vectors are automatically rescaled so that ``|w| < |x|``.

        Returns
        -------
        y : torch.Tensor
            Shape ``(batch_size, n_vectors*dimension)``. The transformed vectors.
        log_det_J : torch.Tensor
            Shape ``(batch_size,)``. The logarithm of the absolute value of the Jacobian
            determinant ``dy / dx``.

        """
        return symmetrized_moebius_transformer(
            x,
            w[:, 0],
            dimension=self.dimension,
            max_radius=self.max_radius,
        )

    def inverse(self, y, w):
        """Reverse the transformation.

        Parameters
        ----------
        y : torch.Tensor
            Shape ``(batch_size, n_vectors*dimension)``. Contiguous elements of ``y``
            are interpreted as vectors (i.e., the first and second input vectors are
            ``y[:dimension]`` and ``y[dimension:2*dimension]``.
        w : torch.Tensor
            Shape ``(batch_size, n_vectors*dimension)``. The transformation parameters.
            These parameter vectors are automatically rescaled so that ``|w| < |y|``.

        Returns
        -------
        x : torch.Tensor
            Shape ``(batch_size, n_vectors*dimension)``. The transformed vectors.
        log_det_J : torch.Tensor
            Shape ``(batch_size,)``. The logarithm of the absolute value of the Jacobian
            determinant ``dx / dy``.

        """
        return symmetrized_moebius_transformer_inverse(
            y,
            w[:, 0],
            dimension=self.dimension,
            max_radius=self.max_radius,
        )

    def get_identity_parameters(self, n_features):
        """Return the value of the parameters that makes this the identity function.

        This can be used to initialize the normalizing flow to perform the identity
        transformation.

        Parameters
        ----------
        n_features : int
            The dimension of the input vector passed to the transformer.

        Returns
        -------
        w : torch.Tensor
            A tensor of shape ``(n_features,)`` representing the parameter
            vector to perform the identity function with a Moebius transformer.

        """
        return torch.zeros(size=(self.n_parameters_per_input, n_features))


# =============================================================================
# FUNCTIONAL API
# =============================================================================

def moebius_transformer(x, w, dimension, max_radius=0.99, unit_sphere=False):
    r"""Moebius transformer.

    This implements a generalization of the Moebius transformation proposed in
    [1, 2] to non-unit spheres. The transformer will expand/contract the distribution
    on the sphere of radius :math:`r`, where :math:`r` is the norm of the input
    vector.

    The transformation has the form

    :math:`y = \frac{||x||^2 - ||w||^2}{||x - w||^2} (x - w) - w`

    where :math:`y, x, w` are all ``dimension``-dimensional vectors and
    :math:`||w|| < ||x||`. The function automatically rescales the ``w`` argument
    following the same strategy as in [2] to satisfy the condition on the norm.
    Consequently, ``w``s of any norm can be passed.

    The implementation of the transformation on the unit sphere is slightly more
    efficient and can be toggled with the ``unit_sphere`` argument.

    Parameters
    ----------
    x : torch.Tensor
        Shape ``(batch_size, n_vectors*dimension)``. Contiguous elements of ``x``
        are interpreted as vectors (i.e., the first and second input vectors are
        ``x[:dimension]`` and ``x[dimension:2*dimension]``.
    w : torch.Tensor
        Shape ``(batch_size, n_vectors*dimension)``. The transformation parameters.
        These parameter vectors are automatically rescaled so that ``|w| < |x|``.
    dimension : int
        The dimensionality of the ``x`` and ``w``.
    max_radius : float
        Must be stringly less than 1. Rescaling of the ``w`` vectors will be
        performed so that its maximum norm will be ``max_radius * |x|``.
    unit_sphere : bool
        If ``True``, the input vectors ``x`` are assumed to be on the unit sphere,
        which makes the implementation slightly faster.

    Returns
    -------
    y : torch.Tensor
        Shape ``(batch_size, n_vectors*dimension)``. The transformed vectors.
    log_det_J : torch.Tensor
        Shape ``(batch_size,)``. The logarithm of the absolute value of the Jacobian
        determinant ``dy / dx``.

    References
    ----------
    [1] Kato S, McCullagh P. Moebius transformation and a Cauchy family
        on the sphere. arXiv preprint arXiv:1510.07679. 2015 Oct 26.
    [2] Rezende DJ, Papamakarios G, Racanière S, Albergo MS, Kanwar G,
        Shanahan PE, Cranmer K. Normalizing Flows on Tori and Spheres.
        arXiv preprint arXiv:2002.02428. 2020 Feb 6.

    """
    # From shape (batch, n_vectors*dimension) to (batch, n_vectors, dimension)
    batch_size, n_features = x.shape
    x = x.reshape(batch_size, -1, dimension)
    w = w.reshape(batch_size, -1, dimension)

    # Transform.
    y, log_det_J = _moebius_transformer(x, w, dimension, max_radius, unit_sphere)

    # Return in the correct shape.
    y = y.reshape(batch_size, n_features)
    return y, log_det_J


def symmetrized_moebius_transformer(x, w, dimension, max_radius=0.99):
    r"""Symmetrized Moebius transformer.

    This implements a generalization of the symmetrized Moebius transformation
    proposed in [1] to non-unit spheres. The transformer will expand/contract
    the distribution on the sphere of radius :math:`r`, where :math:`r` is the
    norm of the input vector.

    The transformation has the form

    :math:`y = ||f(x; w)|| \frac{f(x; w) + f(x; -w)}{||f(x; w) + f(x; -w)||}`

    where :math:`f` is the Moebius transform (see :class:``.MoebiusTransformer``),
    and :math:`y, x, w` are all ``dimension``-dimensional vectors with
    :math:`||w|| < ||x||`. The function automatically rescales the ``w`` argument
    following the same strategy as in [2] to satisfy the condition on the norm.
    Consequently, ``w``s of any norm can be passed.

    Parameters
    ----------
    x : torch.Tensor
        Shape ``(batch_size, n_vectors*dimension)``. Contiguous elements of ``x``
        are interpreted as vectors (i.e., the first and second input vectors are
        ``x[:dimension]`` and ``x[dimension:2*dimension]``.
    w : torch.Tensor
        Shape ``(batch_size, n_vectors*dimension)``. The transformation parameters.
        These parameter vectors are automatically rescaled so that ``|w| < |x|``.
    dimension : int
        The dimensionality of the ``x`` and ``w``.
    max_radius : float
        Must be stringly less than 1. Rescaling of the ``w`` vectors will be
        performed so that its maximum norm will be ``max_radius * |x|``.

    Returns
    -------
    y : torch.Tensor
        Shape ``(batch_size, n_vectors*dimension)``. The transformed vectors.
    log_det_J : torch.Tensor
        Shape ``(batch_size,)``. The logarithm of the absolute value of the Jacobian
        determinant ``dy / dx``.

    References
    ----------
    [1] Köhler J, Invernizzi M, De Haan P, Noé F. Rigid body flows for sampling
        molecular crystal structures. arXiv preprint arXiv:2301.11355. 2023 Jan 26.
    [2] Rezende DJ, Papamakarios G, Racanière S, Albergo MS, Kanwar G,
        Shanahan PE, Cranmer K. Normalizing Flows on Tori and Spheres.
        arXiv preprint arXiv:2002.02428. 2020 Feb 6.

    """
    # From shape (batch, n_vectors*dimension) to (batch, n_vectors, dimension)
    batch_size, n_features = x.shape
    x = x.reshape(batch_size, -1, dimension)
    w = w.reshape(batch_size, -1, dimension)

    # Moebius transform.
    f_w = _moebius_transformer(x, w, dimension, max_radius, unit_sphere=False, return_log_det_J=False)
    f_iw = _moebius_transformer(x, -w, dimension, max_radius, unit_sphere=False, return_log_det_J=False)
    f_symmetrized = f_w + f_iw

    # Rescale to the sphere of radius ||x||
    x_norm = torch.linalg.norm(x, dim=-1, keepdim=True)
    f_symmetrized_norm = torch.linalg.norm(f_symmetrized, dim=-1, keepdim=True)
    f_symmetrized_scaled = x_norm / f_symmetrized_norm * f_symmetrized

    # Compute the Jacobian.
    w = w.reshape(batch_size, -1, dimension)
    w_norm = torch.linalg.norm(w, dim=-1, keepdim=True)
    rescaling = max_radius / (1 + w_norm)
    w = rescaling * w
    w_norm = rescaling * w_norm
    log_det_J = _symmetrized_moebius_transform_log_det_J(x / x_norm, w, w_norm**2)

    # Return in the expected shape.
    f_symmetrized_scaled = f_symmetrized_scaled.reshape(batch_size, n_features)
    return f_symmetrized_scaled, log_det_J


def symmetrized_moebius_transformer_inverse(x, w, dimension, max_radius=0.99):
    r"""Inverse symmetrized Moebius transformer.

    See :func:`.symmetrized_moebius_transformer_inverse` for documentation.

    """
    # From shape (batch, n_vectors*dimension) to (batch, n_vectors, dimension)
    batch_size, n_features = x.shape
    x = x.reshape(batch_size, -1, dimension)
    w = w.reshape(batch_size, -1, dimension)

    # We solve the inversion first on the unit sphere, and then project back.
    x_norm = torch.linalg.norm(x, dim=-1, keepdim=True)
    x_unit = x / x_norm

    # Map parameter vector w to the solid unit sphere.
    w_norm = torch.linalg.norm(w, dim=-1, keepdim=True)
    rescaling = max_radius / (1 + w_norm)
    w_unit = rescaling * w
    w_unit_norm = rescaling * w_norm

    # Change the coordinate system so that w = [r, 0, 0, ...] and x = [a, b, 0, ...] with b = -sqrt(1-a^2)
    da = w_unit / w_unit_norm  # First basis of the new coordinate system: w
    a = batchwise_dot(x_unit, da, keepdim=True)  # Project x on first basis.

    db = x_unit - a * da  # Second orthogonal basis: p - proj(p, q)
    b = torch.linalg.norm(db, dim=-1, keepdim=True)
    db = db / b  # Normalize basis.

    # Now the inversion is analytically solvable following Köhler J, Invernizzi M,
    # De Haan P, Noé F. Rigid body flows for sampling molecular crystal structures.
    # arXiv preprint arXiv:2301.11355. 2023 Jan 26.
    r2 = w_unit_norm**2
    numer = - a * (r2 + 1.0)
    denom = torch.sqrt(1 + r2**2 + r2 * (4*a**2 - 2))
    a_inv = numer / denom
    b_inv = - torch.sqrt(1 - a_inv**2)

    # Project back on the unit hyper-sphere.
    x_unit_inv = - (a_inv*da + b_inv*db)

    # Compute change of volume as the negative of the forward transformation.
    # The contribution from dividing and multiplying by ||x|| cancel out.
    log_det_J = - _symmetrized_moebius_transform_log_det_J(x_unit_inv, w_unit, r2)

    # Project back on the hyper-sphere of radius ||x||.
    x_inv = x_norm * x_unit_inv

    # Return in the expected shape.
    x_inv = x_inv.reshape(batch_size, n_features)
    return x_inv, log_det_J


# =============================================================================
# INTERNAL USE
# =============================================================================

def _moebius_transformer(x, w, dimension, max_radius, unit_sphere, return_log_det_J=True):
    r"""Moebius transformer.

    This is the same as moebius_transformer(), but it returns the log_det_J only
    optionally.

    ``x`` and ``w`` must have shape (batch, n_vectors, dimension).

    The returned value has also shape (batch, n_vectors, dimension)

    """
    # Compute the radius of the vectors.
    w_norm = torch.linalg.norm(w, dim=-1, keepdim=True)

    # First map the w vectors from R^d to the solid sphere of radius x_norms.
    rescaling = max_radius / (1 + w_norm)
    if not unit_sphere:
        x_norm = torch.linalg.norm(x, dim=-1, keepdim=True)
        rescaling = x_norm * rescaling
    w = rescaling * w
    w_norm = rescaling * w_norm

    # Compute the transformed vectors.
    if unit_sphere:
        numerator = 1 - w_norm**2
    else:
        numerator = x_norm**2 - w_norm**2
    diff = x - w
    diff_norm = torch.linalg.norm(diff, dim=-1, keepdim=True)
    y = numerator / diff_norm.pow(2) * diff - w

    if not return_log_det_J:
        return y

    # Compute the log det Jacobian of the transformation on the unit sphere..
    numerator = numerator.unsqueeze(-1)
    diff_norm = diff_norm.unsqueeze(-1)

    dd_outer = batchwise_outer(diff, diff)
    eye = torch.eye(dimension).expand_as(dd_outer)
    jac = numerator * (eye / diff_norm.pow(2) - 2 / diff_norm.pow(4) * dd_outer)

    # Now compute the Jacobian of the transformation on the sphere of radius x_norm
    if not unit_sphere:
        x_norm_expand = x_norm.unsqueeze(-1)
        jac2 = eye - batchwise_outer(x, x) / x_norm_expand**2
        jac = torch.einsum("...ij, ...jk -> ...ik", jac, jac2)  # Batchwise matrix multiplication.
        jac = batchwise_outer(y, x) / x_norm_expand**2 + jac

    # Compute the determinants of the blocks.
    log_det_J = torch.linalg.slogdet(jac)[1]
    log_det_J = log_det_J.sum(dim=-1)

    return y, log_det_J


def _symmetrized_moebius_transform_log_det_J(x, w, r2):
    """Compute the log_det_J of the symmetrized Moebius transform.

    This is based on: Köhler J, Invernizzi M, De Haan P, Noé F. Rigid body flows
    for sampling molecular crystal structures. arXiv preprint arXiv:2301.11355.
    2023 Jan 26.

    ``x`` (input) and ``w`` (parameter) must be within the unit sphere and have
    shape (batch_size, n_vectors, dimension).

    ``r2`` is the norm of ``w`` with shape (batch_size, n_vectors, 1).

    Return log_det_J with shape (batch_size,).

    """
    dimension = x.shape[-1]
    qy2 = r2 - batchwise_dot(x, w, keepdim=True)**2
    numer = (1 - r2) * (1 + r2)**(dimension-1)
    denom = (4*qy2 + (1 - r2)**2)**(dimension / 2)
    dV = numer / denom
    log_det_J = torch.log(dV).squeeze(-1).sum(dim=1)
    return log_det_J
