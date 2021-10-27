#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Math and geometry utility functions to manipulate coordinates.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import torch


# =============================================================================
# MATH
# =============================================================================

def batchwise_dot(x1, x2, keepdim=False):
    """Batchwise dot product between two batches of tensors.

    Parameters
    ----------
    x1 : torch.Tensor
        A tensor of shape ``(batch_size, N)`` or ``(N,)``.
    x2 : torch.Tensor
        A tensor of shape ``(batch_size, N)`` or ``(N,)``.
    keepdim : bool, optional
        If ``True``, the return value has shape ``(batch_size, 1)``.
        Otherwise ``(batch_size,)``.

    Returns
    -------
    result : torch.Tensor
        ``result[i]`` is the dot product between ``x1[i]`` and ``x2[i]``

    """
    return (x1 * x2).sum(dim=-1, keepdim=keepdim)


def batchwise_outer(x1, x2):
    """Batchwise outer product between two 2D tensors.

    Parameters
    ----------
    x1 : torch.Tensor
        A tensor of shape ``(batch_size, N)``.
    x2 : torch.Tensor
        A tensor of shape ``(batch_size, N)``.

    Returns
    -------
    result : torch.Tensor
        A tensor shape ``(batch_size, N, N)``, where ``result[b][i][j]`` is the
        outer product between ``x1[b][i]`` and ``x2[b][j]``.

    """
    # return torch.einsum('bi,bj->bij', x1, x2)
    return torch.matmul(x1[:, :, None], x2[:, None, :])


def cov(x, ddof=1, dim_n=1, inplace=False):
    """Return the covariance matrix of the data.

    Parameters
    ----------
    x : torch.Tensor
        A tensor of shape ``(m, n)``, where ``n`` is the number of samples
        used to estimate the covariance, and ``m`` is the dimension of
        the multivariate variable. If ``dim_n`` is 0, then the expected
        shape is ``(n, m)``.
    ddof : int, optional
        The number of dependent degrees of freedom. The covariance will
        be estimated dividing by ``n - ddof``. Default is 1.
    dim_n : int, optional
        The dimension used to collect the samples. Default is 1.
    inplace : bool, optional
        If ``True``, the input argument ``x`` is modified to be centered
        on its mean. Default is ``False``.

    Returns
    -------
    cov : torch.Tensor
        A tensor of shape ``(m, m)``.

    """
    if len(x.shape) != 2:
        raise ValueError('The function supports only 2D matrices')
    if dim_n not in {0, 1}:
        raise ValueError('dim_n must be either 0 or 1')

    # Center the data on the mean.
    if dim_n == 1:
        keepdim = True
    else:
        keepdim = False
    mean = torch.mean(x, dim_n, keepdim=keepdim)
    if inplace:
        x -= mean
    else:
        x = x - mean

    # Average normalization factor.
    n = x.shape[dim_n] - ddof

    # Compute the covariance matrix
    if dim_n == 0:
        c = x.t().matmul(x) / n
    else:
        c = x.matmul(x.t()) / n

    return c


# =============================================================================
# GEOMETRY
# =============================================================================

def normalize_vector(x):
    """Return the normalized vector.

    Parameters
    ----------
    x : torch.Tensor
        Tensor of shape ``(batch_size, N)`` or ``(N,)``.

    Returns
    -------
    norm_x : torch.Tensor
        Normalized tensor of same shape as ``v``.

    """
    return x / torch.linalg.vector_norm(x, dim=-1, keepdim=True)


def vector_vector_angle(x1, x2):
    """Return the angle in radians between a batch of vectors and another vector.

    Parameters
    ----------
    x1 : torch.Tensor
        A tensor of shape ``(batch_size, N)`` or ``(N,)``.
    x2 : torch.Tensor
        A tensor of shape ``(N,)``.

    Returns
    -------
    angle : torch.Tensor
        A tensor of shape ``(batch_size,)`` where ``angle[i]`` is the angle
        between vectors ``x1[i]`` and ``x2``.

    """
    x1_norm = torch.linalg.vector_norm(x1, dim=-1)
    x2_norm = torch.linalg.vector_norm(x2, dim=-1)
    cos_theta = batchwise_dot(x1, x2) / (x1_norm * x2_norm)
    # Catch round-offs.
    cos_theta = torch.clamp(cos_theta, min=-1, max=1)
    return torch.acos(cos_theta)


def vector_plane_angle(x, plane):
    """Return the angle in radians between a batch of vectors and another vector.

    Parameters
    ----------
    x : torch.Tensor
        A tensor of shape ``(batch_size, N)`` or ``(N,)``.
    plane : torch.Tensor
        A tensor of shape ``(N,)`` that represent a normal vector to the plane.

    Returns
    -------
    angle : torch.Tensor
        A tensor of shape ``(batch_size,)`` where ``angle[i]`` is the angle
        between vector ``x[i]`` and plane ``plane``.

    """
    x_norm = torch.linalg.vector_norm(x, dim=-1)
    plane_norm = torch.linalg.vector_norm(plane, dim=-1)
    cos_theta = batchwise_dot(x, plane) / (x_norm * plane_norm)
    # Catch round-offs.
    cos_theta = torch.clamp(cos_theta, min=-1, max=1)
    return torch.asin(cos_theta)  # asin(x) = pi/2 - acos(x).


def rotation_matrix_3d(angles, directions):
    """Return the matrix rotating vectors for the given angle about a direction.

    The rotation matrix is built using Rodrigues' rotation formula.

    Parameters
    ----------
    angles : torch.Tensor
        A tensor of shape ``(batch_size,)``.
    directions : torch.Tensor
        A tensor of shape ``(batch_size, 3)`` or ``(3,)``.

    Returns
    -------
    R : torch.Tensor
        ``R[i]`` is the 3 by 3 matrix rotating by the angle ``angles[i]`` about
        the vector ``directions[i]``.

    """
    batch_size = len(angles)
    sina = torch.sin(angles)
    cosa = torch.cos(angles)

    # unit rotation vectors (batch_size, 3).
    k = normalize_vector(directions)
    if len(k.shape) < 2:
        k = k.unsqueeze(0)

    # Reshape cosa to have (batch_size, 1, 1) dimension.
    cosa = cosa.unsqueeze(-1).unsqueeze(-1)

    # R[i] is cosa[i] * torch.eye(3).
    R = cosa * torch.eye(3).expand(batch_size, 3, 3)

    # New term of R[i] is outer(k[i], k[i]) * (1 - cosa[i]).
    R = R + (1 - cosa) * batchwise_outer(k, k)

    # Last term of R[i] is cross_product_matrix(k[i]) * sina[i]
    sina_k = sina.unsqueeze(-1) * k

    # cross_matrix has shape (3, 3, batch_size)
    zeros = torch.zeros_like(angles)
    cross_matrix = torch.stack([
        torch.stack([zeros, -sina_k[:,2], sina_k[:,1]]),
        torch.stack([sina_k[:,2], zeros, -sina_k[:,0]]),
        torch.stack([-sina_k[:,1], sina_k[:,0], zeros]),
    ])

    # Put batch_size back at the beginning to sum correctly.
    R = R + cross_matrix.permute(2, 0, 1)

    return R


def batchwise_rotate(x, rotation_matrices, inverse=False):
    """Rotate a batch of configurations with their respective rotation matrix.

    Parameters
    ----------
    x : torch.Tensor
        A tensor of shape ``(batch_size, n_vectors, 3)``.
    rotation_matrices : torch.Tensor
        A tensor of shape ``(batch_size, 3, 3)``.
    inverse : bool, optional
        If ``True`` the inverse rotation is performed.

    Returns
    -------
    y : torch.Tensor
        A tensor of shape ``(batch_size, n_vectors, 3))`` where ``y[b][i]`` is
        the result of rotating vector ``x[b][i]`` with the rotation matrix
        ``rotation_matrices[b]``.

    """
    if inverse:
        return torch.bmm(x, rotation_matrices)
    else:
        return torch.bmm(x, rotation_matrices.permute(0, 2, 1))
