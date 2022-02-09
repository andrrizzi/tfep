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

from tfep.utils.math import batchwise_dot, batchwise_outer


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


# =============================================================================
# INTERNAL COORDINATES
# =============================================================================

def pdist(x, pairs=None, return_diff=False):
    """Compute p-norm distances between pairs of row vectors.

    In comparison to ``torch.nn.functional.pdist``, the function can handle
    batches and compute distances between a subset of all pairs. Only the
    Euclidean norm is currently supported.

    Parameters
    ----------
    x : torch.Tensor
        Positions of particles with shape ``(batch_size, n_particles, D)``, where
        ``D`` is the dimensionality of the vector space.
    pairs : torch.Tensor, optional
        A tensor of shape ``(2, n_pairs)``. For each batch sample, the function
        will compute the ``i``-th distance between the ``pairs[0, i]``-th and the
        ``pairs[1, i]``-th atoms. If not passed, all pairwise distances are computed.
    return_diff : bool, optional
        If ``True``, the difference vector between pairs of particles are also
        returned.

    Returns
    -------
    distances : torch.Tensor
        This has shape ``(batch_size, n_pairs)``, and ``distances[b, i]`` is the
        distance between the particles of the ``i``-th pair for the ``b``-th batch
        sample
    diff : torch.Tensor, optional
        This has shape ``(batch_size, n_pairs, 3)``, and ``diff[b, i]`` is the
        vector ``p1-p0``, where ``pX`` is the position of particle ``pairs[b, X]``.
        This is returned only if ``return_diff`` is ``True``.

    """
    n_particles = x.shape[-2]
    if pairs is None:
        pairs = torch.triu_indices(n_particles, n_particles, offset=1)

    diff = x[:, pairs[1]] - x[:, pairs[0]]
    distances = torch.sqrt(torch.sum(diff**2, dim=-1))
    if return_diff:
        return distances, diff
    return distances


def vector_vector_angle(x1, x2):
    """Return the angle in radians between a two vectors.

    If both ``x1`` and ``x2`` have multiple vectors, the angles are computed
    in a batchwise fashion.

    Parameters
    ----------
    x1 : torch.Tensor
        A tensor of shape ``(*, D)``, where ``D`` is the vector dimensionality.
    x2 : torch.Tensor
        A tensor of shape ``(*, D)``, where ``D`` is the vector dimensionality.

    Returns
    -------
    angles : torch.Tensor
        A tensor of shape ``(*,)``. As an example, if both inputs have shape
        ``(batch_size, D)``, then ``angles`` has shape ``(batch_size,)`` and
        ``angles[i]`` is the angle between vectors ``x1[i]`` and ``x2[i]``.

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


def proper_dihedral_angle(x1, x2, x3):
    """Compute the proper dihedral angle between the plane ``x1``-``x2`` and ``x2``-``x3``.

    If both ``x1``, ``x2``, and ``x3`` have multiple vectors, the angles are
    computed in a batchwise fashion.

    In the description of the parameters, we will use the example of four atoms
    at positions p0, p1, p2, and p3.

    Parameters
    ----------
    x1 : torch.Tensor
        The vector p1 - p0 with shape ``(*, D)``, where ``D`` is the vector
        dimensionality.
    x2 : torch.Tensor
        The vector p2 - p1 with shape ``(*, D)``, where ``D`` is the vector
        dimensionality.
    x3 : torch.Tensor
        The vector p3 - p2 with shape ``(*, D)``, where ``D`` is the vector
        dimensionality.

    Returns
    -------
    dihedrals : torch.Tensor
        A tensor of shape ``(*,)``. As an example, if all inputs have shape
        ``(batch_size, D)``, then ``dihedrals`` has shape ``(batch_size,)`` and
        ``dihedrals[i]`` is the angle between the planes ``x1[i]``-``x2[i]`` and
        ``x2[i]``-``x3[i]``.

    """
    # The implementation is from Praxeolitic.
    # see: https://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python
    x1 = -x1

    # normalize x2 so that it does not influence magnitude of vector
    # rejections that come next
    x2 = x2 / torch.linalg.vector_norm(x2, dim=-1, keepdim=True)

    # vector rejections
    # v = projection of x1 onto plane perpendicular to x2
    #   = x1 minus component that aligns with x2
    # w = projection of x3 onto plane perpendicular to x2
    #   = x3 minus component that aligns with x2
    v = x1 - batchwise_dot(x1, x2, keepdim=True) * x2
    w = x3 - batchwise_dot(x3, x2, keepdim=True) * x2

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = batchwise_dot(v, w)
    x2_cross_v = torch.cross(x2, v, dim=-1)
    y = batchwise_dot(x2_cross_v, w)
    return torch.atan2(y, x)


# =============================================================================
# ROTATION
# =============================================================================

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
