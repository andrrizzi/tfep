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

from typing import Dict, List, Literal, Optional, Tuple

import torch

from tfep.utils.math import batchwise_dot, batchwise_outer


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

        The angle is from 0 to pi.

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
    k = torch.nn.functional.normalize(directions, dim=-1)
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


# =============================================================================
# COORDINATE TRANSFORMATIONS
# =============================================================================

# Map the name of an axis to its 3D unit vector representation. We instantiate
# the tensor in get_axis_from_name to make sure it is represented by the default
# floating type, which might not be set on import.
_AXIS_NAME_TO_VECTOR: Dict[Literal['x', 'y', 'z'], List] = {
    'x': [1.0, 0.0, 0.0],
    'y': [0.0, 1.0, 0.0],
    'z': [0.0, 0.0, 1.0],
}


def get_axis_from_name(name: Literal['x', 'y', 'z']) -> torch.Tensor:
    """Return the 3D vector representation of an axis.

    Parameters
    ----------
    name : Literal['x', 'y', 'z']
        The name of the axis.

    Returns
    -------
    axis : torch.Tensor
        Shape ``(3,)``. The unit vector representation of the axis.

    """
    return torch.tensor(_AXIS_NAME_TO_VECTOR[name])


def reference_frame_rotation_matrix(
        axis_atom_positions: torch.Tensor,
        plane_atom_positions: torch.Tensor,
        axis: torch.Tensor,
        plane_axis: torch.Tensor,
        plane_normal: Optional[torch.Tensor] = None,
        project_on_positive_axis: bool = False
) -> torch.Tensor:
    """Return the rotation matrix required to rotate the frame of reference based on two atoms.

    After the rotation matrix is applied to the coordinates, ``axis_atom_positions``
    lie on the given ``axis`` vector while ``plane_atom_positions`` lie on the
    plane spanned by the ``axis`` and ``plane_axis`` vectors.

    Parameters
    ----------
    axis_atom_positions : torch.Tensor
        Shape ``(batch_size, 3)``. The position of the atom placed on ``axis``.
    plane_atom_positions : torch.Tensor
        Shape ``(batch_size, 3)``. The position of the atom placed on the
        ``axis``-``plane_axis`` plane.
    axis : torch.Tensor
        Shape ``(3,)``. The axis on which to the axis atom is placed. Must be
        a unit vector.
    plane_axis : torch.Tensor,
        Shape ``(3,)``. The second axis used to determine the plane where the
        plane atom is placed. Must be a unit vector and not parallel to ``axis``.
    plane_normal : Optional[torch.Tensor]
        The vector normal to ``axis`` and ``plane_axis``. If not given, it is
        computed here.
    project_on_positive_axis : bool
        If ``True``, the axis atom is rotated so that it always lies on the positive
        ``axis``. Otherwise, it is rotated on the positive or negative ``axis`` based
        on whichever is closest.

        Note that if this is ``True``, a transformation that flips the sign of
        the coordinate of the axis atom might become impossible to invert in practice.

    Returns
    -------
    rotation_matrices : torch.Tensor
        Shape ``(batch_size, 3, 3)``. The rotation matrices.

    Examples
    --------

    >>> # Initialize the coordinates.
    >>> batch_size, n_atoms = 2, 4
    >>> coordinates = torch.randn(batch_size, n_atoms, 3)

    >>> # Fix the orientation of the coordiante frames based on the 2nd and 4th atoms.
    >>> axis_atom_pos = coordinates[:, 1]
    >>> plane_atom_pos = coordinates[:, 3]
    >>> rotation_matrices = reference_frame_rotation_matrix(
    ...     axis_atom_pos,
    ...     plane_atom_pos,
    ...     axis=torch.tensor([1.0, 0, 0]),  # axis atom lies on x-axis
    ...     plane_axis=torch.tensor([0.0, 0, 1]),  # plane atomlies on x-z plane
    ... )
    ...

    >>> # Rotate the coordinates.
    >>> new_coordinates = batchwise_rotate(coordinates, rotation_matrices)
    >>> # Reverse the change of reference frame.
    >>> old_coordinates = batchwise_rotate(new_coordinates, rotation_matrices, inverse=True)

    """
    # Default argument.
    if plane_normal is None:
        plane_normal = torch.cross(axis, plane_axis)

    # Find the direction perpendicular to the plane formed by the axis atom,
    # and the axis. rotation_vectors has shape (batch_size, 3).
    rotation_vectors = torch.cross(axis_atom_positions, axis.unsqueeze(0), dim=1)

    # Find the first rotation angle. r1_angle has shape (batch_size,).
    r1_angles = vector_vector_angle(axis_atom_positions, axis)

    # r1_angles goes from 0 to pi. We want to rotate the point onto the
    # negative/positive axis, depending which is closest.
    if not project_on_positive_axis:
        r1_angles = r1_angles - torch.pi * (r1_angles > torch.pi/2).to(r1_angles.dtype)

    # This are the rotation matrices that bring the axis points onto the axis.
    r1_rotation_matrices = rotation_matrix_3d(r1_angles, rotation_vectors)

    # To bring the plane atom in position, we perform a rotation about
    # axis so that we don't modify the position of the axis atom. We
    # perform the first rotation only on the atom position that will
    # determine the next rotation matrix for now so that we run only
    # a single matmul on all atoms.
    plane_points = plane_atom_positions.unsqueeze(1)
    plane_points = batchwise_rotate(plane_points, r1_rotation_matrices)
    plane_points = plane_points.squeeze(1)

    # Project the atom on the plane perpendicular to the rotation axis plane
    # to measure the rotation angle.
    plane_points = plane_points - axis*batchwise_dot(plane_points, axis, keepdim=True)
    r2_angles = vector_plane_angle(plane_points, plane_normal)

    # r2_angles will be positive in the octants where plane_normal lies
    # and negative in the opposite direction but the rotation happens
    # counterclockwise/clockwise with positive/negative angle so we need
    # to fix the sign of the angle based on where it is.
    r2_angles_sign = -torch.sign(batchwise_dot(plane_points, plane_axis))
    r2_rotation_matrices = rotation_matrix_3d(r2_angles_sign * r2_angles, axis)

    # Now build the rotation composition.
    rotation_matrices = torch.bmm(r2_rotation_matrices, r1_rotation_matrices)

    return rotation_matrices


def cartesian_to_polar(x: torch.Tensor, y: torch.Tensor, return_log_det_J: bool = False) -> Tuple[torch.Tensor]:
    """Transform Cartesian coordinates into polar.

    Parameters
    ----------
    x : torch.Tensor
        Shape ``(batch_size,)``. The x Cartesian coordinate.
    y : torch.Tensor
        Shape ``(batch_size,)``. The y Cartesian coordinate.
    return_log_det_J: bool, optional
        If ``True``, the absolute value of the Jacobian determinant of the
        transformation is also returned.

    Returns
    -------
    r : torch.Tensor
        Shape ``(batch_size,)``. The radius coordinate.
    angle : torch.Tensor
        Shape ``(batch_size,)``. The angle coordinate in radians.
    log_det_J : torch.Tensor, optional
        The absolute value of the Jacobian determinant of the transformation.

    """
    r = (x.pow(2) + y.pow(2)).sqrt()
    angle = torch.atan2(y, x)
    if return_log_det_J:
        return r, angle, -torch.log(r)
    return r, angle


def polar_to_cartesian(r: torch.Tensor, angle: torch.Tensor, return_log_det_J: bool = False) -> Tuple[torch.Tensor]:
    """Transform polar coordinates into Cartesian.

    Parameters
    ----------
    r : torch.Tensor
        Shape ``(batch_size,)``. The radius coordinate.
    angle : torch.Tensor
        Shape ``(batch_size,)``. The angle coordinate in radians.
    return_log_det_J: bool, optional
        If ``True``, the absolute value of the Jacobian determinant of the
        transformation is also returned.

    Returns
    -------
    x : torch.Tensor
        Shape ``(batch_size,)``. The x Cartesian coordinate.
    y : torch.Tensor
        Shape ``(batch_size,)``. The y Cartesian coordinate.
    log_det_J : torch.Tensor, optional
        The absolute value of the Jacobian determinant of the transformation.

    """
    x = r * torch.cos(angle)
    y = r * torch.sin(angle)
    if return_log_det_J:
        return x, y, torch.log(r)
    return x, y
