#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Transformation that constrains the rotational degrees of freedom.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

from typing import Optional, Literal

import torch

from tfep.nn.flows.partial import PartialFlow
from tfep.utils.geometry import (
    vector_vector_angle, vector_plane_angle,
    rotation_matrix_3d, batchwise_rotate)
from tfep.utils.math import batchwise_dot
from tfep.utils.misc import (
    flattened_to_atom, atom_to_flattened, atom_to_flattened_indices)


# =============================================================================
# FRAME TRANSFORMATIONS
# =============================================================================

class OrientedFlow(PartialFlow):
    """A transformation constraining the rotational degrees of freedom.

    .. note::
        This flow currently supports only rotation in 3D space.

    This flow performs the following operations:
    - Rotates the frame of references so that two points selected by the user lie
      on an axis and plane (also selected by the user).
    - Removes 3 degrees of freedom (DOFs) from the input features (2 DOFs from
      the coordinates of the contrained point on the axis and 1 DOF from the
      point on the plane) and runs the remaining features through the wrapped
      flow.
    - Re-adds the 3 constrained DOFs.
    - Optionally, rotates the system to restore the original frame of reference.

    The flow assumes that the input features have shape ``(batch_size, n_dofs)``,
    and each batch sample represent a sequence of points in a 3D space. The
    points must must be listed so that ``input[b][3*i:3*i+3]`` are the x, y, and
    z coordinates (in this order) of the ``i``-th point for batch sample ``b``.

    """

    # Conversion string representation to vector representation.
    _AXES = {
        'x': torch.tensor([1.0, 0.0, 0.0]),
        'y': torch.tensor([0.0, 1.0, 0.0]),
        'z': torch.tensor([0.0, 0.0, 1.0]),
    }

    def __init__(
            self,
            flow: torch.nn.Module,
            axis_point_idx: Optional[int] = None,
            plane_point_idx: Optional[int] = None,
            axis: Literal['x', 'y', 'z'] = 'x',
            plane: Literal['xy', 'yz', 'xz'] = 'xy',
            round_off_imprecisions: bool = True,
            rotate_back: bool = True,
            return_partial: bool = False,
    ):
        """Constructor.

        Parameters
        ----------
        flow : torch.nn.Module
            The wrapped flow.
        axis_point_idx : int, optional
            The index of the point that is constrained on the given axis.

            Note this index must refer to the points, not the feature indices.
            For example, ``axis_point_idx = 1`` will force on ``axis`` the point
            whose coordinates correspond to feature indices ``[3, 4, 5]``.
        plane_point_idx : int, optional
            The index of the point that is forced on the given plane.

            Note this index must refer to the points, not the feature indices.
            For example, ``plane_point_idx = 1`` will force on ``plane`` the
            point whose coordinates correspond to feature indices ``[3, 4, 5]``.
        axis : Literal['x', 'y', 'z'], optional
            The axis on which the position of ``axis_point_idx`` is forced.
        plane : Literal['xy', 'yz', 'xz'], optional
            The plane on which the position of ``plane_point_idx`` is forced.
        round_off_imprecisions : bool, optional
            As a result of the constrains, several coordinates should be exactly
            0.0, but numerical errors may cause these to deviate from it. Setting
            this to ``True`` truncate the least significant decimal values of
            the constrained degrees of freedom.
        rotate_back : bool, optional
            If ``False``, the output configuration has the centroid in the
            ``origin``. Otherwise, it the centroid is restored to the original
            position.
        return_partial : bool, optional
            If ``True``, only the propagated indices are returned.

        """
        if return_partial and rotate_back:
            raise ValueError("'return_partial=True' is supported only if 'rotate_back=False'")

        # Automatic selection of the points placed on the axis/plane.
        if axis_point_idx is None:
            if plane_point_idx != 0:
                axis_point_idx = 0
            else:
                axis_point_idx = 1
        if plane_point_idx is None:
            if axis_point_idx != 0:
                plane_point_idx = 0
            else:
                plane_point_idx = 1

        # Two different points must be used to define the reference frame.
        if axis_point_idx == plane_point_idx:
            raise ValueError("'axis_point_idx' and 'plane_point_idx' must be different.")

        if axis not in plane:
            raise ValueError("To constrain 'plane_atom_idx' to stay on plane {plane} "
                             "'axis_atom_idx' must be constrained on an axis on the same plane.")

        # Save the axis used for contraining the first point as a vector.
        self._axis = self._AXES[axis].type(torch.get_default_dtype())

        # Save the axis that together with self._axis defines the plane on which
        # the second point is contrained.
        self._plane_axis = [x.type(self._axis.dtype) for name, x in self._AXES.items()
                            if (name not in axis) and (name in plane)][0]

        # Save the plane used for constraining the second point as its normal vector.
        self._plane_normal = torch.cross(self._axis, self._plane_axis)

        # The coordinates that are not on the axis are fixed to 0.
        axis_point_flattened_indices = atom_to_flattened_indices(torch.tensor([axis_point_idx]))
        is_constrained_on_axis = self._axis == 0.0

        # The coordinate that are not on the plane is fixed to 0.
        plane_point_flattened_indices = atom_to_flattened_indices(torch.tensor([plane_point_idx]))
        is_constrained_on_plane = self._plane_normal != 0.0

        # Determine which atom is fixed.
        fixed_indices = torch.cat([axis_point_flattened_indices[is_constrained_on_axis],
                                   plane_point_flattened_indices[is_constrained_on_plane]])

        # Call PartialFlow constructor to fix the indices.
        super().__init__(flow, fixed_indices=fixed_indices, return_partial=return_partial)

        # Save all other parameters.
        self._axis_point_idx = axis_point_idx
        self._plane_point_idx = plane_point_idx
        self.round_off_imprecisions = round_off_imprecisions
        self.rotate_back = rotate_back  #: Whether the reference frame is restored to its original orientation in the output configuration.

    def forward(self, x):
        """Transform the input configuration."""
        return self._transform(x)

    def inverse(self, y):
        """Invert the forward transformation.

        This works only if the forward transformation was performed with
        ``rotate_back`` set to ``True``.

        """
        if not self.rotate_back:
            raise ValueError("The inverse of OrientedFlow can be computed only"
                             " if 'rotate_back' is set to True during both the"
                             " forward and inverse transformations.")
        return self._transform(y, inverse=True)

    def _transform(self, x, inverse=False):
        """Apply the forward/inverse transformation."""
        batch_size = x.shape[0]

        # Reshape coordinates to be in standard atom format.
        x = flattened_to_atom(x)

        # Find the direction perpendicular to the plane formed by the axis atom,
        # and the axis. torch.cross() requires tensors of same size but at least
        # torch.expand() does not allocate new memory.
        axis = self._axis.expand((batch_size, 3))

        # rotation_vectors has shape (batch_size, 3).
        rotation_vectors = torch.cross(x[:, self._axis_point_idx], axis, dim=1)

        # Find the first rotation angle. r1_angle has shape (batch_size,).
        r1_angles = vector_vector_angle(
            x[:, self._axis_point_idx], self._axis)

        # r1_angles goes from 0 to pi. We want to rotate the point onto the
        # negative/positive axis, depending which is closest.
        r1_angles = r1_angles - torch.pi * (r1_angles > torch.pi/2).float()

        # This are the rotation matrices that bring the axis points onto the axis.
        r1_rotation_matrices = rotation_matrix_3d(r1_angles, rotation_vectors)

        # To bring the plane atom in position, we perform a rotation about
        # self._axis so that we don't modify the position of the axis atom.
        # We perform the first rotation only on the atom position that will
        # determine the next rotation matrix for now so that we run only
        # a single matmul on all atoms.
        plane_points = x[:, self._plane_point_idx].unsqueeze(1)
        plane_points = batchwise_rotate(plane_points, r1_rotation_matrices)
        plane_points = plane_points.squeeze(1)

        # Project the atom on the plane perpendicular to the rotation axis plane
        # to measure the rotation angle.
        plane_points = plane_points - self._axis*batchwise_dot(plane_points, self._axis, keepdim=True)
        r2_angles = vector_plane_angle(plane_points, self._plane_normal)

        # r2_angles will be positive in the octants where self._plane_normal
        # lies and negative in the opposite direction but the rotation happens
        # counterclockwise/clockwise with positive/negative angle so we need
        # to fix the sign of the angle based on where it is.
        r2_angles_sign = -torch.sign(batchwise_dot(plane_points, self._plane_axis))
        r2_rotation_matrices = rotation_matrix_3d(
            r2_angles_sign * r2_angles, self._axis)

        # Now build the rotation composition before applying the transformation.
        rotation_matrices = torch.bmm(r2_rotation_matrices, r1_rotation_matrices)
        x = batchwise_rotate(x, rotation_matrices)

        # Re-shape back to flattened format.
        x = atom_to_flattened(x)

        # Now round off numerical imprecisions.
        if self.round_off_imprecisions:
            x[:, self._fixed_indices] = 0.0

        # Apply the transformation through the PartialFlow.
        if inverse:
            y, log_det_J = super().inverse(x)
        else:
            y, log_det_J = super().forward(x)

        # Check if we need only to return the partial result. PartialFlow takes
        # care of returning ony the propagated indices.
        if self.return_partial:
            return y, log_det_J

        # If we need to rotate back, the new reference frame must equal the original.
        if self.rotate_back:
            y = flattened_to_atom(y)
            y = batchwise_rotate(y, rotation_matrices, inverse=True)
            y = atom_to_flattened(y)

        return y, log_det_J
