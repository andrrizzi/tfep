"""Small runtime patches used by the ozone demo."""

from __future__ import annotations

from typing import Optional

import torch


def patch_tfep_geometry_device_bug() -> None:
    """Patch a CUDA device mismatch in ``tfep.utils.geometry.reference_frame_rotation_matrix``.

    Some versions create CPU tensors inside the function, which breaks when the
    flow is executed on CUDA. This patch swaps the implementation with a
    device-safe variant.

    It's safe to call multiple times.
    """
    import tfep.utils.geometry as geom

    if getattr(geom.reference_frame_rotation_matrix, "_patched_for_device", False):
        return

    def patched_reference_frame_rotation_matrix(
        axis_atom_positions: torch.Tensor,
        plane_atom_positions: torch.Tensor,
        axis: torch.Tensor,
        plane_axis: torch.Tensor,
        plane_normal: Optional[torch.Tensor] = None,
        project_on_positive_axis: bool = False,
    ) -> torch.Tensor:
        if plane_normal is None:
            plane_normal = torch.cross(axis, plane_axis, dim=0)

        rotation_vectors = torch.cross(axis_atom_positions, axis.unsqueeze(0), dim=1)

        z = torch.zeros(1, device=rotation_vectors.device, dtype=rotation_vectors.dtype)
        is_parallel = torch.isclose(rotation_vectors, z).all(dim=1)
        rotation_vectors[is_parallel] = torch.cross(plane_axis, axis, dim=0)

        r1_angles = geom.vector_vector_angle(axis_atom_positions, axis)

        if not project_on_positive_axis:
            r1_angles = r1_angles - torch.pi * (r1_angles > torch.pi / 2).to(r1_angles.dtype)

        r1_rotation_matrices = geom.rotation_matrix_3d(r1_angles, rotation_vectors)

        plane_points = plane_atom_positions.unsqueeze(1)
        plane_points = geom.batchwise_rotate(plane_points, r1_rotation_matrices)
        plane_points = plane_points.squeeze(1)

        plane_points = plane_points - axis * geom.batchwise_dot(plane_points, axis, keepdim=True)
        r2_angles = geom.vector_plane_angle(plane_points, plane_normal)

        r2_angles_sign = -torch.sign(geom.batchwise_dot(plane_points, plane_axis))
        r2_rotation_matrices = geom.rotation_matrix_3d(r2_angles_sign * r2_angles, axis)

        return torch.bmm(r2_rotation_matrices, r1_rotation_matrices)

    patched_reference_frame_rotation_matrix._patched_for_device = True
    geom.reference_frame_rotation_matrix = patched_reference_frame_rotation_matrix
