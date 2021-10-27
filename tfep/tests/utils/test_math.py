#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test objects and function in tfep.utils.math.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import MDAnalysis.lib
import numpy as np
import pytest
import torch

from tfep.utils.math import cov, vector_vector_angle, vector_plane_angle, rotation_matrix_3d


# =============================================================================
# GLOBAL VARIABLES
# =============================================================================

# Makes random input deterministic.
_GENERATOR = torch.Generator()
_GENERATOR.manual_seed(0)


# =============================================================================
# REFERENCE FUNCTIONS FOR TESTING
# =============================================================================

def reference_vector_vector_angle(v1, v2):
    v1_np, v2_np = v1.detach().numpy(), v2.detach().numpy()
    angles = [MDAnalysis.lib.mdamath.angle(v, v2_np) for v in v1_np]
    return torch.tensor(angles, dtype=v1.dtype)


def reference_vector_plane_angle(vectors, plane):
    vectors_np, plane_np = vectors.detach().numpy(), plane.detach().numpy()
    angles = []
    for v in vectors_np:
        x = np.dot(v, plane) / (np.linalg.norm(v) * np.linalg.norm(plane))
        # Catch roundoffs that lead to nan otherwise.
        if x > 1.0:
            return np.pi/2
        elif x < -1.0:
            return -np.pi/2
        angles.append(np.arcsin(x))
    return torch.tensor(angles, dtype=plane.dtype)


def reference_rotation_matrix_3d(angles, directions):
    angles_np = angles.detach().numpy()
    directions_np = directions.detach().numpy()
    rotation_matrices = [MDAnalysis.lib.transformations.rotation_matrix(a, d)[:3,:3]
                         for a, d in zip(angles_np, directions_np)]
    return torch.tensor(rotation_matrices, dtype=angles.dtype)


# =============================================================================
# TESTS
# =============================================================================

@pytest.mark.parametrize('ddof', [0, 1])
@pytest.mark.parametrize('dim_n', [0, 1])
def test_cov(ddof, dim_n):
    """Test the covariance matrix against the numpy implementation."""
    random_state = np.random.RandomState(0)
    x = random_state.randn(10, 15)

    if dim_n == 0:
        cov_np = np.cov(x.T, ddof=ddof)
    else:
        cov_np = np.cov(x, ddof=ddof)

    cov_torch = cov(torch.tensor(x), dim_n=dim_n, ddof=ddof, inplace=True).numpy()

    assert np.allclose(cov_np, cov_torch)


def test_vector_vector_angle_axes():
    """Test the vector_vector_angle() function to measure angles between axes."""
    v1 = torch.eye(3)
    angles = vector_vector_angle(v1, v1[0])
    assert torch.allclose(angles, torch.tensor([0.0, np.pi/2, np.pi/2]))
    angles = vector_vector_angle(v1, v1[1])
    assert torch.allclose(angles, torch.tensor([np.pi/2, 0.0, np.pi/2]))
    angles = vector_vector_angle(v1, v1[2])
    assert torch.allclose(angles, torch.tensor([np.pi/2, np.pi/2, 0.0]))


@pytest.mark.parametrize('batch_size', [1, 3])
@pytest.mark.parametrize('dimension', [2, 3, 4])
def test_vector_vector_angle_against_reference(batch_size, dimension):
    """Test the vector_vector_angle() function on random tensors against a reference implementation."""
    # Build a random input.
    v1 = torch.randn((batch_size, dimension), generator=_GENERATOR)
    v2 = torch.randn(dimension, generator=_GENERATOR)

    # Compare reference and PyTorch implementation.
    angles = vector_vector_angle(v1, v2)
    ref_angles = reference_vector_vector_angle(v1, v2)
    assert torch.allclose(angles, ref_angles)


def test_vector_plane_angle_axes():
    """Test the vector_plane_angle() function to measure angles between axes planes."""
    vectors = torch.eye(3)
    planes = torch.eye(3)
    angles = vector_plane_angle(vectors, planes[0])
    assert torch.allclose(angles, torch.tensor([np.pi/2, 0.0, 0.0]))
    angles = vector_plane_angle(vectors, planes[1])
    assert torch.allclose(angles, torch.tensor([0.0, np.pi/2, 0.0]))
    angles = vector_plane_angle(vectors, planes[2])
    assert torch.allclose(angles, torch.tensor([0.0, 0.0, np.pi/2]))


@pytest.mark.parametrize('batch_size', [1, 3])
@pytest.mark.parametrize('dimension', [2, 3, 4])
def test_vector_plane_angle_against_reference(batch_size, dimension):
    """Test the vector_plane_angle() function on random tensors against a reference implementation."""
    # Build a random input.
    vectors = torch.randn((batch_size, dimension), generator=_GENERATOR)
    plane = torch.randn(dimension, generator=_GENERATOR)

    # Compare reference and PyTorch implementation.
    angles = vector_plane_angle(vectors, plane)
    ref_angles = reference_vector_plane_angle(vectors, plane)
    assert torch.allclose(angles, ref_angles)


def test_rotation_matrix_axes():
    """Test rotation_matrix reproduce standard rotation matrixes around axes."""
    c = 1 / np.sqrt(2.0)
    angles = torch.tensor([np.pi/4, np.pi/4, np.pi/4], dtype=torch.double)
    axes = torch.eye(3, dtype=torch.double)

    rot_x = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, c, -c],
        [0.0, c, c],
    ])
    rot_y = torch.tensor([
        [c, 0.0, c],
        [0.0, 1.0, 0.0],
        [-c, 0.0, c],
    ])
    rot_z = torch.tensor([
        [c, -c, 0.0],
        [c, c, 0.0],
        [0.0, 0.0, 1.0],
    ])

    rotation_matrices = rotation_matrix_3d(angles, axes)
    assert torch.allclose(rotation_matrices[0], rot_x)
    assert torch.allclose(rotation_matrices[1], rot_y)
    assert torch.allclose(rotation_matrices[2], rot_z)


@pytest.mark.parametrize('batch_size', [1, 10])
def test_rotation_matrix_against_reference(batch_size):
    """Test the rotation_matrix() function on random tensors against a reference implementation."""
    angles = -4*np.pi * torch.rand(batch_size, generator=_GENERATOR) + 2*np.pi
    directions = torch.randn(batch_size, 3, generator=_GENERATOR)

    # Compare reference and PyTorch implementation.
    rotation_matrices = rotation_matrix_3d(angles, directions)
    ref_rotation_matrices = reference_rotation_matrix_3d(angles, directions)
    assert torch.allclose(rotation_matrices, ref_rotation_matrices)
