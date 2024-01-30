#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test objects and function in tfep.utils.geometry.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import cmath

import MDAnalysis.lib
import numpy as np
import pytest
import torch

from tfep.utils.math import batch_autograd_log_abs_det_J
from tfep.utils.geometry import (
    pdist,
    vector_vector_angle,
    vector_plane_angle,
    proper_dihedral_angle,
    rotation_matrix_3d,
    batchwise_rotate,
    cartesian_to_polar,
    polar_to_cartesian,
)


# =============================================================================
# GLOBAL VARIABLES
# =============================================================================

# Makes random input deterministic.
_GENERATOR = torch.Generator()
_GENERATOR.manual_seed(0)


# =============================================================================
# TEST MODULE CONFIGURATION
# =============================================================================

_old_default_dtype = None

def setup_module(module):
    global _old_default_dtype
    _old_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.double)


def teardown_module(module):
    torch.set_default_dtype(_old_default_dtype)


# =============================================================================
# REFERENCE FUNCTIONS FOR TESTING
# =============================================================================

def reference_pdist(x, pairs=None):
    batch_size, n_atoms, dim = x.shape
    if pairs is None:
        pairs = torch.triu_indices(n_atoms, n_atoms, offset=1)
    n_pairs = pairs.shape[1]

    x, pairs = x.detach().numpy(), pairs.transpose(0, 1).detach().numpy()

    distances = np.empty((batch_size, n_pairs))
    diff = np.empty((batch_size, n_pairs, dim))
    for batch_idx in range(batch_size):
        for pair_idx, (i, j) in enumerate(pairs):
            p_i = x[batch_idx, i]
            p_j = x[batch_idx, j]
            diff[batch_idx, pair_idx] = p_j - p_i
            distances[batch_idx, pair_idx] = np.linalg.norm(diff[batch_idx, pair_idx])

    return torch.tensor(distances), torch.tensor(diff)


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


def reference_proper_dihedral_angle(v1, v2, v3):
    v1_np, v2_np, v3_np = v1.detach().numpy(), v2.detach().numpy(), v3.detach().numpy()
    dihedrals = [MDAnalysis.lib.mdamath.dihedral(v1_np[i], v2_np[i], v3_np[i]) for i in range(len(v1_np))]
    return torch.tensor(dihedrals, dtype=v1.dtype)


def reference_rotation_matrix_3d(angles, directions):
    angles_np = angles.detach().numpy()
    directions_np = directions.detach().numpy()
    rotation_matrices = [MDAnalysis.lib.transformations.rotation_matrix(a, d)[:3,:3]
                         for a, d in zip(angles_np, directions_np)]
    return torch.tensor(rotation_matrices, dtype=angles.dtype)


def reference_batchwise_rotate(x, rotation_matrices):
    x_np = x.detach().numpy()
    rotation_matrices_np = rotation_matrices.detach().numpy()
    y = np.empty_like(x_np)
    for i in range(len(x_np)):
        y[i] = x_np[i] @ rotation_matrices_np[i].T
    return torch.tensor(y, dtype=x.dtype)


def reference_cartesian_to_polar(x, y):
    x, y = x.tolist(), y.tolist()
    r, angle = [], []
    for xi, yi in zip(x, y):
        polar = cmath.polar(complex(xi, yi))
        r.append(polar[0])
        angle.append(polar[1])
    return torch.tensor(r), torch.tensor(angle)


# =============================================================================
# TESTS
# =============================================================================

@pytest.mark.parametrize('pairs', [None, torch.tensor([[0, 0, 2], [1, 2, 3]])])
def test_pdist(pairs):
    """Test the pairwise distance function pdist()."""
    batch_size = 2
    n_atoms = 4
    dimension = 3
    x = torch.randn((batch_size, n_atoms, dimension), generator=_GENERATOR)

    distances, diff = pdist(x, pairs=pairs, return_diff=True)
    ref_distances, ref_diff = reference_pdist(x, pairs=pairs)
    assert torch.allclose(distances, ref_distances)
    assert torch.allclose(diff, ref_diff)


def test_vector_vector_angle_axes():
    """Test the vector_vector_angle() function to measure angles between axes."""
    v1 = torch.eye(3)

    angles = vector_vector_angle(v1, v1[0])
    assert torch.allclose(angles, torch.tensor([0.0, np.pi/2, np.pi/2]))
    angles = vector_vector_angle(v1, v1[1])
    assert torch.allclose(angles, torch.tensor([np.pi/2, 0.0, np.pi/2]))
    angles = vector_vector_angle(v1, v1[2])
    assert torch.allclose(angles, torch.tensor([np.pi/2, np.pi/2, 0.0]))

    angles = vector_vector_angle(v1, v1)
    assert torch.allclose(angles, torch.zeros_like(angles))


@pytest.mark.parametrize('batch_size', [1, 3])
@pytest.mark.parametrize('dimension', [2, 3, 4])
def test_vector_vector_angle_against_reference(batch_size, dimension):
    """Test the vector_vector_angle() function on random tensors against a reference implementation."""
    # Build a random input.
    v1 = torch.randn((batch_size, dimension), generator=_GENERATOR)
    v2 = torch.randn(dimension, generator=_GENERATOR)

    # Compare reference and PyTorch implementation.
    angles = vector_vector_angle(v1, v2)
    angles2 = vector_vector_angle(v1, v2.unsqueeze(0).expand(len(v1), -1))
    ref_angles = reference_vector_vector_angle(v1, v2)
    assert torch.allclose(angles, angles2)
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


@pytest.mark.parametrize('batch_size', [1, 20])
def test_proper_dihedral_angle_against_reference(batch_size):
    """Test the proper_dihedral_angle() function on random tensors against a reference implementation."""
    # Cross product works only in 3D.
    dimension = 3

    # Build a random inputs.
    v1 = torch.randn((batch_size, dimension), generator=_GENERATOR)
    v2 = torch.randn((batch_size, dimension), generator=_GENERATOR)
    v3 = torch.randn((batch_size, dimension), generator=_GENERATOR)

    # Compare reference and PyTorch implementation. MDAnalysis adopts the
    # opposite sign convention for the angles.
    dihedrals = proper_dihedral_angle(v1, v2, v3)
    ref_dihedrals = reference_proper_dihedral_angle(v1, v2, v3)
    assert torch.allclose(dihedrals, -ref_dihedrals)


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


def test_batchwise_rotate_axes():
    """Test that batchwise_rotate transforms one axis into another."""
    axes = torch.eye(3, dtype=torch.double)
    x = torch.stack([axes, axes])

    # Two rotations by 90 degrees about the x and z axis respectively.
    rot = rotation_matrix_3d(
        angles=torch.tensor([np.pi/2, np.pi/2], dtype=torch.double),
        directions=torch.stack([axes[0], axes[2]])
    )

    y = batchwise_rotate(x, rot)
    expected = x.detach().clone()
    expected[0, 1], expected[0, 2] = expected[0, 2], -expected[0, 1]
    expected[1, 0], expected[1, 1] = expected[1, 1], -expected[1, 0]
    assert torch.allclose(y, expected)


@pytest.mark.parametrize('batch_size', [1, 10])
@pytest.mark.parametrize('n_vectors', [1, 5])
def test_batchwise_rotate_against_reference(batch_size, n_vectors):
    """Test the batchwise_rotate() function on random tensors against a reference implementation."""
    angles = -4*np.pi * torch.rand(batch_size, generator=_GENERATOR, dtype=torch.float) + 2*np.pi
    directions = torch.randn(batch_size, 3, generator=_GENERATOR, dtype=torch.float)
    rotation_matrices = rotation_matrix_3d(angles, directions)
    x = torch.randn(batch_size, n_vectors, 3)

    y = batchwise_rotate(x, rotation_matrices)
    ref_y = reference_batchwise_rotate(x, rotation_matrices)
    assert torch.allclose(y, ref_y)


@pytest.mark.parametrize('batch_size', [1, 10])
def test_cartesian_to_polar_conversion(batch_size):
    """Test cartesian_to_polar and polar_to_cartesian against reference."""
    # Create Cartesian input.
    input = torch.randn(batch_size, 2, requires_grad=True)
    x, y = input[:, 0], input[:, 1]

    # Convert to polar.
    r, angle, log_det_J = cartesian_to_polar(x, y, return_log_det_J=True)
    output = torch.cat([r.unsqueeze(0), angle.unsqueeze(0)], dim=0).T

    # First test against implementation based on standard python library.
    r_ref, angle_ref = reference_cartesian_to_polar(x, y)
    assert torch.allclose(r, r_ref)
    assert torch.allclose(angle, angle_ref)

    # Test inverse.
    x_inv, y_inv, log_det_J_inv = polar_to_cartesian(r, angle, return_log_det_J=True)
    assert torch.allclose(x, x_inv)
    assert torch.allclose(y, y_inv)
    assert torch.allclose(log_det_J + log_det_J_inv, torch.zeros_like(log_det_J))

    # Test log det J.
    log_det_J_ref = batch_autograd_log_abs_det_J(input, output)
    assert torch.allclose(log_det_J, log_det_J_ref)
