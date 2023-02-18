#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test classes and functions in tfep.nn.dynamics.egnn.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import numpy as np
import pytest
import scipy.spatial.transform
import torch

from tfep.nn.dynamics.egnn import EGNNDynamics
from tfep.utils.geometry import batchwise_rotate
from tfep.utils.misc import atom_to_flattened, flattened_to_atom


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
# TESTS EQUIVARIANT GRAPH NEURAL NETWORK DYNAMICS
# =============================================================================

@pytest.mark.parametrize('batch_size', [1, 2])
def test_egnn_dynamics_node_embedding(batch_size):
    """Test the embedding of the atom types and time into the node features."""
    particle_types = torch.tensor([0, 0, 1])  # Three particles.
    time_feat_dim = 2
    node_feat_dim = 4

    egnn_dynamics = EGNNDynamics(
        particle_types=particle_types,
        r_cutoff=6.0,
        time_feat_dim=time_feat_dim,
        node_feat_dim=node_feat_dim,
    )

    h = egnn_dynamics._create_node_embedding(t=torch.tensor(1.0), batch_size=batch_size)

    # The features result from the concatenation of the atom type and time expansions.
    assert h.shape == (batch_size*len(particle_types), node_feat_dim)

    # Identical atom types must have identical embeddings.
    if batch_size == 1:
        zero_type = [1]
        one_type = []
    else:
        zero_type = [1, 3, 4]
        one_type = [5]

    for i in zero_type:
        assert torch.allclose(h[0], h[i])
    for i in one_type:
        assert torch.allclose(h[2], h[i])
    assert not torch.allclose(h[0], h[2])


@pytest.mark.parametrize('batch_size', [1, 4])
@pytest.mark.parametrize('r_cutoff', [1.5, 10.0])
@pytest.mark.parametrize('seed', [0, 5, 10])
def test_egnn_dynamics_equivariance_property(batch_size, r_cutoff, seed):
    """Test that the output of the EGNN has the correct equivariance properties."""
    particle_types = torch.tensor([0, 0, 1])
    n_particles = len(particle_types)

    # Create the equivariant graph.
    egnn_dynamics = EGNNDynamics(
        particle_types=particle_types,
        r_cutoff=r_cutoff,
        time_feat_dim=2,
        node_feat_dim=5,
        distance_feat_dim=4,
        n_layers=2,
        initialize_identity=False,
    )

    # Create a random input.
    generator = torch.Generator()
    generator.manual_seed(seed)
    t = torch.rand(1, generator=generator)
    x = torch.randn(batch_size, n_particles*3, generator=generator)

    # Run the dynamics.
    vel = egnn_dynamics(t, x)

    # Now re-run the dynamics with a randomly rotated input.
    random_state = np.random.RandomState(seed)
    rotation_matrices = scipy.spatial.transform.Rotation.random(
        num=batch_size, random_state=random_state)
    rotation_matrices = torch.tensor(rotation_matrices.as_matrix())

    # Rotate the input.
    x_rotated = batchwise_rotate(flattened_to_atom(x), rotation_matrices)
    x_rotated = atom_to_flattened(x_rotated)

    # The velocity should rotate accordingly.
    vel_rotated = egnn_dynamics(t, x_rotated)
    ref_vel = atom_to_flattened(batchwise_rotate(flattened_to_atom(vel), rotation_matrices))
    assert torch.allclose(ref_vel, vel_rotated)

    # Translate the input with shape (batch_size, n_particles*3).
    translation = torch.randn(batch_size, 3)
    translation = translation.repeat(1, n_particles)
    x_translated = x + translation

    # The velocity should be left invariant.
    vel_translated = egnn_dynamics(t, x_translated)
    assert torch.allclose(vel, vel_translated)

    # Velocity is equivariant w.r.t. permuting two particles of the same type.
    def permute(_x, idx1, idx2):
        _x_permuted = _x.clone()
        _x_permuted[:,idx1*3:(idx1+1)*3] = _x[:,idx2*3:(idx2+1)*3]
        _x_permuted[:,idx2*3:(idx2+1)*3] = _x[:,idx1*3:(idx1+1)*3]
        return _x_permuted

    x_permuted = permute(x, 0, 1)
    vel_permuted = egnn_dynamics(t, x_permuted)
    assert torch.allclose(permute(vel, 0, 1), vel_permuted)

    # Permuting two particles with different type changes the velocity.
    x_permuted = permute(x, 0, 2)
    vel_permuted = egnn_dynamics(t, x_permuted)
    assert not torch.allclose(permute(vel, 0, 2), vel_permuted)


def test_egnn_dynamics_identity():
    """Test that EGNNDynamics can be initialized to output zero velocity."""
    batch_size = 10
    particle_types = torch.tensor([0, 0, 1])
    n_particles = len(particle_types)

    # Create the equivariant graph.
    egnn_dynamics = EGNNDynamics(
        particle_types=particle_types,
        r_cutoff=10.0,
        time_feat_dim=2,
        node_feat_dim=5,
        distance_feat_dim=4,
        n_layers=2,
        initialize_identity=True,
    )

    # Create a random input.
    generator = torch.Generator()
    generator.manual_seed(0)
    t = torch.rand(1, generator=generator)
    x = torch.randn(batch_size, n_particles*3, generator=generator)

    # Run the dynamics.
    vel = egnn_dynamics(t, x)

    # The output is identically 0.0.
    assert torch.allclose(vel, torch.zeros_like(vel))
