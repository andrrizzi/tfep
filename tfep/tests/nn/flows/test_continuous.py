#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test the continuous normalizing flow layer in tfep.nn.flows.continuous.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import pytest
import torch

from tfep.nn.flows.continuous import (
    ContinuousFlow,
    _trace_exact,
    _trace_hutchinson,
    _frobenious_squared_norm_exact,
    _frobenious_squared_norm_hutchinson,
    _trace_and_frobenious_squared_norm_hutchinson,
    _trace_and_frobenious_squared_norm_exact,
)


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
# TEST UTILITIES
# =============================================================================

def torch_jacobian_trace(f, t, x):
    batch_size, n_features = x.shape
    trace = torch.empty(batch_size)
    for batch_idx in range(x.shape[0]):
        jacobian = torch.autograd.functional.jacobian(f, (t, x[batch_idx]))
        trace[batch_idx] = torch.trace(jacobian[1])
    return trace


def torch_jacobian_frobenious_squared_norm(f, t, x):
    batch_size, n_features = x.shape
    norm = torch.empty(batch_size)
    for batch_idx in range(x.shape[0]):
        jacobian = torch.autograd.functional.jacobian(f, (t, x[batch_idx]))
        norm[batch_idx] = torch.linalg.matrix_norm(jacobian[1])**2
    return norm


class DynamicsMLP(torch.nn.Module):
    """A multilayer perceptron mapping time and positions to velocities."""

    def __init__(self, x):
        super().__init__()
        batch_size, n_features = x.shape
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(n_features+1, n_features),
            torch.nn.SiLU(),
            torch.nn.Linear(n_features, n_features),
            torch.nn.SiLU(),
            torch.nn.Linear(n_features, n_features),
            torch.nn.Tanh(),
        )

    def forward(self, t, x):
        # Make sure there is a batch dimension for t.
        if len(x.shape) > 1:
            t = t.unsqueeze(0).expand(x.shape[0], -1)
        input = torch.cat([t, x], dim=-1)
        return self.mlp(input)


def create_input_and_dynamics(batch_size, n_atoms, seed):
    # Create random input.
    generator = torch.Generator()
    generator.manual_seed(seed)
    x = torch.randn(batch_size, n_atoms*3, requires_grad=True, generator=generator)
    t = torch.rand(1, generator=generator)

    # Create a dynamics.
    dynamics = DynamicsMLP(x)

    return t, x, dynamics, generator


# =============================================================================
# TESTS
# =============================================================================

@pytest.mark.parametrize('batch_size', [1, 10])
@pytest.mark.parametrize('seed', [0, 3, 6])
def test_exact_trace_estimator(batch_size, seed):
    """Test the exact trace estimator."""
    t, x, dynamics, _ = create_input_and_dynamics(batch_size, n_atoms=3, seed=seed)

    # Compute the reference trace.
    ref_trace = torch_jacobian_trace(dynamics, t, x)

    # Compute the estimated trace.
    vel = dynamics(t, x)
    trace = _trace_exact(vel, x)
    trace_with_frobenious = _trace_and_frobenious_squared_norm_exact(vel, x)[0]

    assert torch.allclose(trace, ref_trace)
    assert torch.allclose(trace_with_frobenious, ref_trace)


@pytest.mark.parametrize('batch_size', [1, 10])
@pytest.mark.parametrize('seed', [0, 3, 6])
def test_hutchinson_trace_estimator(batch_size, seed):
    """Test the implementation of Hutchinson's trace estimator."""
    t, x, dynamics, generator = create_input_and_dynamics(batch_size, n_atoms=3, seed=seed)

    # Compute the reference trace.
    ref_trace = torch_jacobian_trace(dynamics, t, x)

    # Compute the estimated trace with many samples.
    vel = dynamics(t, x)
    n_samples = 2000

    traces = torch.empty(n_samples, batch_size)
    traces_with_frobenious = torch.empty(n_samples, batch_size)

    for i in range(n_samples):
        eps = torch.randn(x.shape, generator=generator)
        traces[i] = _trace_hutchinson(vel, x, eps)
        traces_with_frobenious[i] = _trace_and_frobenious_squared_norm_hutchinson(vel, x, eps)[0]

    trace = torch.mean(traces, dim=0)
    trace_with_frobenious = torch.mean(traces_with_frobenious, dim=0)

    assert torch.allclose(trace, ref_trace, atol=1e-2)
    assert torch.allclose(trace_with_frobenious, ref_trace, atol=1e-2)


@pytest.mark.parametrize('batch_size', [1, 10])
@pytest.mark.parametrize('seed', [0, 3, 6])
def test_frobenious_norm_exact(batch_size, seed):
    """Test the Hutchinson's estimation of the Jacobian Frobenious norm."""
    t, x, dynamics, generator = create_input_and_dynamics(batch_size, n_atoms=3, seed=seed)

    # Compute the reference regularization term.
    ref_reg = torch_jacobian_frobenious_squared_norm(dynamics, t, x)

    # Compute the estimated Frobenious norm.
    vel = dynamics(t, x)
    reg = _frobenious_squared_norm_exact(vel, x)
    reg_with_trace = _trace_and_frobenious_squared_norm_exact(vel, x)[1]

    assert torch.allclose(reg, ref_reg)
    assert torch.allclose(reg_with_trace, ref_reg)


@pytest.mark.parametrize('batch_size', [1, 10])
@pytest.mark.parametrize('seed', [0, 3, 6])
def test_frobenious_norm_hutchinson(batch_size, seed):
    """Test the Hutchinson's estimation of the Jacobian Frobenious norm."""
    t, x, dynamics, generator = create_input_and_dynamics(batch_size, n_atoms=3, seed=seed)

    # Compute the reference regularization term.
    ref_reg = torch_jacobian_frobenious_squared_norm(dynamics, t, x)

    # Compute the estimated Frobenious norm.
    vel = dynamics(t, x)
    n_samples = 2000

    regs = torch.empty(n_samples, batch_size)
    regs_with_trace = torch.empty(n_samples, batch_size)

    for i in range(n_samples):
        eps = torch.randn(x.shape, generator=generator)
        regs[i] = _frobenious_squared_norm_hutchinson(vel, x, eps)
        regs_with_trace[i] = _trace_and_frobenious_squared_norm_hutchinson(vel, x, eps)[1]

    reg = torch.mean(regs, dim=0)
    reg_with_trace = torch.mean(regs_with_trace, dim=0)

    assert torch.allclose(reg, ref_reg, atol=1e-2)
    assert torch.allclose(reg_with_trace, ref_reg, atol=1e-2)


@pytest.mark.parametrize('batch_size', [1, 10])
@pytest.mark.parametrize('seed', [1, 4, 8])
def test_continuous_flow_round_trip(batch_size, seed):
    """Test that the ContinuousFlow.inverse(ContinuousFlow.forward(x)) equals the identity."""
    _, x, dynamics, generator = create_input_and_dynamics(batch_size, n_atoms=3, seed=seed)

    # Create normalizing flow.
    flow = ContinuousFlow(dynamics=dynamics, trace_estimator='exact')

    # Round trip.
    y, trace, reg = flow(x)
    x_inv, trace_inv, reg_inv = flow.inverse(y)

    # Inverting the transformation produces the input vector.
    assert torch.allclose(x, x_inv, atol=1e-3)
    assert torch.allclose(trace + trace_inv, torch.zeros(batch_size), atol=1e-3)
