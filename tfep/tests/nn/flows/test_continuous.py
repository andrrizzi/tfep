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
    _trace_and_frobenious_squared_norm_hutchinson,
    _trace_and_frobenious_squared_norm_exact,
)


# =============================================================================
# TEST MODULE CONFIGURATION
# =============================================================================

# torchdiffeq is an optional dependency of tfep.
try:
    import torchdiffeq
except ImportError:
    TORCHDIFFEQ_INSTALLED = False
else:
    TORCHDIFFEQ_INSTALLED  = True


_old_default_dtype = None
_old_torch_seed = None

def setup_module(module):
    global _old_default_dtype, _old_torch_seed
    _old_default_dtype = torch.get_default_dtype()
    _old_torch_seed = torch.initial_seed()
    torch.set_default_dtype(torch.double)

    # Setting the seed is necessary to make the neural network parameters reproducible.
    torch.manual_seed(2)


def teardown_module(module):
    global _old_default_dtype, _old_torch_seed
    torch.set_default_dtype(_old_default_dtype)
    torch.manual_seed(_old_torch_seed)


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
            torch.nn.Linear(n_features, n_features, bias=False),
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

# TODO: test both vmap=True and False when pytorch 1.11 is released.

@pytest.mark.parametrize('batch_size', [1, 10])
@pytest.mark.parametrize('create_graph', [True, False])
@pytest.mark.parametrize('seed', [0, 3, 6])
def test_exact_trace_estimator(batch_size, create_graph, seed):
    """Test the exact estimator of the trace and Frobenious norm of the Jacobian."""
    vmap = False
    t, x, dynamics, _ = create_input_and_dynamics(batch_size, n_atoms=3, seed=seed)

    # Compute the reference trace and regularization terms.
    ref_trace = torch_jacobian_trace(dynamics, t, x)
    ref_reg = torch_jacobian_frobenious_squared_norm(dynamics, t, x)

    # Grad outputs.
    grads_out = torch.eye(x.shape[1])

    # Compute the estimated trace.
    vel = dynamics(t, x)
    trace = _trace_exact(vel, x, grads_out, vmap=vmap, create_graph=create_graph)
    trace_with_frobenious, reg = _trace_and_frobenious_squared_norm_exact(
        vel, x, grads_out, vmap=vmap, create_graph=create_graph)

    assert torch.allclose(trace, ref_trace)
    assert torch.allclose(trace_with_frobenious, ref_trace)
    assert torch.allclose(reg, ref_reg)


@pytest.mark.parametrize('batch_size', [1, 10])
@pytest.mark.parametrize('create_graph', [True, False])
@pytest.mark.parametrize('seed', [0, 3, 6])
def test_hutchinson_trace_estimator(batch_size, create_graph, seed):
    """Test the implementation of Hutchinson's trace estimator."""
    tol = 1e-2
    vmap = False
    t, x, dynamics, generator = create_input_and_dynamics(batch_size, n_atoms=3, seed=seed)

    # Compute the reference trace and regularization terms.
    ref_trace = torch_jacobian_trace(dynamics, t, x)
    ref_reg = torch_jacobian_frobenious_squared_norm(dynamics, t, x)

    # Compute the estimated trace with many samples.
    vel = dynamics(t, x)
    n_samples = 2500

    # First compute the traces using n_sample repetitions of the function.
    traces = torch.empty(n_samples, batch_size)
    traces_with_frobenious = torch.empty(n_samples, batch_size)
    regs = torch.empty(n_samples, batch_size)

    for i in range(n_samples):
        eps = torch.randn(1, *x.shape, generator=generator)
        traces[i] = _trace_hutchinson(vel, x, eps, vmap=vmap, create_graph=create_graph)
        traces_with_frobenious[i], regs[i] = _trace_and_frobenious_squared_norm_hutchinson(
            vel, x, eps, vmap=vmap, create_graph=create_graph)

    trace = traces.mean(dim=0)
    trace_with_frobenious = traces_with_frobenious.mean(dim=0)
    reg = regs.mean(dim=0)

    assert torch.allclose(trace, ref_trace, atol=tol)
    assert torch.allclose(trace_with_frobenious, ref_trace, atol=tol)
    assert torch.allclose(reg, ref_reg, atol=tol)

    # Now compute it in a single pass.
    eps = torch.randn(n_samples, *x.shape, generator=generator)
    trace = _trace_hutchinson(vel, x, eps, vmap=vmap, create_graph=create_graph)
    trace_with_frobenious, reg = _trace_and_frobenious_squared_norm_hutchinson(
        vel, x, eps, vmap=vmap, create_graph=create_graph)

    assert torch.allclose(trace, ref_trace, atol=tol)
    assert torch.allclose(trace_with_frobenious, ref_trace, atol=tol)
    assert torch.allclose(reg, ref_reg, atol=tol)


@pytest.mark.skipif(not TORCHDIFFEQ_INSTALLED, reason='requires torchdiffeq to be installed')
@pytest.mark.parametrize('vmap', [True, False])
def test_identity_flow(vmap):
    """Check that the continuous flow behaves well with an identity dynamics."""
    _, x, dynamics, generator = create_input_and_dynamics(batch_size=5, n_atoms=3, seed=0)

    # Fix the weights of the last layer so that the flow is the identity.
    dynamics.mlp[-2].weight.data.fill_(0.0)

    # Create normalizing flow.
    flow = ContinuousFlow(dynamics=dynamics, trace_estimator='exact', vmap=vmap, requires_backward=True)

    # Check that the output is not transformed.
    y, trace, reg = flow(x)
    assert torch.allclose(x, y)
    assert torch.allclose(trace, torch.zeros_like(trace))

    # Verify that backpropagation is stable.
    loss = torch.sum(y)
    loss.backward()
    assert torch.allclose(x.grad, torch.ones_like(x.grad))


@pytest.mark.skipif(not TORCHDIFFEQ_INSTALLED, reason='requires torchdiffeq to be installed')
@pytest.mark.parametrize('batch_size', [1, 10])
@pytest.mark.parametrize('vmap', [True, False])
@pytest.mark.parametrize('create_graph', [True, False])
@pytest.mark.parametrize('seed', [1, 4, 8])
def test_continuous_flow_round_trip(batch_size, vmap, create_graph, seed):
    """Test that the ContinuousFlow.inverse(ContinuousFlow.forward(x)) equals the identity."""
    _, x, dynamics, generator = create_input_and_dynamics(batch_size, n_atoms=3, seed=seed)

    # Create normalizing flow.
    flow = ContinuousFlow(dynamics=dynamics, trace_estimator='exact', vmap=vmap, requires_backward=create_graph)

    # Round trip.
    y, trace, reg = flow(x)
    x_inv, trace_inv, reg_inv = flow.inverse(y)

    # Inverting the transformation produces the input vector.
    assert torch.allclose(x, x_inv, atol=1e-3)
    assert torch.allclose(trace + trace_inv, torch.zeros(batch_size), atol=1e-3)
