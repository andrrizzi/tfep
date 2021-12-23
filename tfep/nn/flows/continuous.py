#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Continuous normalizing flow layer for PyTorch.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import torch
from torchdiffeq import odeint_adjoint as odeint


# =============================================================================
# CONTINUOUS FLOW
# =============================================================================

class ContinuousFlow(torch.nn.Module):
    """Continuous normalizing flow.

    This implements continuous normalizing flows as proposed in [1]. The trace
    can be estimated using Hutchinson's stochastic estimator [2] at the cost of
    one backpropagation or exactly using D backpropagations, where D is the
    dimension of each sample.

    Optionally, the flow can return also a regularization term as proposed in
    [3] that can be incorporated into the loss to keep the ODE dynamics used
    for the flow smoother.

    Parameters
    ----------

    References
    ----------
    [1] Chen RT, Rubanova Y, Bettencourt J, Duvenaud D. Neural ordinary differential
        equations. arXiv preprint arXiv:1806.07366. 2018 Jun 19.
    [2] Grathwohl W, Chen RT, Bettencourt J, Sutskever I, Duvenaud D. Ffjord:
        Free-form continuous dynamics for scalable reversible generative models.
        arXiv preprint arXiv:1810.01367. 2018 Oct 2.
    [3] Finlay C, Jacobsen JH, Nurbekyan L, Oberman A. How to train your neural
        ODE: the world of Jacobian and kinetic regularization. InInternational
        Conference on Machine Learning 2020 Nov 21 (pp. 3154-3164). PMLR.

    """

    def __init__(
            self,
            dynamics,
            trace_estimator='hutchinson-gaussian',
            regularization=True,
    ):
        super().__init__()
        self.ode_func = _ODEFunc(dynamics, trace_estimator)
        self.regularization = regularization

    def forward(self, x):
        """Map the input data.

        Parameters
        ----------
        x : torch.Tensor
            An input batch of data of shape ``(batch_size, dimension_in)``.

        Returns
        -------
        y : torch.Tensor
            The mapped data of shape ``(batch_size, dimension_in)``.
        trace : torch.Tensor
            The instantaneous log absolute value of the Jacobian of the flow
            (equal to the trace of the jacobian) as a tensor of shape ``(batch_size,)``.
        reg : torch.Tensor, optional
            A regularization term to include in the loss. This is returned only
            if ``self.regularization`` is ``True``.

        """
        return self._pass(x, inverse=False)

    def inverse(self, y):
        return self._pass(y, inverse=True)

    def _pass(self, x, inverse):
        # Determine integration extremes.
        if inverse:
            t = [1.0, 0.0]
        else:
            t = [0.0, 1.0]
        t = torch.tensor(t, dtype=x.dtype)

        # Initialize initial trace and regularization that must be integrated..
        trace = x.new_zeros(x.shape[0])
        reg = x.new_zeros(x.shape[0])

        # Prepare function for a new integration.
        self.ode_func.before_odeint(x)

        # Check if we need to compute also the regularization term.
        if self.regularization:
            state_t0 = (x, trace, reg)
        else:
            state_t0 = (x, trace)

        # Integrate.
        state_traj = odeint(
            func=self.ode_func, y0=state_t0,
            t=t, method='dopri5', rtol=1e-4, atol=1e-4,
        )

        # Return the last value of the trajectory obtained through integration.
        state_t1 = [v[-1] for v in state_traj]

        # If this is the inverse, we need to invert the sign to the trace since
        # we started the integration from 0.0.
        state_t1[1] = -state_t1[1]
        return state_t1


# =============================================================================
# HELPER CLASSES AND FUNCTIONS
# =============================================================================

class _ODEFunc(torch.nn.Module):
    """Wraps the dynamics and profide a function for odeint()."""

    _HUTCH_ESTIMATOR_NAME = 'hutchinson-gaussian'
    _EXACT_ESTIMATOR_NAME = 'exact'

    def __init__(self, dynamics, trace_estimator):
        super().__init__()
        self.dynamics = dynamics
        self.trace_estimator = trace_estimator

        # This holds the random sample used for Gaussian estimation.
        # It will be initialized lazily in before_odeint().
        self._eps = None

    @property
    def trace_estimator(self):
        return self._trace_estimator

    @trace_estimator.setter
    def trace_estimator(self, new_trace_estimator):
        supported = {self._HUTCH_ESTIMATOR_NAME, self._EXACT_ESTIMATOR_NAME}
        if new_trace_estimator not in supported:
            raise ValueError('trace_estimator must be one of {}'.format(supported))
        self._trace_estimator = new_trace_estimator

    def before_odeint(self, x):
        """Prepares the function for a new integration.

        This regenerates the random sample used for Hutchinson's trace/Frobenious norm estimator.
        """
        if self.trace_estimator == self._HUTCH_ESTIMATOR_NAME:
            self._eps = torch.randn_like(x)

    def forward(self, t, state):
        # Check if we need regularization and unpack current state.
        try:
            x, trace, reg = state
        except ValueError:
            x, trace = state
            regularization = False
        else:
            regularization = True

        with torch.enable_grad():
            t.requires_grad = True
            # During the backwards pass, we might try to set this on a non-leaf
            # variable, which is forbidden, but x might already be set correctly.
            try:
                x.requires_grad = True
            except RuntimeError:
                if x.requires_grad is not True:
                    raise

            # Compute the velocity.
            vel = self.dynamics(t, x)

            # Compute regularization terms and/or estimate the Jacobian trace.
            if regularization:
                # Compute the squared L2-norm of the velocity used for regularization.
                vel_squared_norm = _batch_squared_norm(vel)

                # Compute the Frobenious norm of the divergence used for regularization.
                if self.trace_estimator == self._EXACT_ESTIMATOR_NAME:
                    trace, jac_norm = _trace_and_frobenious_squared_norm_exact(vel, x)
                else:
                    trace, jac_norm = _trace_and_frobenious_squared_norm_hutchinson(vel, x, self._eps)

                # Pack the value of the integrands.
                regularization_term = vel_squared_norm + jac_norm
                integrands = (vel, trace, regularization_term)
            else:
                if self.trace_estimator == self._EXACT_ESTIMATOR_NAME:
                    trace = _trace_exact(vel, x)
                else:
                    trace = _trace_hutchinson(vel, x, self._eps)

                # Pack the value of the integrands.
                integrands = (vel, trace)

        return integrands


def _batch_squared_norm(x):
    return torch.sum(x**2, dim=1)


def _trace_exact(f, x):
    """Compute the exact trace of the Jacobian df/dx using autograd."""
    trace = 0.0
    for idx in range(x.shape[1]):
        trace += torch.autograd.grad(f[:, idx].sum(), x, create_graph=True)[0][:, idx]
    return trace


def _trace_hutchinson(f, x, e, e_dfdx=None):
    """Compute the trace of the Jacobian df/dx using Hutchinson's estimator as in [2]."""
    if e_dfdx is None:
        e_dfdx = torch.autograd.grad(f, x, e, create_graph=True)[0]
    e_dfdx_e = e_dfdx * e
    trace = torch.sum(e_dfdx_e, dim=1)
    return trace


def _frobenious_squared_norm_exact(f, x):
    """Compute the exact Frobenious norm of the Jacobian df/dx."""
    norm = 0.0
    for idx in range(x.shape[1]):
        grad = torch.autograd.grad(f[:, idx].sum(), x, create_graph=True)[0]
        norm += torch.sum(grad**2, dim=-1)
    return norm


def _frobenious_squared_norm_hutchinson(f, x, e, e_dfdx=None):
    """Compute the Frobenious norm of the Jacobian df/dx using the Hutchinson estimate as in [3]."""
    if e_dfdx is None:
        e_dfdx = torch.autograd.grad(f, x, e, create_graph=True)[0]
    return _batch_squared_norm(e_dfdx)


def _trace_and_frobenious_squared_norm_exact(f, x):
    """Compute the exact trace and Frobenious norm of the Jacobian df/dx using autograd in D passes."""
    trace = 0.0
    norm = 0.0
    for idx in range(x.shape[1]):
        grad = torch.autograd.grad(f[:, idx].sum(), x, create_graph=True)[0]
        trace += grad[:, idx]
        norm += torch.sum(grad**2, dim=-1)
    return trace, norm


def _trace_and_frobenious_squared_norm_hutchinson(f, x, e):
    """Compute both trace and Frobenious norm of the Jacobian df/dx in a single pass using Hutchinson estimate."""
    e_dfdx = torch.autograd.grad(f, x, e, create_graph=True)[0]
    trace = _trace_hutchinson(f, x, e, e_dfdx)
    frobenius = _frobenious_squared_norm_hutchinson(f, x, e, e_dfdx)
    return trace, frobenius
