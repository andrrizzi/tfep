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

import enum

import torch

from tfep.utils.math import batchwise_dot


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
    dynamics : torch.nn.Module
        The neural network taking a time tensor (shape ``(1)``) and the current
        positions (shape ``(batch_size, n_particles*3)``) and returning the
        velocity of the dynamics (shape ``(batch_size, n_particles*3)``).
    trace_estimator : 'exact' or 'hutchinson', optional
        Whether the trace (and the Frobenious norm if ``regularization`` is
        ``True``) of the Jacobian is computed exactly with ``n_particles*3``
        backpropagation passes or using the hutchinson estimates described
        in [3] using ``n_hutchinson_samples`` backpropagation passes. The
        random variable is sampled from a normal distribution.
    solver : str, optional
        One of the solvers supported by the ``torchdiffeq`` package.
    solver_options : dict, optional
        A dictionary of solver options to pass to ``torchdiffeq.odeint``.
    n_hutchinson_samples : int, optional
        The number of normally-distributed sampled to be drawn for the Hutchinson
        estimate of the trace. If ``trace_estimator == 'exact'`` this is ignored.
    adjoint : bool, optional
        If ``True`` the backpropagation is performed using the adjoint method
        as described in [1]. Otherwise, automatic differentiation is used.
    regularization : bool, optional
        If ``True``, ``forward()`` returns also a regularization term, which
        is the sum of the velocity norm and the Frobenious norm of the Jacobian
        as described in [3].
    vmap : bool, optional
        If ``True``, the estimato of the trace and Frobenious norm are performed
        using the experimental vectorization features of ``torch.autograd.grad``
        (which are currently only in the unreleased development version).
    requires_backward : bool, optional
        If ``False``, the ``autograd`` calls used to compute the trace and
        regularization terms will not create a graph for differentiation. This
        means that backpropagation (even with the adjoint method) will not take
        into account the contribution from these terms.

    References
    ----------
    [1] Chen RT, Rubanova Y, Bettencourt J, Duvenaud D. Neural ordinary differential
        equations. arXiv preprint arXiv:1806.07366. 2018 Jun 19.
    [2] Grathwohl W, Chen RT, Bettencourt J, Sutskever I, Duvenaud D. Ffjord:
        Free-form continuous dynamics for scalable reversible generative models.
        arXiv preprint arXiv:1810.01367. 2018 Oct 2.
    [3] Finlay C, Jacobsen JH, Nurbekyan L, Oberman A. How to train your neural
        ODE: the world of Jacobian and kinetic regularization. In International
        Conference on Machine Learning 2020 Nov 21 (pp. 3154-3164). PMLR.

    """

    def __init__(
            self,
            dynamics,
            trace_estimator='hutchinson',
            solver='dopri5',
            solver_options=None,
            n_hutchinson_samples=1,
            adjoint=True,
            regularization=True,
            vmap=False,
            requires_backward=True,
    ):
        super().__init__()
        self.ode_func = _ODEFunc(dynamics, trace_estimator, n_hutchinson_samples, vmap, requires_backward)
        self.solver = solver
        self.solver_options = solver_options
        self.adjoint = adjoint
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
            A regularization term of shape ``(batch_size,)`` that can be included
            in the loss for regularization. This is returned only if
            ``self.regularization`` is ``True``.

        """
        return self._pass(x, inverse=False)

    def inverse(self, y):
        return self._pass(y, inverse=True)

    def _pass(self, x, inverse):
        # We import these here as torchdiffeq is an optional dependency.
        from torchdiffeq import odeint_adjoint
        from torchdiffeq import odeint

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
        if self.adjoint:
            integrator = odeint_adjoint
        else:
            integrator = odeint

        state_traj = integrator(
            func=self.ode_func, y0=state_t0, t=t,
            method=self.solver, options=self.solver_options,
            rtol=1e-4, atol=1e-4,
        )

        # Return the value of the trajectories at t=1.0.
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

    class TraceEstimators(enum.Enum):
        exact, hutchinson = range(2)

    def __init__(self, dynamics, trace_estimator, n_hutchinson_samples, vmap, requires_backward):
        super().__init__()
        self.dynamics = dynamics
        self.trace_estimator = trace_estimator
        self.n_hutchinson_samples = n_hutchinson_samples
        self.vmap = vmap
        self.requires_backward = requires_backward

        # This holds the random sample used for Gaussian estimation.
        # It will be initialized lazily in before_odeint().
        self._eps = None

        # This is used for vectorizing the exact calculation of the
        # Jacobian and it is initialized lazily since it needs the
        # feature dimension.
        self._cached_eye = None

    @property
    def trace_estimator(self):
        return self._trace_estimator.name

    @trace_estimator.setter
    def trace_estimator(self, new_trace_estimator):
        try:
            self._trace_estimator = getattr(self.TraceEstimators, new_trace_estimator)
        except AttributeError:
            raise ValueError('trace_estimator must be one of {}'.format(
                [e.name for e in self.TraceEstimators]))

    def before_odeint(self, x):
        """Prepares the function for a new integration.

        This regenerates the random sample used for Hutchinson's trace/Frobenious norm estimator.
        """
        if self._trace_estimator == self.TraceEstimators.hutchinson:
            self._eps = torch.randn(self.n_hutchinson_samples, *x.shape)
        elif self._cached_eye is None:
            # Initialize and cache grad_outputs used for vectorizing the
            # calculation of the Jacobian.
            self._cached_eye = torch.eye(x.shape[1])

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
            # During the backwards pass, we might try to set this on a non-leaf
            # variable, which is forbidden, but the variable might already be set correctly.
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
                if self._trace_estimator == self.TraceEstimators.exact:
                    trace, jac_norm = _trace_and_frobenious_squared_norm_exact(
                        vel, x, self._cached_eye, self.vmap, self.requires_backward)
                else:
                    trace, jac_norm = _trace_and_frobenious_squared_norm_hutchinson(
                        vel, x, self._eps, self.vmap, self.requires_backward)

                # Pack the value of the integrands.
                regularization_term = vel_squared_norm + jac_norm
                integrands = (vel, trace, regularization_term)
            else:
                if self._trace_estimator == self.TraceEstimators.exact:
                    trace = _trace_exact(vel, x, self._cached_eye, self.vmap, self.requires_backward)
                else:
                    trace = _trace_hutchinson(vel, x, self._eps, self.vmap, self.requires_backward)

                # Pack the value of the integrands.
                integrands = (vel, trace)

        return integrands


def _batch_squared_norm(x):
    return torch.sum(x**2, dim=-1)


def _trace_exact(f, x, cached_eye, vmap, create_graph):
    """Compute the exact trace of the Jacobian df/dx using autograd.

    f and x are the output and input tensors.

    cached_eye must be an eye matrix of shape (n_atoms*3, n_atoms*3) and it is
    used as the grads_output argument of torch.autograd.grad.
    """
    f_sum = f.sum(dim=0)
    if vmap:
        # torch._C._debug_only_display_vmap_fallback_warnings(True)
        grad = torch.autograd.grad(f_sum, x, cached_eye, create_graph=create_graph,
                                   retain_graph=True, is_grads_batched=True)[0]
        trace = torch.diagonal(grad, dim1=0, dim2=2).sum(dim=1)
    else:
        trace = 0.0
        for idx, grads_out in enumerate(cached_eye):
            trace += torch.autograd.grad(f_sum, x, grads_out, create_graph=create_graph,
                                         retain_graph=True)[0][:, idx]
    return trace


def _trace_hutchinson(f, x, eps, vmap, create_graph):
    """Compute the trace of the Jacobian df/dx using Hutchinson's estimator as in [2].

    f and x are the output and input tensors.

    eps are random Gaussian samples with shape (n_hutchinson_samples, batch_size, n_atoms*3).
    """
    if vmap:
        e_dfdx = torch.autograd.grad(f, x, eps, create_graph=create_graph,
                                     retain_graph=True, is_grads_batched=True)[0]
    else:
        e_dfdx = torch.empty_like(eps)
        for idx, e in enumerate(eps):
            e_dfdx[idx] = torch.autograd.grad(f, x, e, create_graph=create_graph,
                                              retain_graph=True)[0]

    trace = batchwise_dot(e_dfdx, eps).mean(dim=0)
    return trace


def _trace_and_frobenious_squared_norm_exact(f, x, cached_eye, vmap, create_graph):
    """Compute the exact trace and Frobenious norm of the Jacobian df/dx using autograd in D passes.

    f and x are the output and input tensors.

    cached_eye must be an eye matrix of shape (n_atoms*3, n_atoms*3) and it is
    used as the grads_output argument of torch.autograd.grad.
    """
    f_sum = f.sum(dim=0)

    if vmap:
        # torch._C._debug_only_display_vmap_fallback_warnings(True)
        grad = torch.autograd.grad(f_sum, x, cached_eye, create_graph=create_graph,
                                   retain_graph=True, is_grads_batched=True)[0]
        trace = torch.diagonal(grad, dim1=0, dim2=2).sum(dim=1)
        norm = torch.sum(grad**2, dim=-1).sum(dim=0)
    else:
        trace = 0.0
        norm = 0.0
        for idx, grads_out in enumerate(cached_eye):
            grad = torch.autograd.grad(f_sum, x, grads_out, create_graph=create_graph,
                                       retain_graph=True)[0]
            trace += grad[:, idx]
            norm += torch.sum(grad**2, dim=-1)

    return trace, norm


def _trace_and_frobenious_squared_norm_hutchinson(f, x, eps, vmap, create_graph):
    """Compute both trace and Frobenious norm of the Jacobian df/dx in a single pass using Hutchinson estimate.

    For details on the Frobenious estimator, see [3] in ContinuousFlow docstring.

    f and x are the output and input tensors.

    eps are random Gaussian samples with shape (n_hutchinson_samples, batch_size, n_atoms*3).
    """
    if vmap:
        e_dfdx = torch.autograd.grad(f, x, eps, create_graph=create_graph,
                                     retain_graph=True, is_grads_batched=True)[0]
    else:
        e_dfdx = torch.empty_like(eps)
        for idx, e in enumerate(eps):
            e_dfdx[idx] = torch.autograd.grad(f, x, e, create_graph=create_graph,
                                              retain_graph=True)[0]

    trace = batchwise_dot(e_dfdx, eps).mean(dim=0)
    frobenius = _batch_squared_norm(e_dfdx).mean(dim=0)
    return trace, frobenius
