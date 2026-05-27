"""Work, free-energy, and diagnostic utilities for stochastic path TFEP."""

from __future__ import annotations

import torch

from tfep.regularizers.bar import _bar_newton_solve_detached


def _as_batch(x: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(x).reshape(-1)


def log_mean_exp(x: torch.Tensor) -> torch.Tensor:
    x = _as_batch(x)
    m = torch.amax(x)
    return m + torch.log(torch.mean(torch.exp(x - m)))


def compute_path_work(
    u_source_x0: torch.Tensor,
    u_target_xK: torch.Tensor,
    sum_logJ: torch.Tensor,
    sum_logq_forward: torch.Tensor,
    sum_logq_reverse: torch.Tensor,
) -> torch.Tensor:
    """Compute reduced path work with explicit stochastic density terms.

    The convention is

    ``u_target - u_source - logJ + logq_forward - logq_reverse``.

    With no stochastic kernels this reduces exactly to deterministic TFEP.
    """
    u_source_x0 = _as_batch(u_source_x0)
    u_target_xK = _as_batch(u_target_xK)
    sum_logJ = _as_batch(sum_logJ)
    sum_logq_forward = _as_batch(sum_logq_forward)
    sum_logq_reverse = _as_batch(sum_logq_reverse)
    return u_target_xK - u_source_x0 - sum_logJ + sum_logq_forward - sum_logq_reverse


def jarzynski_delta_f_forward(w01: torch.Tensor) -> torch.Tensor:
    """Forward Jarzynski/FEP estimate ``Delta f = -log <exp(-w01)>``."""
    return -log_mean_exp(-_as_batch(w01))


def jarzynski_delta_f_reverse(w10: torch.Tensor) -> torch.Tensor:
    """Reverse FEP estimate using the current TFEP ``w10`` convention."""
    return log_mean_exp(-_as_batch(w10))


def bar_delta_f(w01: torch.Tensor, w10: torch.Tensor, *, max_iter: int = 100, tol: float = 1e-12) -> torch.Tensor:
    """BAR estimate compatible with :mod:`tfep.regularizers.bar` conventions."""
    w01 = _as_batch(w01)
    w10 = _as_batch(w10)
    if w01.numel() == 0 or w10.numel() == 0:
        raise ValueError("BAR requires non-empty forward and reverse work arrays")
    log_ratio = torch.log(
        torch.as_tensor(float(w10.numel()), dtype=w01.dtype, device=w01.device)
        / torch.as_tensor(float(w01.numel()), dtype=w01.dtype, device=w01.device)
    )
    return _bar_newton_solve_detached(
        w01.detach(),
        w10.detach(),
        log_ratio=log_ratio.detach(),
        max_iter=max_iter,
        tol=tol,
    )


def effective_sample_size_from_log_weights(log_weights: torch.Tensor) -> torch.Tensor:
    """Return normalized importance-sampling ESS from log weights."""
    log_weights = _as_batch(log_weights)
    normalized = torch.softmax(log_weights, dim=0)
    return 1.0 / torch.sum(normalized.pow(2))


def assert_finite_path_terms(**terms: torch.Tensor) -> None:
    """Raise if any supplied path term contains NaN or Inf."""
    for name, value in terms.items():
        tensor = torch.as_tensor(value)
        if not torch.all(torch.isfinite(tensor)):
            raise FloatingPointError(f"Non-finite stochastic TFEP term: {name}")
