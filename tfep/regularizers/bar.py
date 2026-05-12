# tfep/regularizers/bar.py
"""
Regularizers related to bidirectional TFEP / work overlap.

This module provides:

- fep_forward_df / fep_reverse_df: one-sided exponential estimators (dimensionless Δf).
- BARLikeRegularizer: a *Bennett/BAR-objective* regularizer for training.

Important
---------
This is a TRAINING REGULARIZER, not the final Δf estimator.
It adds a penalty that is small when the two generalized-work distributions
overlap well, using the Bennett (logistic) objective.

Given dimensionless generalized works (kT units):
    w01 = u1(f(x0)) - log|J01| - u0(x0)
    w10 = u0(f^{-1}(x1)) - log|J10| - u1(x1)

Define the Bennett objective (negative log-likelihood up to a constant):
    J(Δf) = E0[ softplus(w01 - Δf - log(N1/N0)) ] + E1[ softplus(w10 + Δf + log(N1/N0)) ]

The BAR estimate Δf_BAR minimizes J(Δf).

Regularizer:
    L_reg = λ * J(Δf_BAR)

Implementation details
----------------------
- Δf_BAR is solved via Newton steps on *detached* works (no gradients through the solver).
- The objective J is evaluated on non-detached works, but with Δf_BAR detached by default.
  This yields stable gradients that flow through w01/w10 into the map.
- Supports compatibility alias: lam == lambda_bar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _log_mean_exp(x: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
    """Stable log(mean(exp(x)))."""
    if dim is None:
        x = x.reshape(-1)
        dim = 0
    m = torch.amax(x, dim=dim, keepdim=True)
    return (m + torch.log(torch.mean(torch.exp(x - m), dim=dim, keepdim=True))).squeeze(dim)


def fep_forward_df(w01: torch.Tensor) -> torch.Tensor:
    """
    Forward exponential estimator (dimensionless Δf) from w01:
        Δf = -log < exp(-w01) >
    """
    w01 = torch.as_tensor(w01).reshape(-1)
    return -_log_mean_exp(-w01)


def fep_reverse_df(w10: torch.Tensor) -> torch.Tensor:
    """
    Reverse exponential estimator (dimensionless Δf) from w10:
        Δf = +log < exp(-w10) >
    (With w10 = u0(x1) - u1(x1) convention.)
    """
    w10 = torch.as_tensor(w10).reshape(-1)
    return _log_mean_exp(-w10)


def bar_objective(
    w01: torch.Tensor,
    w10: torch.Tensor,
    df: torch.Tensor,
    log_ratio: torch.Tensor,
) -> torch.Tensor:
    """
    Bennett/BAR objective (dimensionless).

    J(df) = mean0[ softplus(w01 - df - log_ratio) ] + mean1[ softplus(w10 + df + log_ratio) ]
    where log_ratio = log(N1/N0).
    """
    w01 = w01.reshape(-1)
    w10 = w10.reshape(-1)
    # softplus(x) = log(1+exp(x)) is stable.
    return F.softplus(w01 - df - log_ratio).mean() + F.softplus(w10 + df + log_ratio).mean()


def _bar_newton_solve_detached(
    w01_det: torch.Tensor,
    w10_det: torch.Tensor,
    *,
    log_ratio: torch.Tensor,
    df_init: Optional[torch.Tensor] = None,
    max_iter: int = 25,
    tol: float = 1e-10,
) -> torch.Tensor:
    """
    Newton solve for df minimizing the Bennett objective, using DETACHED works.
    Returns a scalar tensor df (same device/dtype as inputs).

    Gradient:
      dJ/ddf = -E0[ sigmoid(w01 - df - log_ratio) ] + E1[ sigmoid(w10 + df + log_ratio) ]

    Hessian:
      d2J/ddf2 = E0[ s0*(1-s0) ] + E1[ s1*(1-s1) ]
    """
    w01_det = w01_det.reshape(-1)
    w10_det = w10_det.reshape(-1)

    if w01_det.numel() == 0 or w10_det.numel() == 0:
        # No data -> return 0
        return torch.zeros((), device=w01_det.device, dtype=w01_det.dtype)

    # Initialize df. A decent heuristic is mid of means.
    if df_init is None:
        df = 0.5 * (w01_det.mean() - w10_det.mean())
    else:
        df = df_init.reshape(())

    # Newton iterations (with small damping for numerical safety).
    for _ in range(int(max_iter)):
        # s0 = sigmoid(w01 - df - log_ratio)
        # s1 = sigmoid(w10 + df + log_ratio)
        s0 = torch.sigmoid(w01_det - df - log_ratio)
        s1 = torch.sigmoid(w10_det + df + log_ratio)

        g = -s0.mean() + s1.mean()
        h = (s0 * (1.0 - s0)).mean() + (s1 * (1.0 - s1)).mean()

        # Avoid division by zero in extreme regimes.
        h = h + torch.as_tensor(1e-12, device=h.device, dtype=h.dtype)

        step = g / h
        df_new = df - step

        if torch.abs(step).item() < tol:
            df = df_new
            break
        df = df_new

    return df.reshape(())


class BARLikeRegularizer(nn.Module):
    """
    Bennett/BAR-objective training regularizer.

    Computes df_BAR on detached works and adds:
        lambda_bar * J(df_BAR)
    where J is the Bennett softplus objective (see module docstring).

    Parameters
    ----------
    lambda_bar : float
        Regularization weight λ.
    lam : float, optional
        Backward-compatible alias for lambda_bar.
    max_iter : int
        Newton iterations for the detached df solve.
    tol : float
        Convergence tolerance for Newton updates (on |step|).
    detach_df : bool
        If True (recommended), df_BAR is treated as a constant in J(df_BAR)
        so gradients flow only through w01/w10 (not through the solver).
    warm_start : bool
        If True, reuses the previous batch df as initialization.
    """

    def __init__(
        self,
        *,
        lambda_bar: float = 0.0,
        lam: Optional[float] = None,
        max_iter: int = 25,
        tol: float = 1e-10,
        detach_df: bool = True,
        warm_start: bool = True,
    ) -> None:
        super().__init__()

        if lam is not None:
            lambda_bar = float(lam)

        self.lambda_bar = float(lambda_bar)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.detach_df = bool(detach_df)
        self.warm_start = bool(warm_start)

        # Keep last df as a buffer for warm start (scalar).
        self.register_buffer("_last_df", torch.zeros(()), persistent=False)

    # Backward-compatible attribute name
    @property
    def lam(self) -> float:
        return self.lambda_bar

    @lam.setter
    def lam(self, value: float) -> None:
        self.lambda_bar = float(value)

    @property
    def last_df(self) -> float:
        """Last detached df_BAR value (Python float) for monitoring."""
        return float(self._last_df.item())

    def forward(self, w01: torch.Tensor, w10: torch.Tensor) -> torch.Tensor:
        """
        Return scalar regularization loss (same dtype/device as inputs).
        """
        if self.lambda_bar <= 0.0:
            # Return a differentiable zero on the right device/dtype.
            w01 = torch.as_tensor(w01)
            return torch.zeros((), device=w01.device, dtype=w01.dtype)

        w01 = torch.as_tensor(w01).reshape(-1)
        w10 = torch.as_tensor(w10).reshape(-1)

        n0 = int(w01.numel())
        n1 = int(w10.numel())
        if n0 == 0 or n1 == 0:
            return torch.zeros((), device=w01.device, dtype=w01.dtype)

        # log_ratio = log(N1/N0) for unequal sample sizes (usually 0 in your setup).
        log_ratio = torch.log(
            torch.as_tensor(float(n1), device=w01.device, dtype=w01.dtype)
            / torch.as_tensor(float(n0), device=w01.device, dtype=w01.dtype)
        )

        # Solve df on DETACHED works (no gradient through solve).
        df_init = self._last_df if self.warm_start else None
        df_bar = _bar_newton_solve_detached(
            w01.detach(),
            w10.detach(),
            log_ratio=log_ratio.detach(),
            df_init=df_init.detach() if df_init is not None else None,
            max_iter=self.max_iter,
            tol=self.tol,
        ).detach()

        # Update warm start buffer
        self._last_df = df_bar

        # Evaluate Bennett objective on non-detached works for gradients.
        df_for_obj = df_bar.detach() if self.detach_df else df_bar
        reg = bar_objective(w01, w10, df_for_obj, log_ratio)

        return torch.as_tensor(self.lambda_bar, device=reg.device, dtype=reg.dtype) * reg
