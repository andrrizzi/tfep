"""Reweighting estimators for biased/enhanced-sampling trajectories.

The functions in this module use dimensionless log weights, i.e. the logarithm
of the factor that converts a sample from the biased simulation ensemble back
to the intended unbiased ensemble. For a static PLUMED bias this is typically
``log_weight = +bias/kT``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def normalize_log_weights(log_weights: np.ndarray) -> np.ndarray:
    """Return log weights normalized so that ``sum(exp(logw)) == 1``."""
    logw = np.asarray(log_weights, dtype=np.float64).reshape(-1)
    if logw.size == 0:
        return logw.copy()
    finite = np.isfinite(logw)
    out = np.full_like(logw, -np.inf, dtype=np.float64)
    if not np.any(finite):
        return out
    m = float(np.max(logw[finite]))
    norm = m + float(np.log(np.sum(np.exp(logw[finite] - m))))
    out[finite] = logw[finite] - norm
    return out


def normalize_log_weights_torch(log_weights: torch.Tensor) -> torch.Tensor:
    """Torch version of :func:`normalize_log_weights`."""
    logw = torch.as_tensor(log_weights).reshape(-1)
    return logw - torch.logsumexp(logw, dim=0)


def effective_sample_size(log_weights: np.ndarray) -> float:
    """Return Kish effective sample size for unnormalized log weights."""
    logw_norm = normalize_log_weights(log_weights)
    weights = np.exp(logw_norm[np.isfinite(logw_norm)])
    if weights.size == 0:
        return float("nan")
    denom = float(np.sum(weights * weights))
    if denom <= 0.0:
        return float("nan")
    return float(1.0 / denom)


def effective_sample_size_torch(log_weights: torch.Tensor) -> torch.Tensor:
    """Torch version of :func:`effective_sample_size`."""
    w = torch.exp(normalize_log_weights_torch(log_weights))
    return 1.0 / torch.sum(w * w)


def log_weight_diagnostics(log_weights: Optional[np.ndarray]) -> dict[str, Any]:
    """Summarize log-weight stability and weight collapse risk."""
    if log_weights is None:
        return {
            "weighted": False,
            "n": 0,
            "ess": float("nan"),
            "ess_ratio": float("nan"),
            "log_weight_min": float("nan"),
            "log_weight_max": float("nan"),
            "log_weight_span": float("nan"),
        }
    logw = np.asarray(log_weights, dtype=np.float64).reshape(-1)
    finite = logw[np.isfinite(logw)]
    ess = effective_sample_size(finite) if finite.size else float("nan")
    return {
        "weighted": True,
        "n": int(logw.size),
        "n_finite": int(finite.size),
        "ess": float(ess),
        "ess_ratio": float(ess / finite.size) if finite.size and np.isfinite(ess) else float("nan"),
        "log_weight_min": float(np.min(finite)) if finite.size else float("nan"),
        "log_weight_max": float(np.max(finite)) if finite.size else float("nan"),
        "log_weight_span": float(np.max(finite) - np.min(finite)) if finite.size else float("nan"),
    }


def _finite_pair(work: np.ndarray, log_weights: Optional[np.ndarray]) -> tuple[np.ndarray, Optional[np.ndarray]]:
    work = np.asarray(work, dtype=np.float64).reshape(-1)
    if log_weights is None:
        return work[np.isfinite(work)], None
    logw = np.asarray(log_weights, dtype=np.float64).reshape(-1)
    if logw.shape != work.shape:
        raise ValueError("work and log_weights must have the same shape")
    finite = np.isfinite(work) & np.isfinite(logw)
    return work[finite], logw[finite]


def weighted_fep_deltaf(work: np.ndarray, log_weights: Optional[np.ndarray] = None) -> float:
    """Weighted exponential FEP estimator in reduced units."""
    work, logw = _finite_pair(work, log_weights)
    if work.size == 0:
        return float("nan")
    if logw is None:
        logw_norm = np.full(work.shape, -np.log(float(work.size)), dtype=np.float64)
    else:
        logw_norm = normalize_log_weights(logw)
    x = -work + logw_norm
    return float(-np.logaddexp.reduce(x))


def _default_log_ratio(
    n_forward: int,
    n_reverse: int,
    log_weights_forward: Optional[np.ndarray],
    log_weights_reverse: Optional[np.ndarray],
) -> float:
    del n_forward, n_reverse, log_weights_forward, log_weights_reverse
    # The two importance-weighted ensemble averages are normalized separately.
    # ESS controls uncertainty; it is not a thermodynamic free-energy offset.
    return 0.0


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid for NumPy arrays."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def weighted_bar_deltaf(
    w_forward: np.ndarray,
    w_reverse: np.ndarray,
    log_weights_forward: Optional[np.ndarray] = None,
    log_weights_reverse: Optional[np.ndarray] = None,
    *,
    log_ratio: Optional[float] = None,
    max_iter: int = 200,
    tol: float = 1e-12,
) -> Tuple[float, float]:
    """Weighted BAR estimate in reduced units.

    The default uses a symmetric Bennett bridge between the two separately
    normalized target-ensemble averages. In particular, unequal Kish ESS values
    must not be inserted as an additive free-energy offset.

    The uncertainty is intentionally returned as NaN for now; use weighted
    bootstrap for uncertainty.
    """
    wf, lwf = _finite_pair(w_forward, log_weights_forward)
    wr, lwr = _finite_pair(w_reverse, log_weights_reverse)
    if wf.size < 2 or wr.size < 2:
        return float("nan"), float("nan")
    af = np.exp(normalize_log_weights(np.zeros_like(wf) if lwf is None else lwf))
    ar = np.exp(normalize_log_weights(np.zeros_like(wr) if lwr is None else lwr))
    lr = _default_log_ratio(wf.size, wr.size, lwf, lwr) if log_ratio is None else float(log_ratio)
    df = 0.5 * (float(np.sum(af * wf)) - float(np.sum(ar * wr)))
    for _ in range(int(max_iter)):
        sf = _sigmoid_np(wf - df - lr)
        sr = _sigmoid_np(wr + df + lr)
        g = -float(np.sum(af * sf)) + float(np.sum(ar * sr))
        h = float(np.sum(af * sf * (1.0 - sf)) + np.sum(ar * sr * (1.0 - sr))) + 1e-14
        step = g / h
        df_new = df - step
        if abs(step) < float(tol):
            df = df_new
            break
        df = df_new
    return float(df), float("nan")


def _torch_weights(
    log_weights: Optional[torch.Tensor],
    template: torch.Tensor,
    *,
    log_normalizer: Optional[torch.Tensor] = None,
    population_size: Optional[int] = None,
) -> torch.Tensor:
    """Return local or globally normalized importance weights.

    With ``log_normalizer`` and ``population_size``, this returns a
    Horvitz-Thompson minibatch contribution. Averaging these contributions over
    uniformly sampled batches reproduces the objective normalized over the full
    training population rather than a separate softmax objective in each batch.
    """
    values = template.reshape(-1)
    if (log_normalizer is None) != (population_size is None):
        raise ValueError("log_normalizer and population_size must be provided together")

    if log_normalizer is None:
        if log_weights is None:
            return torch.full_like(values, 1.0 / float(values.numel()))
        return torch.exp(normalize_log_weights_torch(log_weights.to(device=values.device, dtype=values.dtype)))

    n_population = int(population_size)
    if n_population <= 0 or values.numel() <= 0:
        raise ValueError("population_size and minibatch size must be positive")
    if log_weights is None:
        logw = torch.zeros_like(values)
    else:
        logw = torch.as_tensor(log_weights, device=values.device, dtype=values.dtype).reshape(-1)
        if logw.shape != values.shape:
            raise ValueError("log_weights and template must have the same shape")
    log_norm = torch.as_tensor(log_normalizer, device=values.device, dtype=values.dtype).reshape(())
    log_ht_scale = torch.log(
        torch.as_tensor(float(n_population) / float(values.numel()), device=values.device, dtype=values.dtype)
    )
    return torch.exp(logw - log_norm + log_ht_scale)


def weighted_bar_objective(
    w_forward: torch.Tensor,
    w_reverse: torch.Tensor,
    df: torch.Tensor,
    log_ratio: torch.Tensor,
    log_weights_forward: Optional[torch.Tensor] = None,
    log_weights_reverse: Optional[torch.Tensor] = None,
    *,
    log_weight_normalizer_forward: Optional[torch.Tensor] = None,
    log_weight_normalizer_reverse: Optional[torch.Tensor] = None,
    population_size_forward: Optional[int] = None,
    population_size_reverse: Optional[int] = None,
) -> torch.Tensor:
    """Differentiable weighted Bennett objective."""
    wf = torch.as_tensor(w_forward).reshape(-1)
    wr = torch.as_tensor(w_reverse).reshape(-1)
    af = _torch_weights(
        log_weights_forward,
        wf,
        log_normalizer=log_weight_normalizer_forward,
        population_size=population_size_forward,
    )
    ar = _torch_weights(
        log_weights_reverse,
        wr,
        log_normalizer=log_weight_normalizer_reverse,
        population_size=population_size_reverse,
    )
    return torch.sum(af * F.softplus(wf - df - log_ratio)) + torch.sum(ar * F.softplus(wr + df + log_ratio))


@dataclass(frozen=True)
class WeightedBarSolveResult:
    """Detached BAR root-solve result plus passive diagnostics."""

    df: torch.Tensor
    converged: bool
    used_fallback: bool
    iterations: int
    residual_abs: torch.Tensor


def _weighted_bar_detached_terms(
    w_forward: torch.Tensor,
    w_reverse: torch.Tensor,
    *,
    log_weights_forward: Optional[torch.Tensor] = None,
    log_weights_reverse: Optional[torch.Tensor] = None,
    log_ratio: Optional[torch.Tensor] = None,
    log_weight_normalizer_forward: Optional[torch.Tensor] = None,
    log_weight_normalizer_reverse: Optional[torch.Tensor] = None,
    population_size_forward: Optional[int] = None,
    population_size_reverse: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.dtype]:
    """Return detached float64 BAR terms on the input device."""
    wf_template = torch.as_tensor(w_forward).detach().reshape(-1)
    wr_template = torch.as_tensor(w_reverse).detach().reshape(-1)
    out_dtype = wf_template.dtype
    wf = wf_template.to(dtype=torch.float64)
    wr = wr_template.to(device=wf.device, dtype=torch.float64)
    af = _torch_weights(
        None if log_weights_forward is None else log_weights_forward.detach(),
        wf,
        log_normalizer=log_weight_normalizer_forward,
        population_size=population_size_forward,
    )
    ar = _torch_weights(
        None if log_weights_reverse is None else log_weights_reverse.detach(),
        wr,
        log_normalizer=log_weight_normalizer_reverse,
        population_size=population_size_reverse,
    )
    if log_ratio is None:
        lr = torch.zeros((), device=wf.device, dtype=wf.dtype)
    else:
        lr = torch.as_tensor(log_ratio, device=wf.device, dtype=wf.dtype).detach().reshape(())
    return wf, wr, af, ar, lr, out_dtype


def _weighted_bar_gradient(
    wf: torch.Tensor,
    wr: torch.Tensor,
    af: torch.Tensor,
    ar: torch.Tensor,
    df: torch.Tensor,
    log_ratio: torch.Tensor,
) -> torch.Tensor:
    """Derivative of the weighted BAR objective with respect to ``df``."""
    sf = torch.sigmoid(wf - df - log_ratio)
    sr = torch.sigmoid(wr + df + log_ratio)
    return -torch.sum(af * sf) + torch.sum(ar * sr)


def weighted_bar_newton_solve_detached(
    w_forward: torch.Tensor,
    w_reverse: torch.Tensor,
    *,
    log_weights_forward: Optional[torch.Tensor] = None,
    log_weights_reverse: Optional[torch.Tensor] = None,
    log_ratio: Optional[torch.Tensor] = None,
    log_weight_normalizer_forward: Optional[torch.Tensor] = None,
    log_weight_normalizer_reverse: Optional[torch.Tensor] = None,
    population_size_forward: Optional[int] = None,
    population_size_reverse: Optional[int] = None,
    df_init: Optional[torch.Tensor] = None,
    max_iter: int = 25,
    tol: float = 1e-10,
) -> torch.Tensor:
    """Detached Newton solve for the weighted BAR optimum."""
    wf = torch.as_tensor(w_forward).detach().reshape(-1)
    wr = torch.as_tensor(w_reverse).detach().reshape(-1)
    af = _torch_weights(
        None if log_weights_forward is None else log_weights_forward.detach(),
        wf,
        log_normalizer=log_weight_normalizer_forward,
        population_size=population_size_forward,
    )
    ar = _torch_weights(
        None if log_weights_reverse is None else log_weights_reverse.detach(),
        wr,
        log_normalizer=log_weight_normalizer_reverse,
        population_size=population_size_reverse,
    )
    if log_ratio is None:
        lr = torch.zeros((), device=wf.device, dtype=wf.dtype)
    else:
        lr = torch.as_tensor(log_ratio, device=wf.device, dtype=wf.dtype).detach().reshape(())
    if df_init is None:
        mean_f = torch.sum(af * wf) / torch.sum(af)
        mean_r = torch.sum(ar * wr) / torch.sum(ar)
        df = 0.5 * (mean_f - mean_r)
    else:
        df = df_init.detach().reshape(())
    for _ in range(int(max_iter)):
        sf = torch.sigmoid(wf - df - lr)
        sr = torch.sigmoid(wr + df + lr)
        g = -torch.sum(af * sf) + torch.sum(ar * sr)
        h = torch.sum(af * sf * (1.0 - sf)) + torch.sum(ar * sr * (1.0 - sr)) + torch.as_tensor(1e-12, device=wf.device, dtype=wf.dtype)
        step = g / h
        df_new = df - step
        if torch.abs(step).item() < tol:
            df = df_new
            break
        df = df_new
    return df.reshape(())


def weighted_bar_robust_solve_detached(
    w_forward: torch.Tensor,
    w_reverse: torch.Tensor,
    *,
    log_weights_forward: Optional[torch.Tensor] = None,
    log_weights_reverse: Optional[torch.Tensor] = None,
    log_ratio: Optional[torch.Tensor] = None,
    log_weight_normalizer_forward: Optional[torch.Tensor] = None,
    log_weight_normalizer_reverse: Optional[torch.Tensor] = None,
    population_size_forward: Optional[int] = None,
    population_size_reverse: Optional[int] = None,
    df_init: Optional[torch.Tensor] = None,
    max_iter: int = 25,
    tol: float = 1e-10,
) -> WeightedBarSolveResult:
    """Robust detached solve for the weighted BAR optimum.

    The solved equation is identical to :func:`weighted_bar_newton_solve_detached`.
    This first attempts Newton iterations in float64 and falls back to bracketed
    bisection only when Newton is not numerically trustworthy.
    """
    wf, wr, af, ar, lr, out_dtype = _weighted_bar_detached_terms(
        w_forward,
        w_reverse,
        log_weights_forward=log_weights_forward,
        log_weights_reverse=log_weights_reverse,
        log_ratio=log_ratio,
        log_weight_normalizer_forward=log_weight_normalizer_forward,
        log_weight_normalizer_reverse=log_weight_normalizer_reverse,
        population_size_forward=population_size_forward,
        population_size_reverse=population_size_reverse,
    )
    if wf.numel() == 0 or wr.numel() == 0:
        zero = torch.zeros((), device=wf.device, dtype=out_dtype)
        nan = torch.as_tensor(float("nan"), device=wf.device, dtype=torch.float64)
        return WeightedBarSolveResult(
            df=zero,
            converged=False,
            used_fallback=False,
            iterations=0,
            residual_abs=nan,
        )

    if df_init is None:
        mean_f = torch.sum(af * wf) / torch.sum(af)
        mean_r = torch.sum(ar * wr) / torch.sum(ar)
        df = 0.5 * (mean_f - mean_r)
    else:
        df = torch.as_tensor(df_init, device=wf.device, dtype=wf.dtype).detach().reshape(())

    finite_data = torch.cat([wf.reshape(-1), wr.reshape(-1), lr.reshape(1)])
    finite_data = finite_data[torch.isfinite(finite_data)]
    scale = float(finite_data.abs().max().item()) if finite_data.numel() else 1.0
    scale = max(scale, 1.0)
    residual_tol = max(float(tol), 1e-10)
    iterations = 0

    # Newton phase. If this looks unsafe, use the monotone root fallback below.
    if torch.isfinite(df):
        step_limit = 100.0 * (scale + abs(float(df.item())) + 1.0)
        for _ in range(max(0, int(max_iter))):
            iterations += 1
            sf = torch.sigmoid(wf - df - lr)
            sr = torch.sigmoid(wr + df + lr)
            g = -torch.sum(af * sf) + torch.sum(ar * sr)
            h = (
                torch.sum(af * sf * (1.0 - sf))
                + torch.sum(ar * sr * (1.0 - sr))
                + torch.as_tensor(1e-12, device=wf.device, dtype=wf.dtype)
            )
            step = g / h
            df_new = df - step
            if (
                not torch.isfinite(step)
                or not torch.isfinite(df_new)
                or abs(float(step.detach().item())) > step_limit
            ):
                break
            df = df_new
            residual_abs = torch.abs(_weighted_bar_gradient(wf, wr, af, ar, df, lr))
            if float(residual_abs.detach().item()) <= residual_tol or abs(float(step.detach().item())) <= float(tol):
                return WeightedBarSolveResult(
                    df=df.to(dtype=out_dtype).reshape(()),
                    converged=True,
                    used_fallback=False,
                    iterations=iterations,
                    residual_abs=residual_abs,
                )
        residual_abs = torch.abs(_weighted_bar_gradient(wf, wr, af, ar, df, lr))
        if torch.isfinite(df) and torch.isfinite(residual_abs) and float(residual_abs.detach().item()) <= residual_tol:
            return WeightedBarSolveResult(
                df=df.to(dtype=out_dtype).reshape(()),
                converged=True,
                used_fallback=False,
                iterations=iterations,
                residual_abs=residual_abs,
            )

    # Bracketed bisection fallback. The gradient is monotone increasing in df.
    if not torch.isfinite(df):
        mean_f = torch.sum(af * wf) / torch.sum(af)
        mean_r = torch.sum(ar * wr) / torch.sum(ar)
        df = 0.5 * (mean_f - mean_r)
    center = df.detach().reshape(())
    width = torch.as_tensor(scale + 1.0, device=wf.device, dtype=wf.dtype)
    left = center - width
    right = center + width
    g_left = _weighted_bar_gradient(wf, wr, af, ar, left, lr)
    g_right = _weighted_bar_gradient(wf, wr, af, ar, right, lr)
    bracketed = bool(
        torch.isfinite(g_left).item()
        and torch.isfinite(g_right).item()
        and float(g_left.item()) <= 0.0
        and float(g_right.item()) >= 0.0
    )
    for _ in range(64):
        if bracketed:
            break
        width = width * 2.0
        left = center - width
        right = center + width
        g_left = _weighted_bar_gradient(wf, wr, af, ar, left, lr)
        g_right = _weighted_bar_gradient(wf, wr, af, ar, right, lr)
        bracketed = bool(
            torch.isfinite(g_left).item()
            and torch.isfinite(g_right).item()
            and float(g_left.item()) <= 0.0
            and float(g_right.item()) >= 0.0
        )

    if not bracketed:
        candidates = torch.stack([left, center, right])
        residuals = torch.stack([
            torch.abs(_weighted_bar_gradient(wf, wr, af, ar, left, lr)),
            torch.abs(_weighted_bar_gradient(wf, wr, af, ar, center, lr)),
            torch.abs(_weighted_bar_gradient(wf, wr, af, ar, right, lr)),
        ])
        safe_residuals = torch.where(torch.isfinite(residuals), residuals, torch.full_like(residuals, float("inf")))
        idx = int(torch.argmin(safe_residuals).item())
        best_df = candidates[idx]
        best_residual = residuals[idx]
        return WeightedBarSolveResult(
            df=best_df.to(dtype=out_dtype).reshape(()),
            converged=False,
            used_fallback=True,
            iterations=iterations,
            residual_abs=best_residual,
        )

    max_bisect_iter = max(64, 2 * int(max_iter))
    mid = center
    g_mid = _weighted_bar_gradient(wf, wr, af, ar, mid, lr)
    converged = False
    for _ in range(max_bisect_iter):
        iterations += 1
        mid = 0.5 * (left + right)
        g_mid = _weighted_bar_gradient(wf, wr, af, ar, mid, lr)
        if torch.abs(g_mid).item() <= residual_tol or torch.abs(right - left).item() <= float(tol):
            converged = True
            break
        if float(g_mid.item()) < 0.0:
            left = mid
        else:
            right = mid
    residual_abs = torch.abs(g_mid)
    return WeightedBarSolveResult(
        df=mid.to(dtype=out_dtype).reshape(()),
        converged=bool(converged or float(residual_abs.detach().item()) <= residual_tol),
        used_fallback=True,
        iterations=iterations,
        residual_abs=residual_abs,
    )


@dataclass(frozen=True)
class WeightedBootstrapResult:
    summary: dict[str, Any]
    samples: np.ndarray


def weighted_bootstrap_bar_summary(
    w_forward: np.ndarray,
    w_reverse: np.ndarray,
    log_weights_forward: Optional[np.ndarray],
    log_weights_reverse: Optional[np.ndarray],
    *,
    n_boot: int,
    ci: float,
    seed: int,
    block_size: int = 1,
) -> WeightedBootstrapResult:
    """Bootstrap weighted BAR by resampling aligned work/log-weight blocks."""
    wf, lwf = _finite_pair(w_forward, log_weights_forward)
    wr, lwr = _finite_pair(w_reverse, log_weights_reverse)
    rng = np.random.default_rng(seed)
    estimates = np.empty(int(n_boot), dtype=np.float64)

    def _resample_indices(n_samples: int) -> np.ndarray:
        size = max(1, int(block_size))
        if size == 1:
            return rng.integers(0, n_samples, size=n_samples)
        starts = np.arange(0, n_samples, size, dtype=int)
        chunks = []
        total = 0
        while total < n_samples:
            start = int(starts[rng.integers(0, len(starts))])
            chunk = np.arange(start, min(start + size, n_samples), dtype=int)
            chunks.append(chunk)
            total += len(chunk)
        return np.concatenate(chunks)[:n_samples]

    for i in range(int(n_boot)):
        idx_f = _resample_indices(wf.size)
        idx_r = _resample_indices(wr.size)
        estimates[i] = weighted_bar_deltaf(
            wf[idx_f],
            wr[idx_r],
            None if lwf is None else lwf[idx_f],
            None if lwr is None else lwr[idx_r],
        )[0]
    estimates = estimates[np.isfinite(estimates)]
    alpha = 1.0 - float(ci)
    summary = {
        "n_boot": int(n_boot),
        "n_finite": int(estimates.size),
        "mean": float(np.mean(estimates)) if estimates.size else float("nan"),
        "std": float(np.std(estimates, ddof=1)) if estimates.size > 1 else float("nan"),
        "ci_low": float(np.quantile(estimates, alpha / 2.0)) if estimates.size else float("nan"),
        "ci_high": float(np.quantile(estimates, 1.0 - alpha / 2.0)) if estimates.size else float("nan"),
    }
    return WeightedBootstrapResult(summary=summary, samples=estimates)
