#!/usr/bin/env python

import numpy as np
import torch

from tfep.analysis import reweighting as rw
from tfep.regularizers import bar as barlib


def _unweighted_fep(work):
    work = np.asarray(work, dtype=np.float64)
    x = -work
    m = np.max(x)
    return -(m + np.log(np.mean(np.exp(x - m))))


def test_weighted_fep_reduces_to_unweighted_for_zero_log_weights():
    work = np.array([0.1, -0.2, 0.4, 1.0, -0.8], dtype=np.float64)

    got = rw.weighted_fep_deltaf(work, np.zeros_like(work))

    assert np.isclose(got, _unweighted_fep(work), atol=1e-12, rtol=1e-12)


def test_log_weight_normalization_and_ess_are_shift_invariant():
    logw = np.array([-1000.0, -999.0, -998.0], dtype=np.float64)

    norm0 = rw.normalize_log_weights(logw)
    norm1 = rw.normalize_log_weights(logw + 12345.0)

    assert np.allclose(norm0, norm1)
    assert np.isclose(np.exp(norm0).sum(), 1.0)
    assert np.isclose(rw.effective_sample_size(logw), rw.effective_sample_size(logw + 12345.0))


def test_weighted_bar_reduces_to_unweighted_for_zero_log_weights():
    w01 = torch.tensor([0.3, 0.7, 1.1, 0.2], dtype=torch.double)
    w10 = torch.tensor([-0.1, 0.4, 0.8, 1.2], dtype=torch.double)
    log_ratio = torch.log(torch.as_tensor(float(w10.numel()) / float(w01.numel()), dtype=torch.double))
    expected = barlib._bar_newton_solve_detached(
        w01,
        w10,
        log_ratio=log_ratio,
        max_iter=200,
        tol=1e-12,
    ).item()

    got = rw.weighted_bar_deltaf(
        w01.numpy(),
        w10.numpy(),
        np.zeros(w01.numel()),
        np.zeros(w10.numel()),
        max_iter=200,
        tol=1e-12,
    )[0]

    assert np.isclose(got, expected, atol=1e-10, rtol=1e-10)


def test_weighted_bar_default_does_not_turn_ess_into_free_energy_offset():
    w01 = np.array([-0.4, 0.1, 0.8, 1.3, 0.5], dtype=np.float64)
    w10 = np.array([-0.2, 0.3, 0.9, 1.4, 0.0], dtype=np.float64)
    logw01 = np.array([0.0, -1.0, -3.0, -5.0, -7.0], dtype=np.float64)
    logw10 = np.array([0.0, -0.1, -0.2, -0.3, -0.4], dtype=np.float64)

    assert not np.isclose(rw.effective_sample_size(logw01), rw.effective_sample_size(logw10))
    default = rw.weighted_bar_deltaf(w01, w10, logw01, logw10)[0]
    symmetric = rw.weighted_bar_deltaf(w01, w10, logw01, logw10, log_ratio=0.0)[0]

    assert np.isclose(default, symmetric, atol=1e-12, rtol=1e-12)


def test_weighted_bar_block_bootstrap_keeps_work_weight_pairs_aligned():
    work = np.arange(12, dtype=np.float64)
    logw = -work.copy()

    result = rw.weighted_bootstrap_bar_summary(
        work,
        -work,
        logw,
        logw,
        n_boot=5,
        ci=0.95,
        seed=17,
        block_size=3,
    )

    assert result.summary["n_finite"] == 5
    assert np.isfinite(result.samples).all()


def test_weighted_bar_objective_reduces_to_unweighted_for_zero_log_weights():
    w01 = torch.tensor([0.3, 0.7, 1.1, 0.2], dtype=torch.double)
    w10 = torch.tensor([-0.1, 0.4, 0.8, 1.2], dtype=torch.double)
    df = torch.tensor(0.25, dtype=torch.double)
    log_ratio = torch.zeros((), dtype=torch.double)

    got = rw.weighted_bar_objective(
        w01,
        w10,
        df,
        log_ratio,
        log_weights_forward=torch.zeros_like(w01),
        log_weights_reverse=torch.zeros_like(w10),
    )
    expected = barlib.bar_objective(w01, w10, df, log_ratio)

    assert torch.allclose(got, expected, atol=1e-12, rtol=1e-12)


def test_globally_normalized_minibatch_objectives_average_to_full_objective():
    w01 = torch.tensor([0.2, 0.5, 1.2, -0.1, 0.8, 1.6], dtype=torch.double)
    w10 = torch.tensor([-0.4, 0.3, 0.7, 1.1, -0.2, 0.9], dtype=torch.double)
    logw01 = torch.tensor([-2.0, 0.0, 1.0, -0.5, 0.7, -1.2], dtype=torch.double)
    logw10 = torch.tensor([0.4, -0.8, 1.3, -1.1, 0.0, 0.6], dtype=torch.double)
    df = torch.tensor(0.35, dtype=torch.double)
    norm01 = torch.logsumexp(logw01, dim=0)
    norm10 = torch.logsumexp(logw10, dim=0)
    ess01 = rw.effective_sample_size_torch(logw01)
    ess10 = rw.effective_sample_size_torch(logw10)
    log_ratio = torch.log(ess10 / ess01)

    full = rw.weighted_bar_objective(
        w01,
        w10,
        df,
        log_ratio,
        log_weights_forward=logw01,
        log_weights_reverse=logw10,
    )
    batch_objectives = []
    for start in (0, 2, 4):
        stop = start + 2
        batch_objectives.append(rw.weighted_bar_objective(
            w01[start:stop],
            w10[start:stop],
            df,
            log_ratio,
            log_weights_forward=logw01[start:stop],
            log_weights_reverse=logw10[start:stop],
            log_weight_normalizer_forward=norm01,
            log_weight_normalizer_reverse=norm10,
            population_size_forward=len(w01),
            population_size_reverse=len(w10),
        ))

    assert torch.allclose(torch.stack(batch_objectives).mean(), full, atol=1e-12, rtol=1e-12)


def test_global_weight_normalization_is_invariant_to_log_weight_shifts():
    work = torch.tensor([0.2, 0.5, 1.2], dtype=torch.double)
    logw = torch.tensor([-2.0, 0.0, 1.0], dtype=torch.double)
    normalizer = torch.logsumexp(logw, dim=0)

    weights0 = rw._torch_weights(
        logw[:2],
        work[:2],
        log_normalizer=normalizer,
        population_size=3,
    )
    weights1 = rw._torch_weights(
        logw[:2] + 1000.0,
        work[:2],
        log_normalizer=normalizer + 1000.0,
        population_size=3,
    )

    assert torch.allclose(weights0, weights1, atol=1e-12, rtol=1e-12)


def test_weighted_bar_robust_solver_matches_newton_on_well_behaved_data():
    w01 = torch.tensor([-0.3, 0.1, 0.4, 0.9, 1.3], dtype=torch.double)
    w10 = torch.tensor([-0.5, -0.1, 0.2, 0.6, 0.8], dtype=torch.double)
    logw01 = torch.tensor([0.0, 0.4, -0.2, 0.1, -0.5], dtype=torch.double)
    logw10 = torch.tensor([0.2, -0.3, 0.0, 0.5, -0.4], dtype=torch.double)
    weights01 = torch.exp(rw.normalize_log_weights_torch(logw01))
    weights10 = torch.exp(rw.normalize_log_weights_torch(logw10))
    log_ratio = torch.log((1.0 / torch.sum(weights10 * weights10)) / (1.0 / torch.sum(weights01 * weights01)))

    expected = rw.weighted_bar_newton_solve_detached(
        w01,
        w10,
        log_weights_forward=logw01,
        log_weights_reverse=logw10,
        log_ratio=log_ratio,
        max_iter=200,
        tol=1e-12,
    )
    got = rw.weighted_bar_robust_solve_detached(
        w01,
        w10,
        log_weights_forward=logw01,
        log_weights_reverse=logw10,
        log_ratio=log_ratio,
        max_iter=200,
        tol=1e-12,
    )

    assert got.converged
    assert not got.used_fallback
    assert torch.allclose(got.df, expected, atol=1e-10, rtol=1e-10)
    assert torch.isfinite(got.residual_abs)


def test_weighted_bar_robust_solver_falls_back_cleanly():
    w01 = torch.tensor([0.0, 0.4, 0.8, 1.1], dtype=torch.double)
    w10 = torch.tensor([-0.2, 0.2, 0.5, 0.9], dtype=torch.double)

    got = rw.weighted_bar_robust_solve_detached(
        w01,
        w10,
        df_init=torch.tensor(float("nan"), dtype=torch.double),
        max_iter=1,
        tol=1e-12,
    )

    assert got.used_fallback
    assert got.converged
    assert torch.isfinite(got.df)
    assert torch.isfinite(got.residual_abs)


def test_weighted_bar_robust_solver_handles_large_shifts():
    w01 = torch.tensor([1.0e6, 1.0e6 + 0.5, 1.0e6 + 1.0], dtype=torch.double)
    w10 = torch.tensor([-1.0e6 + 0.2, -1.0e6 + 0.7, -1.0e6 + 1.1], dtype=torch.double)
    logw01 = torch.tensor([1000.0, 1001.0, 999.0], dtype=torch.double)
    logw10 = torch.tensor([-1000.0, -999.5, -1001.0], dtype=torch.double)

    got = rw.weighted_bar_robust_solve_detached(
        w01,
        w10,
        log_weights_forward=logw01,
        log_weights_reverse=logw10,
        max_iter=1,
        tol=1e-10,
    )

    assert got.used_fallback
    assert got.converged
    assert torch.isfinite(got.df)
    assert torch.isfinite(got.residual_abs)


def test_weighted_bar_robust_solver_matches_unweighted_bar_for_zero_log_weights():
    w01 = torch.tensor([0.3, 0.7, 1.1, 0.2], dtype=torch.double)
    w10 = torch.tensor([-0.1, 0.4, 0.8, 1.2], dtype=torch.double)
    log_ratio = torch.log(torch.as_tensor(float(w10.numel()) / float(w01.numel()), dtype=torch.double))
    expected = barlib._bar_newton_solve_detached(
        w01,
        w10,
        log_ratio=log_ratio,
        max_iter=200,
        tol=1e-12,
    )

    got = rw.weighted_bar_robust_solve_detached(
        w01,
        w10,
        log_weights_forward=torch.zeros_like(w01),
        log_weights_reverse=torch.zeros_like(w10),
        log_ratio=log_ratio,
        max_iter=200,
        tol=1e-12,
    )

    assert got.converged
    assert torch.allclose(got.df, expected, atol=1e-10, rtol=1e-10)
