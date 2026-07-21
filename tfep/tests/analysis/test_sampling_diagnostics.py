"""Tests for sampling diagnostics used by TFEP/TBar holdout analyses."""
from __future__ import annotations

import numpy as np
import pandas as pd

from tfep.analysis.sampling import (
    build_split_masks,
    compute_cluster_coverage,
    compute_coverage_metrics,
    compute_sampling_risk,
    compute_timeseries_metrics,
    compute_work_cv_correlations,
)
from tfep.analysis.sampling.core import add_pca_and_clusters, shell_residence_metrics, split_consistency_row


def _synthetic_features() -> pd.DataFrame:
    n = 100
    frames = np.arange(n)
    slow = np.concatenate([np.zeros(80), np.ones(20) * 10.0])
    return pd.DataFrame(
        {
            "frame_index": frames,
            "slow_cv": slow,
            "fast_cv": np.sin(frames / 3.0),
            "nearest_water_resid": np.r_[np.repeat(1, 90), np.repeat(2, 10)],
        }
    )


def test_build_split_masks_and_consistency_flags_stale_split():
    frames = np.arange(10)
    train = np.array([0, 1, 2, 3, 4])
    val = np.array([5, 6, 7, 8, 9])
    masks = build_split_masks(frames, train, val)
    assert masks.train.sum() == 5
    assert masks.validation.sum() == 5

    ok = split_consistency_row(
        compound_id="m",
        leg="solv",
        seed=123,
        state="state0",
        n_samples=10,
        train_indices=train,
        validation_indices=val,
        expected_train_count=5,
    )
    assert ok["status"] == "ok"
    assert not ok["stale_split"]

    stale = split_consistency_row(
        compound_id="m",
        leg="solv",
        seed=123,
        state="state0",
        n_samples=10,
        train_indices=np.array([0, 1, 1, 12]),
        validation_indices=val,
        expected_train_count=5,
    )
    assert stale["status"] == "stale_or_incomplete"
    assert stale["stale_split"]
    assert stale["out_of_range_count"] == 1
    assert stale["duplicate_train_count"] == 1


def test_coverage_metrics_detect_missing_slow_mode_bins():
    df = _synthetic_features()
    # Training deliberately misses the last slow-state block.
    masks = build_split_masks(df["frame_index"], train_indices=np.arange(0, 60), validation_indices=np.arange(80, 100))
    rows = compute_coverage_metrics(df, masks, feature_columns=["slow_cv"])
    assert len(rows) == 1
    row = rows[0]
    assert row["ks_train_full"] > 0.15
    assert row["empty_full_bin_fraction_train"] > 0.0


def test_cluster_coverage_flags_cluster_absent_from_training():
    df = _synthetic_features()
    df["cluster"] = np.where(df["slow_cv"] > 1.0, 1, 0)
    masks = build_split_masks(df["frame_index"], train_indices=np.arange(0, 60), validation_indices=np.arange(80, 100))
    rows = compute_cluster_coverage(df, masks, min_full_fraction=0.05)
    cluster1 = [r for r in rows if r["cluster"] == 1][0]
    assert cluster1["important_cluster"]
    assert cluster1["absent_from_train"]
    assert not cluster1["absent_from_validation"]


def test_work_cv_correlations_find_injected_outlier_mode():
    df = _synthetic_features()
    work = pd.DataFrame(
        {
            "frame_index": np.arange(100),
            "raw_work": df["slow_cv"].to_numpy() * 0.2,
            "tfep_work": df["slow_cv"].to_numpy() * 1.5,
        }
    )
    work["work_shift"] = work["tfep_work"] - work["raw_work"]
    work["abs_work_shift"] = np.abs(work["work_shift"])
    rows = compute_work_cv_correlations(df, work, feature_columns=["slow_cv", "fast_cv"])
    assert rows[0]["feature"] == "slow_cv"
    assert rows[0]["target"] in {"tfep_work", "work_shift", "abs_work_shift", "raw_work"}
    assert rows[0]["abs_spearman_r"] > 0.8


def test_sampling_risk_combines_split_coverage_cluster_and_timeseries_flags():
    split = {"status": "ok", "stale_split": False}
    coverage = [{"ks_train_full": 0.45, "ks_validation_full": 0.1, "empty_full_bin_fraction_train": 0.3}]
    cluster = [{"absent_from_train": True, "absent_from_validation": False}]
    timeseries = [{"effective_sample_count": 40.0}]
    risk = compute_sampling_risk(
        split_row=split,
        coverage_rows=coverage,
        cluster_rows=cluster,
        timeseries_rows=timeseries,
    )
    assert risk["sampling_risk_label"] in {"high", "critical"}
    assert risk["sampling_risk_score"] >= 6.0


def test_timeseries_and_shell_residence_metrics_are_finite_for_synthetic_data():
    df = _synthetic_features()
    rows = compute_timeseries_metrics(df, feature_columns=["fast_cv"])
    assert rows
    assert np.isfinite(rows[0]["effective_sample_count"])

    shell = shell_residence_metrics(df)
    assert shell
    assert shell[0]["unique_nearest_waters"] == 2
    assert shell[0]["nearest_water_exchange_count"] == 1


def test_add_pca_and_clusters_preserves_shape():
    df = pd.DataFrame(
        {
            "frame_index": np.arange(30),
            "a": np.r_[np.zeros(15), np.ones(15)],
            "b": np.linspace(0.0, 1.0, 30),
            "c": np.sin(np.arange(30)),
        }
    )
    out, usable = add_pca_and_clusters(df, feature_columns=["a", "b", "c"], max_clusters=3)
    assert len(out) == len(df)
    assert {"pca1", "pca2", "cluster"}.issubset(out.columns)
    assert len(usable) >= 2
