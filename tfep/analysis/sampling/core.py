"""Core metrics for diagnosing TFEP/TBar sampling and split coverage.

The functions in this module are intentionally estimator-neutral. They inspect
trajectory-derived collective variables (CVs), saved train/validation indices,
and optional work arrays to flag sampling risks. They never modify TFEP work
arrays, split files, or training outputs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

try:  # sklearn is available in the project environment, but keep imports lazy-safe.
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover - exercised only in reduced dependency envs.
    KMeans = None  # type: ignore[assignment]
    PCA = None  # type: ignore[assignment]
    StandardScaler = None  # type: ignore[assignment]


@dataclass(frozen=True)
class SplitMasks:
    """Boolean train/validation masks aligned to a feature table."""

    train: np.ndarray
    validation: np.ndarray
    full: np.ndarray


def finite_float(value: Any, default: float = float("nan")) -> float:
    """Return a finite float or ``default`` for missing/non-finite values."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def write_csv_rows(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    """Write dictionaries to CSV, preserving first-seen field order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    pd.DataFrame(list(rows), columns=fieldnames).to_csv(path, index=False)


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric CV columns, excluding identifiers and bookkeeping fields."""
    excluded_prefixes = ("is_",)
    excluded = {
        "frame_index",
        "time_ps",
        "dataset_sample_index",
        "trajectory_sample_index",
        "state_index",
        "cluster",
        "nearest_water_resid",
        "nearest_water_slot",
        "raw_work",
        "tfep_work",
        "work_shift",
        "abs_work_shift",
        "abs_raw_work",
        "abs_tfep_work",
    }
    cols: list[str] = []
    for col in df.columns:
        if col in excluded or any(str(col).startswith(prefix) for prefix in excluded_prefixes):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            if np.isfinite(vals).sum() >= 3:
                cols.append(str(col))
    return cols


def build_split_masks(
    frame_indices: Sequence[int],
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
) -> SplitMasks:
    """Build masks for feature rows identified by trajectory frame index."""
    frames = np.asarray(frame_indices, dtype=int).reshape(-1)
    train_set = set(np.asarray(train_indices, dtype=int).reshape(-1).tolist())
    val_set = set(np.asarray(validation_indices, dtype=int).reshape(-1).tolist())
    train = np.asarray([int(i) in train_set for i in frames], dtype=bool)
    validation = np.asarray([int(i) in val_set for i in frames], dtype=bool)
    return SplitMasks(train=train, validation=validation, full=np.ones(frames.shape[0], dtype=bool))


def split_consistency_row(
    *,
    compound_id: str,
    molecule: str | None = None,
    leg: str,
    seed: int,
    state: str,
    n_samples: int,
    train_indices: Sequence[int] | None,
    validation_indices: Sequence[int] | None,
    expected_train_count: int | None = None,
) -> dict[str, Any]:
    """Check saved split indices for duplicates, overlap, range, and expected counts."""
    row: dict[str, Any] = {
        "compound_id": compound_id,
        "molecule": compound_id if molecule is None else molecule,
        "leg": leg,
        "seed": int(seed),
        "state": state,
        "n_samples": int(n_samples),
        "expected_train_count": "" if expected_train_count is None else int(expected_train_count),
        "status": "ok",
    }
    if train_indices is None or validation_indices is None:
        row.update(
            train_count=0,
            validation_count=0,
            duplicate_train_count=0,
            duplicate_validation_count=0,
            overlap_count=0,
            out_of_range_count=0,
            covered_count=0,
            missing_count=int(n_samples),
            stale_split=False,
            status="missing_split_indices",
        )
        return row

    train = np.asarray(train_indices, dtype=int).reshape(-1)
    val = np.asarray(validation_indices, dtype=int).reshape(-1)
    train_unique = np.unique(train)
    val_unique = np.unique(val)
    out_of_range = int(np.sum((train < 0) | (train >= n_samples)) + np.sum((val < 0) | (val >= n_samples)))
    overlap = int(np.intersect1d(train_unique, val_unique).size)
    covered = int(np.union1d(train_unique, val_unique).size)
    missing = int(max(0, n_samples - covered))
    stale = False
    if expected_train_count is not None and int(train.size) != int(expected_train_count):
        stale = True
    if covered != int(n_samples) or overlap != 0 or out_of_range != 0:
        stale = True

    row.update(
        train_count=int(train.size),
        validation_count=int(val.size),
        duplicate_train_count=int(train.size - train_unique.size),
        duplicate_validation_count=int(val.size - val_unique.size),
        overlap_count=overlap,
        out_of_range_count=out_of_range,
        covered_count=covered,
        missing_count=missing,
        stale_split=bool(stale),
        status="stale_or_incomplete" if stale else "ok",
    )
    return row


def _safe_ks(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return float("nan")
    return float(stats.ks_2samp(a, b, mode="auto").statistic)


def _safe_wasserstein(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 1 or b.size < 1:
        return float("nan")
    return float(stats.wasserstein_distance(a, b))


def _empty_full_bin_fraction(full: np.ndarray, subset: np.ndarray, n_bins: int = 10) -> float:
    full = full[np.isfinite(full)]
    subset = subset[np.isfinite(subset)]
    if full.size < 3 or subset.size < 1:
        return float("nan")
    quantiles = np.linspace(0.0, 1.0, int(n_bins) + 1)
    edges = np.unique(np.quantile(full, quantiles))
    if edges.size < 3:
        return 0.0
    full_counts, _ = np.histogram(full, bins=edges)
    subset_counts, _ = np.histogram(subset, bins=edges)
    relevant = full_counts > 0
    if not np.any(relevant):
        return float("nan")
    return float(np.mean(subset_counts[relevant] == 0))


def compute_coverage_metrics(
    df: pd.DataFrame,
    masks: SplitMasks,
    *,
    feature_columns: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Compare train/validation/full distributions for each numeric CV."""
    feature_columns = list(feature_columns or numeric_feature_columns(df))
    meta = dict(metadata or {})
    rows: list[dict[str, Any]] = []
    for col in feature_columns:
        values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        full = values[masks.full & np.isfinite(values)]
        train = values[masks.train & np.isfinite(values)]
        val = values[masks.validation & np.isfinite(values)]
        if full.size < 3:
            continue
        row = {
            **meta,
            "feature": col,
            "n_full": int(full.size),
            "n_train": int(train.size),
            "n_validation": int(val.size),
            "full_mean": float(np.mean(full)),
            "train_mean": float(np.mean(train)) if train.size else float("nan"),
            "validation_mean": float(np.mean(val)) if val.size else float("nan"),
            "full_std": float(np.std(full, ddof=1)) if full.size > 1 else float("nan"),
            "train_std": float(np.std(train, ddof=1)) if train.size > 1 else float("nan"),
            "validation_std": float(np.std(val, ddof=1)) if val.size > 1 else float("nan"),
            "ks_train_full": _safe_ks(train, full),
            "ks_validation_full": _safe_ks(val, full),
            "ks_train_validation": _safe_ks(train, val),
            "wasserstein_train_full": _safe_wasserstein(train, full),
            "wasserstein_validation_full": _safe_wasserstein(val, full),
            "empty_full_bin_fraction_train": _empty_full_bin_fraction(full, train),
            "empty_full_bin_fraction_validation": _empty_full_bin_fraction(full, val),
        }
        rows.append(row)
    return rows


def _autocorrelation_1d(values: np.ndarray, max_lag: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 3:
        return np.asarray([], dtype=float)
    values = values - float(np.mean(values))
    denom = float(np.dot(values, values))
    if denom <= 0.0:
        return np.zeros(min(max_lag, values.size - 1), dtype=float)
    lags = min(int(max_lag), values.size - 1)
    return np.asarray([float(np.dot(values[:-lag], values[lag:]) / denom) for lag in range(1, lags + 1)], dtype=float)


def statistical_inefficiency(values: Sequence[float], max_lag: int = 200) -> tuple[float, float, float]:
    """Estimate statistical inefficiency using positive initial ACF terms."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 3:
        return float("nan"), float("nan"), float("nan")
    acf = _autocorrelation_1d(arr, max_lag=max_lag)
    positive_terms: list[float] = []
    for val in acf:
        if not math.isfinite(float(val)) or float(val) <= 0.0:
            break
        positive_terms.append(float(val))
    g = 1.0 + 2.0 * float(np.sum(positive_terms)) if positive_terms else 1.0
    g = max(1.0, min(g, float(arr.size)))
    neff = float(arr.size) / g
    acf1 = float(acf[0]) if acf.size else float("nan")
    return g, neff, acf1


def compute_timeseries_metrics(
    df: pd.DataFrame,
    *,
    feature_columns: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
    max_lag: int = 200,
) -> list[dict[str, Any]]:
    """Compute drift/autocorrelation summaries for ordered CV time series."""
    feature_columns = list(feature_columns or numeric_feature_columns(df))
    meta = dict(metadata or {})
    rows: list[dict[str, Any]] = []
    ordered = df.sort_values("frame_index") if "frame_index" in df.columns else df
    x = np.arange(len(ordered), dtype=float)
    for col in feature_columns:
        values = pd.to_numeric(ordered[col], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(values)
        if finite.sum() < 3:
            continue
        g, neff, acf1 = statistical_inefficiency(values[finite], max_lag=max_lag)
        slope = float(np.polyfit(x[finite], values[finite], deg=1)[0]) if finite.sum() > 2 else float("nan")
        rows.append(
            {
                **meta,
                "feature": col,
                "n": int(finite.sum()),
                "mean": float(np.mean(values[finite])),
                "std": float(np.std(values[finite], ddof=1)) if finite.sum() > 1 else float("nan"),
                "acf_lag1": acf1,
                "statistical_inefficiency": g,
                "effective_sample_count": neff,
                "linear_drift_slope_per_frame": slope,
            }
        )
    return rows


def add_pca_and_clusters(
    df: pd.DataFrame,
    *,
    feature_columns: Sequence[str] | None = None,
    max_clusters: int = 6,
    random_state: int = 0,
) -> tuple[pd.DataFrame, list[str]]:
    """Return a copy of ``df`` with ``pca1``, ``pca2``, and ``cluster`` columns."""
    out = df.copy()
    if PCA is None or StandardScaler is None or KMeans is None:
        out["pca1"] = np.nan
        out["pca2"] = np.nan
        out["cluster"] = -1
        return out, []
    cols = list(feature_columns or numeric_feature_columns(out))
    usable: list[str] = []
    arrays: list[np.ndarray] = []
    for col in cols:
        values = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(values)
        if finite.sum() >= 5 and np.nanstd(values) > 1.0e-12:
            med = float(np.nanmedian(values[finite]))
            values = np.where(finite, values, med)
            usable.append(col)
            arrays.append(values)
    if len(usable) < 2 or len(out) < 4:
        out["pca1"] = np.nan
        out["pca2"] = np.nan
        out["cluster"] = -1
        return out, usable
    x = np.vstack(arrays).T
    x_scaled = StandardScaler().fit_transform(x)
    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(x_scaled)
    n_clusters = int(max(2, min(max_clusters, np.sqrt(max(2, len(out) // 10)))))
    n_clusters = min(n_clusters, len(out))
    labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state).fit_predict(x_scaled)
    out["pca1"] = coords[:, 0]
    out["pca2"] = coords[:, 1]
    out["cluster"] = labels.astype(int)
    return out, usable


def compute_cluster_coverage(
    df: pd.DataFrame,
    masks: SplitMasks,
    *,
    metadata: Mapping[str, Any] | None = None,
    min_full_fraction: float = 0.02,
) -> list[dict[str, Any]]:
    """Summarize train/validation occupancy for each cluster."""
    if "cluster" not in df.columns:
        return []
    meta = dict(metadata or {})
    labels = pd.to_numeric(df["cluster"], errors="coerce").fillna(-1).to_numpy(dtype=int)
    valid = labels >= 0
    n_full_total = int(np.sum(valid))
    rows: list[dict[str, Any]] = []
    for label in sorted(int(x) for x in np.unique(labels[valid])):
        cluster_mask = valid & (labels == label)
        n_full = int(np.sum(cluster_mask))
        n_train = int(np.sum(cluster_mask & masks.train))
        n_val = int(np.sum(cluster_mask & masks.validation))
        frac_full = float(n_full / n_full_total) if n_full_total else float("nan")
        important = bool(math.isfinite(frac_full) and frac_full >= float(min_full_fraction))
        rows.append(
            {
                **meta,
                "cluster": int(label),
                "n_full": n_full,
                "n_train": n_train,
                "n_validation": n_val,
                "fraction_full": frac_full,
                "fraction_train_within_cluster": float(n_train / n_full) if n_full else float("nan"),
                "fraction_validation_within_cluster": float(n_val / n_full) if n_full else float("nan"),
                "important_cluster": important,
                "absent_from_train": bool(important and n_train == 0),
                "absent_from_validation": bool(important and n_val == 0),
            }
        )
    return rows


def compute_work_cv_correlations(
    features: pd.DataFrame,
    work_table: pd.DataFrame,
    *,
    metadata: Mapping[str, Any] | None = None,
    feature_columns: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Compute Spearman correlations between validation work terms and CVs."""
    if work_table.empty or "frame_index" not in features.columns or "frame_index" not in work_table.columns:
        return []
    merged = work_table.merge(features, on="frame_index", how="inner", suffixes=("", "_cv"))
    if merged.empty:
        return []
    meta = dict(metadata or {})
    cols = list(feature_columns or numeric_feature_columns(features))
    targets = [c for c in ("raw_work", "tfep_work", "work_shift", "abs_work_shift") if c in merged.columns]
    rows: list[dict[str, Any]] = []
    for target in targets:
        y = pd.to_numeric(merged[target], errors="coerce").to_numpy(dtype=float)
        for col in cols:
            x = pd.to_numeric(merged[col], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 8 or np.nanstd(x[mask]) <= 1.0e-12 or np.nanstd(y[mask]) <= 1.0e-12:
                continue
            corr, pval = stats.spearmanr(x[mask], y[mask])
            rows.append(
                {
                    **meta,
                    "target": target,
                    "feature": col,
                    "n": int(mask.sum()),
                    "spearman_r": float(corr),
                    "spearman_pvalue": float(pval),
                    "abs_spearman_r": float(abs(corr)),
                }
            )
    rows.sort(key=lambda r: finite_float(r.get("abs_spearman_r"), 0.0), reverse=True)
    return rows


def shell_residence_metrics(df: pd.DataFrame, *, metadata: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Summarize nearest-water identity persistence when shell features exist."""
    if "nearest_water_resid" not in df.columns:
        return []
    meta = dict(metadata or {})
    ordered = df.sort_values("frame_index") if "frame_index" in df.columns else df
    ids = pd.to_numeric(ordered["nearest_water_resid"], errors="coerce").to_numpy(dtype=float)
    ids = ids[np.isfinite(ids)].astype(int)
    if ids.size == 0:
        return []
    transitions = int(np.sum(ids[1:] != ids[:-1])) if ids.size > 1 else 0
    unique, counts = np.unique(ids, return_counts=True)
    probs = counts.astype(float) / float(np.sum(counts))
    entropy = float(-np.sum(probs * np.log(np.maximum(probs, 1.0e-300))))
    return [
        {
            **meta,
            "feature": "nearest_water_resid",
            "n": int(ids.size),
            "unique_nearest_waters": int(unique.size),
            "nearest_water_exchange_count": transitions,
            "nearest_water_exchange_rate": float(transitions / max(1, ids.size - 1)),
            "nearest_water_identity_entropy": entropy,
            "dominant_nearest_water_fraction": float(np.max(probs)),
        }
    ]


def compute_sampling_risk(
    *,
    split_row: Mapping[str, Any],
    coverage_rows: Sequence[Mapping[str, Any]],
    cluster_rows: Sequence[Mapping[str, Any]],
    timeseries_rows: Sequence[Mapping[str, Any]],
    work_corr_rows: Sequence[Mapping[str, Any]] = (),
    shell_rows: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Assign a practical sampling-risk label from diagnostic summaries."""
    score = 0.0
    reasons: list[str] = []

    if str(split_row.get("status", "ok")) != "ok" or bool(split_row.get("stale_split", False)):
        score += 3.0
        reasons.append("split indices are stale, incomplete, overlapping, or inconsistent")

    max_ks = 0.0
    max_empty_train = 0.0
    max_empty_val = 0.0
    for row in coverage_rows:
        max_ks = max(max_ks, finite_float(row.get("ks_train_full"), 0.0), finite_float(row.get("ks_validation_full"), 0.0))
        max_empty_train = max(max_empty_train, finite_float(row.get("empty_full_bin_fraction_train"), 0.0))
        max_empty_val = max(max_empty_val, finite_float(row.get("empty_full_bin_fraction_validation"), 0.0))
    if max_ks > 0.40:
        score += 2.0
        reasons.append(f"large train/validation-vs-full CV mismatch (max KS={max_ks:.3f})")
    elif max_ks > 0.20:
        score += 1.0
        reasons.append(f"moderate train/validation-vs-full CV mismatch (max KS={max_ks:.3f})")
    if max(max_empty_train, max_empty_val) > 0.20:
        score += 1.0
        reasons.append("some full-distribution CV bins are absent from train or validation")

    absent_train = [r for r in cluster_rows if bool(r.get("absent_from_train", False))]
    absent_val = [r for r in cluster_rows if bool(r.get("absent_from_validation", False))]
    if absent_train:
        score += 2.0
        reasons.append(f"{len(absent_train)} important cluster(s) absent from training")
    if absent_val:
        score += 1.0
        reasons.append(f"{len(absent_val)} important cluster(s) absent from validation")

    neffs = [finite_float(r.get("effective_sample_count")) for r in timeseries_rows]
    neffs = [x for x in neffs if math.isfinite(x)]
    min_neff = min(neffs) if neffs else float("nan")
    if math.isfinite(min_neff) and min_neff < 50:
        score += 2.0
        reasons.append(f"low effective sample count for at least one CV (min Neff={min_neff:.1f})")
    elif math.isfinite(min_neff) and min_neff < 100:
        score += 1.0
        reasons.append(f"moderate autocorrelation for at least one CV (min Neff={min_neff:.1f})")

    max_work_corr = max([finite_float(r.get("abs_spearman_r"), 0.0) for r in work_corr_rows] or [0.0])
    if max_work_corr > 0.6:
        score += 1.0
        reasons.append(f"work outliers strongly correlate with a CV (max |rho|={max_work_corr:.3f})")

    for row in shell_rows:
        exchange_rate = finite_float(row.get("nearest_water_exchange_rate"))
        dominant = finite_float(row.get("dominant_nearest_water_fraction"))
        if math.isfinite(exchange_rate) and exchange_rate < 0.002:
            score += 1.0
            reasons.append("nearest shell-water identity changes very slowly")
            break
        if math.isfinite(dominant) and dominant > 0.5:
            score += 0.5
            reasons.append("nearest shell-water distribution is dominated by one identity")
            break

    if score >= 6.0:
        label = "critical"
    elif score >= 4.0:
        label = "high"
    elif score >= 2.0:
        label = "moderate"
    else:
        label = "low"
    return {
        "sampling_risk_score": float(score),
        "sampling_risk_label": label,
        "risk_reasons": "; ".join(reasons) if reasons else "no major sampling-risk flags",
        "max_ks_train_or_validation_vs_full": float(max_ks),
        "max_empty_full_bin_fraction": float(max(max_empty_train, max_empty_val)),
        "min_effective_sample_count": float(min_neff),
        "max_abs_work_cv_spearman": float(max_work_corr),
    }
