#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt


COLORS = {
    "raw": "#1f77b4",
    "tfep": "#d95f02",
    "snf": "#009E73",
    "forward": "#2ca02c",
    "reverse": "#9467bd",
    "neutral": "#444444",
}


def has_core_analysis_files(path: Path) -> bool:
    return (path / "summary.json").exists() and (path / "convergence.csv").exists()


def resolve_analysis_dir(outdir: Path, analysis_subdir: str, explicit_analysis_dir: Optional[Path]) -> Path:
    if explicit_analysis_dir is not None:
        return explicit_analysis_dir.resolve()

    requested = (outdir / analysis_subdir).resolve()
    if has_core_analysis_files(requested):
        return requested

    holdout = (outdir / "holdout_small_train").resolve()
    if analysis_subdir == "cv_bootstrap" and has_core_analysis_files(holdout):
        return holdout

    return requested


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--outdir", type=Path, default=Path("tfep_tmbar_openmm_cv"),
                    help="Root output directory used by bromomethane_tfep_cv_bootstrap.py")
    ap.add_argument("--analysis-subdir", type=str, default="cv_bootstrap",
                    help="Analysis subdirectory created by the CV/bootstrap workflow")
    ap.add_argument("--analysis-dir", type=Path, default=None,
                    help="Use this analysis directory directly instead of <outdir>/<analysis-subdir>")
    ap.add_argument("--plots", type=Path, default=None,
                    help="Directory where diagnostic plots will be written")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--max-scatter-points", type=int, default=8000,
                    help="Maximum number of scatter points plotted per work-vs-index series")
    return ap


def savefig(fig, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _maybe_float(value: Any) -> Any:
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip()
    if text == "":
        return np.nan
    low = text.lower()
    if low == "nan":
        return np.nan
    if low == "inf":
        return np.inf
    if low == "-inf":
        return -np.inf
    try:
        return float(text)
    except ValueError:
        return value


def load_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    rows: List[Dict[str, Any]] = []
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({key: _maybe_float(value) for key, value in row.items()})
    return rows


def latest_metrics_csv(fold_dir: Path) -> Optional[Path]:
    candidates = sorted((fold_dir / "train" / "logs" / "tfep").glob("version_*/metrics.csv"))
    if not candidates:
        return None
    return candidates[-1]


def select_fold_dirs(analysis_dir: Path, allowed_fold_ids: Optional[Sequence[int]] = None) -> List[Path]:
    fold_dirs = sorted((analysis_dir / "folds").glob("fold_*"))
    if allowed_fold_ids is None:
        return fold_dirs
    allowed = {int(fold_id) for fold_id in allowed_fold_ids}
    selected: List[Path] = []
    for fold_dir in fold_dirs:
        try:
            fold_id = int(fold_dir.name.split("_")[-1])
        except ValueError:
            continue
        if fold_id in allowed:
            selected.append(fold_dir)
    return selected


def evenly_spaced_subset(x: np.ndarray, y: np.ndarray, max_points: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(x) <= max_points:
        return x, y
    keep = np.linspace(0, len(x) - 1, int(max_points), dtype=int)
    return x[keep], y[keep]


def finite_or_nan(values: Sequence[Any]) -> np.ndarray:
    return np.array([_maybe_float(v) for v in values], dtype=float)


def finite_values(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def has_finite(values: Sequence[Any]) -> bool:
    return bool(np.any(np.isfinite(finite_or_nan(values))))


def _safe_ratio(num: Any, den: Any) -> float:
    num_f = _maybe_float(num)
    den_f = _maybe_float(den)
    if not np.isfinite(num_f) or not np.isfinite(den_f) or float(den_f) == 0.0:
        return float("nan")
    return float(num_f / den_f)


def write_csv_rows(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    if len(rows) == 0:
        path.write_text("")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_fold_comparison_rows(fold_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in fold_rows:
        fold_id = int(_maybe_float(row.get("fold", len(out))))
        raw_deltaf = _maybe_float(row.get("raw_deltaf"))
        tfep_deltaf = _maybe_float(row.get("tfep_deltaf"))
        snf_deltaf = _maybe_float(row.get("snf_deltaf"))
        shift = tfep_deltaf - raw_deltaf
        abs_shift = abs(shift) if np.isfinite(shift) else float("nan")
        snf_shift = snf_deltaf - raw_deltaf
        snf_abs_shift = abs(snf_shift) if np.isfinite(snf_shift) else float("nan")
        snf_minus_tfep = snf_deltaf - tfep_deltaf

        raw_boot = _maybe_float(row.get("raw_bootstrap_std"))
        tfep_boot = _maybe_float(row.get("tfep_bootstrap_std"))
        snf_boot = _maybe_float(row.get("snf_bootstrap_std"))
        combined_unc = float(np.sqrt(raw_boot ** 2 + tfep_boot ** 2)) if np.isfinite(raw_boot) and np.isfinite(tfep_boot) else float("nan")
        shift_z = abs_shift / combined_unc if np.isfinite(abs_shift) and np.isfinite(combined_unc) and combined_unc > 0.0 else float("nan")

        raw_overlap = _maybe_float(row.get("raw_overlap"))
        tfep_overlap = _maybe_float(row.get("tfep_overlap"))
        snf_overlap = _maybe_float(row.get("snf_overlap"))
        raw_direct_overlap = _maybe_float(row.get("raw_direct_overlap"))
        tfep_direct_overlap = _maybe_float(row.get("tfep_direct_overlap"))
        snf_direct_overlap = _maybe_float(row.get("snf_direct_overlap"))
        overlap_gain = tfep_overlap - raw_overlap
        snf_overlap_gain = snf_overlap - raw_overlap
        direct_overlap_gain = tfep_direct_overlap - raw_direct_overlap
        snf_direct_overlap_gain = snf_direct_overlap - raw_direct_overlap

        sigma_ratio = _safe_ratio(row.get("tfep_sigma"), row.get("raw_sigma"))
        snf_sigma_ratio = _safe_ratio(row.get("snf_sigma"), row.get("raw_sigma"))
        boot_ratio = _safe_ratio(tfep_boot, raw_boot)
        snf_boot_ratio = _safe_ratio(snf_boot, raw_boot)
        forward_ratio = _safe_ratio(row.get("tfep_forward_std"), row.get("raw_forward_std"))
        reverse_ratio = _safe_ratio(row.get("tfep_reverse_std"), row.get("raw_reverse_std"))
        snf_forward_ratio = _safe_ratio(row.get("snf_forward_std"), row.get("raw_forward_std"))
        snf_reverse_ratio = _safe_ratio(row.get("snf_reverse_std"), row.get("raw_reverse_std"))

        score = float(abs_shift) if np.isfinite(abs_shift) else 1e6
        if np.isfinite(boot_ratio):
            score += 0.6 * max(0.0, boot_ratio - 1.0)
        else:
            score += 1.0
        if np.isfinite(sigma_ratio):
            score += 0.3 * max(0.0, sigma_ratio - 1.0)
        else:
            score += 0.5
        if np.isfinite(overlap_gain):
            score += 0.1 * max(0.0, -overlap_gain)
        else:
            score += 0.2

        out.append(
            {
                "fold": int(fold_id),
                "raw_deltaf": float(raw_deltaf),
                "tfep_deltaf": float(tfep_deltaf),
                "snf_deltaf": float(snf_deltaf),
                "deltaf_shift_tfep_minus_raw": float(shift),
                "abs_deltaf_shift": float(abs_shift),
                "deltaf_shift_snf_minus_raw": float(snf_shift),
                "abs_snf_deltaf_shift": float(snf_abs_shift),
                "deltaf_shift_snf_minus_tfep": float(snf_minus_tfep),
                "raw_bootstrap_std": float(raw_boot),
                "tfep_bootstrap_std": float(tfep_boot),
                "snf_bootstrap_std": float(snf_boot),
                "bootstrap_std_ratio_tfep_over_raw": float(boot_ratio),
                "bootstrap_std_ratio_snf_over_raw": float(snf_boot_ratio),
                "bootstrap_std_gain_raw_minus_tfep": float(raw_boot - tfep_boot) if np.isfinite(raw_boot) and np.isfinite(tfep_boot) else float("nan"),
                "bootstrap_std_gain_raw_minus_snf": float(raw_boot - snf_boot) if np.isfinite(raw_boot) and np.isfinite(snf_boot) else float("nan"),
                "raw_sigma": float(_maybe_float(row.get("raw_sigma"))),
                "tfep_sigma": float(_maybe_float(row.get("tfep_sigma"))),
                "snf_sigma": float(_maybe_float(row.get("snf_sigma"))),
                "sigma_ratio_tfep_over_raw": float(sigma_ratio),
                "sigma_ratio_snf_over_raw": float(snf_sigma_ratio),
                "raw_overlap": float(raw_overlap),
                "tfep_overlap": float(tfep_overlap),
                "snf_overlap": float(snf_overlap),
                "overlap_gain_tfep_minus_raw": float(overlap_gain),
                "overlap_gain_snf_minus_raw": float(snf_overlap_gain),
                "raw_direct_overlap": float(raw_direct_overlap),
                "tfep_direct_overlap": float(tfep_direct_overlap),
                "snf_direct_overlap": float(snf_direct_overlap),
                "direct_overlap_gain_tfep_minus_raw": float(direct_overlap_gain),
                "direct_overlap_gain_snf_minus_raw": float(snf_direct_overlap_gain),
                "forward_std_ratio_tfep_over_raw": float(forward_ratio),
                "reverse_std_ratio_tfep_over_raw": float(reverse_ratio),
                "forward_std_ratio_snf_over_raw": float(snf_forward_ratio),
                "reverse_std_ratio_snf_over_raw": float(snf_reverse_ratio),
                "combined_deltaf_uncertainty": float(combined_unc),
                "abs_shift_over_combined_uncertainty": float(shift_z),
                "comparison_score_smaller_is_better": float(score),
            }
        )

    out.sort(key=lambda item: int(item["fold"]))
    rank_order = sorted(out, key=lambda item: (_maybe_float(item["comparison_score_smaller_is_better"]), int(item["fold"])))
    for rank, item in enumerate(rank_order, start=1):
        item["rank_smaller_is_better"] = int(rank)
    return out


def build_fold_comparison_summary(comp_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not comp_rows:
        return {"n_folds": 0}

    abs_shift = finite_or_nan([row.get("abs_deltaf_shift") for row in comp_rows])
    boot_ratio = finite_or_nan([row.get("bootstrap_std_ratio_tfep_over_raw") for row in comp_rows])
    snf_boot_ratio = finite_or_nan([row.get("bootstrap_std_ratio_snf_over_raw") for row in comp_rows])
    sigma_ratio = finite_or_nan([row.get("sigma_ratio_tfep_over_raw") for row in comp_rows])
    snf_sigma_ratio = finite_or_nan([row.get("sigma_ratio_snf_over_raw") for row in comp_rows])
    overlap_gain = finite_or_nan([row.get("overlap_gain_tfep_minus_raw") for row in comp_rows])
    snf_overlap_gain = finite_or_nan([row.get("overlap_gain_snf_minus_raw") for row in comp_rows])
    direct_gain = finite_or_nan([row.get("direct_overlap_gain_tfep_minus_raw") for row in comp_rows])
    snf_direct_gain = finite_or_nan([row.get("direct_overlap_gain_snf_minus_raw") for row in comp_rows])
    snf_abs_shift = finite_or_nan([row.get("abs_snf_deltaf_shift") for row in comp_rows])

    top_ranked = sorted(comp_rows, key=lambda row: (_maybe_float(row.get("rank_smaller_is_better")), int(row.get("fold", 0))))[:5]
    top_ranked_export = [
        {
            "fold": int(row["fold"]),
            "rank": int(row["rank_smaller_is_better"]),
            "score": float(row["comparison_score_smaller_is_better"]),
            "abs_deltaf_shift": float(row["abs_deltaf_shift"]),
            "bootstrap_std_ratio_tfep_over_raw": float(row["bootstrap_std_ratio_tfep_over_raw"]),
            "overlap_gain_tfep_minus_raw": float(row["overlap_gain_tfep_minus_raw"]),
        }
        for row in top_ranked
    ]

    return {
        "n_folds": int(len(comp_rows)),
        "mean_abs_deltaf_shift": float(np.nanmean(abs_shift)),
        "median_abs_deltaf_shift": float(np.nanmedian(abs_shift)),
        "mean_abs_snf_deltaf_shift": float(np.nanmean(snf_abs_shift)),
        "median_abs_snf_deltaf_shift": float(np.nanmedian(snf_abs_shift)),
        "mean_bootstrap_std_ratio_tfep_over_raw": float(np.nanmean(boot_ratio)),
        "mean_bootstrap_std_ratio_snf_over_raw": float(np.nanmean(snf_boot_ratio)),
        "mean_sigma_ratio_tfep_over_raw": float(np.nanmean(sigma_ratio)),
        "mean_sigma_ratio_snf_over_raw": float(np.nanmean(snf_sigma_ratio)),
        "mean_overlap_gain_tfep_minus_raw": float(np.nanmean(overlap_gain)),
        "mean_overlap_gain_snf_minus_raw": float(np.nanmean(snf_overlap_gain)),
        "mean_direct_overlap_gain_tfep_minus_raw": float(np.nanmean(direct_gain)),
        "mean_direct_overlap_gain_snf_minus_raw": float(np.nanmean(snf_direct_gain)),
        "fraction_folds_bootstrap_ratio_below_1": float(np.nanmean(boot_ratio < 1.0)),
        "fraction_folds_snf_bootstrap_ratio_below_1": float(np.nanmean(snf_boot_ratio < 1.0)),
        "fraction_folds_sigma_ratio_below_1": float(np.nanmean(sigma_ratio < 1.0)),
        "fraction_folds_snf_sigma_ratio_below_1": float(np.nanmean(snf_sigma_ratio < 1.0)),
        "fraction_folds_overlap_gain_positive": float(np.nanmean(overlap_gain > 0.0)),
        "fraction_folds_snf_overlap_gain_positive": float(np.nanmean(snf_overlap_gain > 0.0)),
        "fraction_folds_direct_overlap_gain_positive": float(np.nanmean(direct_gain > 0.0)),
        "fraction_folds_snf_direct_overlap_gain_positive": float(np.nanmean(snf_direct_gain > 0.0)),
        "top_ranked_folds": top_ranked_export,
    }


def extract_summary_views(summary: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], str, str]:
    if "pooled" in summary:
        pooled = summary.get("pooled", {})
        comparison = summary.get("comparison", {})
        return (
            pooled.get("raw_full", {}),
            pooled.get("tfep_oof", {}),
            comparison,
            "CV/Bootstrap Summary",
            "tfep_minus_raw_full_deltaf",
        )
    if "held_out" in summary:
        held_out = summary.get("held_out", {})
        comparison = dict(summary.get("comparison", {}))
        use_reweighted = (
            bool(summary.get("reweighting", {}).get("enabled"))
            and isinstance(held_out.get("raw_reweighted"), dict)
            and isinstance(held_out.get("tfep_reweighted"), dict)
        )
        if use_reweighted:
            raw = held_out.get("raw_reweighted", {})
            tfep = held_out.get("tfep_reweighted", {})
            if np.isfinite(_maybe_float(raw.get("deltaf"))) and np.isfinite(_maybe_float(tfep.get("deltaf"))):
                comparison["tfep_minus_raw_validation_deltaf"] = float(tfep["deltaf"]) - float(raw["deltaf"])
            return (
                raw,
                tfep,
                comparison,
                "Holdout Summary (reweighted)",
                "tfep_minus_raw_validation_deltaf",
            )
        return (
            held_out.get("raw", {}),
            held_out.get("tfep", {}),
            comparison,
            "Holdout Summary",
            "tfep_minus_raw_validation_deltaf",
        )
    return {}, {}, {}, "TFEP Summary", "tfep_minus_raw_full_deltaf"


def plot_text_summary(summary: Dict[str, Any], out_path: Path, dpi: int) -> None:
    raw, tfep, comparison, title, shift_key = extract_summary_views(summary)
    held_or_pooled = summary.get("held_out", summary.get("pooled", {}))
    snf = held_or_pooled.get("stochastic_path_tfep", {})
    snf_shift_key = "snf_minus_raw_validation_deltaf" if "held_out" in summary else "snf_minus_raw_full_deltaf"
    snf_vs_tfep_key = "snf_minus_tfep_validation_deltaf" if "held_out" in summary else "snf_minus_tfep_full_deltaf"

    lines = [
        f"raw BAR deltaf: {raw.get('deltaf', np.nan):.6f} kT",
        f"raw BAR sigma: {raw.get('sigma', np.nan)}",
        f"raw BAR overlap: {raw.get('overlap', np.nan):.4f}",
        f"raw direct overlap: {raw.get('direct_overlap', np.nan):.4f}",
        "",
        f"TFEP BAR deltaf: {tfep.get('deltaf', np.nan):.6f} kT",
        f"TFEP BAR sigma: {tfep.get('sigma', np.nan)}",
        f"TFEP BAR overlap: {tfep.get('overlap', np.nan):.4f}",
        f"TFEP direct overlap: {tfep.get('direct_overlap', np.nan):.4f}",
        "",
        f"TFEP - raw reference deltaf: {comparison.get(shift_key, np.nan):.6f} kT",
        f"mean sigma ratio TFEP/raw: {comparison.get('mean_sigma_ratio_tfep_over_raw', np.nan)}",
        f"mean forward std ratio TFEP/raw: {comparison.get('mean_forward_std_ratio_tfep_over_raw', np.nan)}",
        f"mean reverse std ratio TFEP/raw: {comparison.get('mean_reverse_std_ratio_tfep_over_raw', np.nan)}",
    ]
    if snf:
        lines.extend(
            [
                "",
                f"SNF/path BAR deltaf: {snf.get('deltaf', np.nan):.6f} kT",
                f"SNF/path BAR sigma: {snf.get('sigma', np.nan)}",
                f"SNF/path BAR overlap: {snf.get('overlap', np.nan):.4f}",
                f"SNF/path direct overlap: {snf.get('direct_overlap', np.nan):.4f}",
                f"SNF - raw reference deltaf: {comparison.get(snf_shift_key, np.nan):.6f} kT",
                f"SNF - deterministic TFEP deltaf: {comparison.get(snf_vs_tfep_key, np.nan):.6f} kT",
            ]
        )

    fig = plt.figure(figsize=(8, 6 if snf else 5))
    plt.axis("off")
    plt.text(0.03, 0.97, title, va="top", family="monospace", fontsize=12)
    plt.text(0.03, 0.90, "\n".join(lines), va="top", family="monospace", fontsize=10)
    savefig(fig, out_path, dpi)


def plot_fold_deltaf(fold_rows: List[Dict[str, Any]], out_path: Path, dpi: int) -> None:
    if not fold_rows:
        return
    fold_ids = np.array([int(row["fold"]) for row in fold_rows], dtype=int)
    raw_deltaf = finite_or_nan([row.get("raw_deltaf") for row in fold_rows])
    tfep_deltaf = finite_or_nan([row.get("tfep_deltaf") for row in fold_rows])
    snf_deltaf = finite_or_nan([row.get("snf_deltaf") for row in fold_rows])

    raw_err = finite_or_nan([
        row.get("raw_sigma") if np.isfinite(_maybe_float(row.get("raw_sigma"))) else row.get("raw_bootstrap_std")
        for row in fold_rows
    ])
    tfep_err = finite_or_nan([
        row.get("tfep_sigma") if np.isfinite(_maybe_float(row.get("tfep_sigma"))) else row.get("tfep_bootstrap_std")
        for row in fold_rows
    ])
    snf_err = finite_or_nan([
        row.get("snf_sigma") if np.isfinite(_maybe_float(row.get("snf_sigma"))) else row.get("snf_bootstrap_std")
        for row in fold_rows
    ])

    fig = plt.figure(figsize=(8, 4.8))
    offset = 0.12 if np.any(np.isfinite(snf_deltaf)) else 0.08
    plt.errorbar(fold_ids - offset, raw_deltaf, yerr=raw_err, fmt="o-", capsize=3, color=COLORS["raw"], label="raw BAR")
    plt.errorbar(fold_ids, tfep_deltaf, yerr=tfep_err, fmt="o-", capsize=3, color=COLORS["tfep"], label="TFEP BAR")
    if np.any(np.isfinite(snf_deltaf)):
        plt.errorbar(fold_ids + offset, snf_deltaf, yerr=snf_err, fmt="o-", capsize=3, color=COLORS["snf"], label="stochastic path TFEP")
    plt.xlabel("fold")
    plt.ylabel("deltaf (kT)")
    plt.title("Held-out deltaf by fold")
    plt.xticks(fold_ids)
    plt.legend()
    savefig(fig, out_path, dpi)


def plot_fold_overlap(fold_rows: List[Dict[str, Any]], out_path: Path, dpi: int) -> None:
    if not fold_rows:
        return
    fold_ids = np.array([int(row["fold"]) for row in fold_rows], dtype=int)
    raw_overlap = finite_or_nan([row.get("raw_overlap") for row in fold_rows])
    tfep_overlap = finite_or_nan([row.get("tfep_overlap") for row in fold_rows])
    snf_overlap = finite_or_nan([row.get("snf_overlap") for row in fold_rows])

    fig = plt.figure(figsize=(8, 4.8))
    plt.plot(fold_ids, raw_overlap, "o-", color=COLORS["raw"], label="raw BAR")
    plt.plot(fold_ids, tfep_overlap, "o-", color=COLORS["tfep"], label="TFEP BAR")
    if np.any(np.isfinite(snf_overlap)):
        plt.plot(fold_ids, snf_overlap, "o-", color=COLORS["snf"], label="stochastic path TFEP")
    plt.xlabel("fold")
    plt.ylabel("BAR overlap")
    plt.title("Held-out BAR-consistent overlap by fold")
    plt.ylim(0.0, 1.05)
    plt.xticks(fold_ids)
    plt.legend()
    savefig(fig, out_path, dpi)


def plot_fold_direct_overlap(fold_rows: List[Dict[str, Any]], out_path: Path, dpi: int) -> None:
    if not fold_rows:
        return
    fold_ids = np.array([int(row["fold"]) for row in fold_rows], dtype=int)
    raw_overlap = finite_or_nan([row.get("raw_direct_overlap") for row in fold_rows])
    tfep_overlap = finite_or_nan([row.get("tfep_direct_overlap") for row in fold_rows])
    snf_overlap = finite_or_nan([row.get("snf_direct_overlap") for row in fold_rows])
    if not np.any(np.isfinite(raw_overlap)) and not np.any(np.isfinite(tfep_overlap)) and not np.any(np.isfinite(snf_overlap)):
        return

    fig = plt.figure(figsize=(8, 4.8))
    plt.plot(fold_ids, raw_overlap, "o-", color=COLORS["raw"], label="raw BAR")
    plt.plot(fold_ids, tfep_overlap, "o-", color=COLORS["tfep"], label="TFEP BAR")
    if np.any(np.isfinite(snf_overlap)):
        plt.plot(fold_ids, snf_overlap, "o-", color=COLORS["snf"], label="stochastic path TFEP")
    plt.xlabel("fold")
    plt.ylabel("direct overlap")
    plt.title("Held-out direct overlap by fold")
    plt.ylim(0.0, 1.05)
    plt.xticks(fold_ids)
    plt.legend()
    savefig(fig, out_path, dpi)


def plot_fold_bootstrap_std(fold_rows: List[Dict[str, Any]], out_path: Path, dpi: int) -> None:
    if not fold_rows:
        return
    fold_ids = np.array([int(row["fold"]) for row in fold_rows], dtype=int)
    raw_std = finite_or_nan([row.get("raw_bootstrap_std") for row in fold_rows])
    tfep_std = finite_or_nan([row.get("tfep_bootstrap_std") for row in fold_rows])
    snf_std = finite_or_nan([row.get("snf_bootstrap_std") for row in fold_rows])
    has_snf = np.any(np.isfinite(snf_std))

    fig = plt.figure(figsize=(8, 4.8))
    width = 0.25 if has_snf else 0.35
    if has_snf:
        plt.bar(fold_ids - width, raw_std, width=width, color=COLORS["raw"], alpha=0.8, label="raw BAR")
        plt.bar(fold_ids, tfep_std, width=width, color=COLORS["tfep"], alpha=0.8, label="TFEP BAR")
        plt.bar(fold_ids + width, snf_std, width=width, color=COLORS["snf"], alpha=0.8, label="stochastic path TFEP")
    else:
        plt.bar(fold_ids - width / 2.0, raw_std, width=width, color=COLORS["raw"], alpha=0.8, label="raw BAR")
        plt.bar(fold_ids + width / 2.0, tfep_std, width=width, color=COLORS["tfep"], alpha=0.8, label="TFEP BAR")
    plt.xlabel("fold")
    plt.ylabel("bootstrap std(deltaf)")
    plt.title("Held-out bootstrap spread by fold")
    plt.xticks(fold_ids)
    plt.legend()
    savefig(fig, out_path, dpi)


def plot_fold_work_stds(fold_rows: List[Dict[str, Any]], out_path: Path, dpi: int) -> None:
    if not fold_rows:
        return
    fold_ids = np.array([int(row["fold"]) for row in fold_rows], dtype=int)
    raw_f = finite_or_nan([row.get("raw_forward_std") for row in fold_rows])
    raw_r = finite_or_nan([row.get("raw_reverse_std") for row in fold_rows])
    tfep_f = finite_or_nan([row.get("tfep_forward_std") for row in fold_rows])
    tfep_r = finite_or_nan([row.get("tfep_reverse_std") for row in fold_rows])
    snf_f = finite_or_nan([row.get("snf_forward_std") for row in fold_rows])
    snf_r = finite_or_nan([row.get("snf_reverse_std") for row in fold_rows])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharey=False)
    axes[0].plot(fold_ids, raw_f, "o-", color=COLORS["raw"], label="raw")
    axes[0].plot(fold_ids, tfep_f, "o-", color=COLORS["tfep"], label="tfep")
    if np.any(np.isfinite(snf_f)):
        axes[0].plot(fold_ids, snf_f, "o-", color=COLORS["snf"], label="stochastic")
    axes[0].set_title("Forward work std")
    axes[0].set_xlabel("fold")
    axes[0].set_ylabel("std(work)")
    axes[0].legend()

    axes[1].plot(fold_ids, raw_r, "o-", color=COLORS["raw"], label="raw")
    axes[1].plot(fold_ids, tfep_r, "o-", color=COLORS["tfep"], label="tfep")
    if np.any(np.isfinite(snf_r)):
        axes[1].plot(fold_ids, snf_r, "o-", color=COLORS["snf"], label="stochastic")
    axes[1].set_title("Reverse work std")
    axes[1].set_xlabel("fold")
    axes[1].set_ylabel("std(work)")
    axes[1].legend()

    savefig(fig, out_path, dpi)


def plot_manifest_coverage(analysis_dir: Path, manifest: Dict[str, Any], out_path: Path, dpi: int) -> None:
    folds = manifest.get("folds", [])
    if not folds:
        state0_val = analysis_dir / "state0_val_indices.npy"
        state1_val = analysis_dir / "state1_val_indices.npy"
        if not state0_val.exists() or not state1_val.exists():
            return

        fig, axes = plt.subplots(2, 1, figsize=(10, 5.2), sharex=False)
        for ax, state_name, path in zip(axes, ("state0", "state1"), (state0_val, state1_val)):
            idx = np.load(path)
            ax.scatter(idx, np.zeros_like(idx, dtype=float), s=3, alpha=0.6, color=COLORS["tfep"])
            ax.set_ylabel("split")
            ax.set_yticks([0.0], ["held-out"])
            ax.set_title(f"Validation coverage: {state_name}")
            ax.grid(alpha=0.2)
        axes[-1].set_xlabel("trajectory index")
        savefig(fig, out_path, dpi)
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 5.8), sharex=False)
    for ax_idx, state_name in enumerate(("state0", "state1")):
        ax = axes[ax_idx]
        for fold in folds:
            fold_id = int(fold["fold"])
            fold_dir = analysis_dir / "folds" / f"fold_{fold_id:02d}"
            val_idx_path = fold_dir / f"{state_name}_val_indices.npy"
            if val_idx_path.exists():
                idx = np.load(val_idx_path)
                ax.scatter(idx, np.full_like(idx, fold_id, dtype=float), s=3, alpha=0.6, label=f"fold {fold_id}" if ax_idx == 0 else None)
            else:
                lo = float(fold.get(f"{state_name}_val_start", 0))
                hi = float(fold.get(f"{state_name}_val_stop", lo))
                ax.hlines(fold_id, lo, hi, linewidth=6)
        ax.set_ylabel(f"{state_name} fold")
        ax.set_title(f"Validation coverage: {state_name}")
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("trajectory index")
    savefig(fig, out_path, dpi)


def plot_convergence(rows: List[Dict[str, Any]], value_key: str, ylabel: str, title: str, out_path: Path, dpi: int) -> None:
    if not rows:
        return
    methods = sorted({str(row.get("method")) for row in rows})
    fig = plt.figure(figsize=(8, 4.8))
    for method in methods:
        subset = [row for row in rows if str(row.get("method")) == method and np.isfinite(_maybe_float(row.get(value_key)))]
        if not subset:
            continue
        xs = finite_or_nan([row.get("sample_size") for row in subset])
        ys = finite_or_nan([row.get(value_key) for row in subset])
        order = np.argsort(xs)
        plt.plot(xs[order], ys[order], marker="o", label=method)
    plt.xlabel("held-out samples per direction")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    savefig(fig, out_path, dpi)


def plot_convergence_mean(rows: List[Dict[str, Any]], summary: Dict[str, Any], out_path: Path, dpi: int) -> None:
    if not rows:
        return
    raw, _, _, _, _ = extract_summary_views(summary)
    raw_ref = _maybe_float(raw.get("deltaf", np.nan))
    methods = sorted({str(row.get("method")) for row in rows})
    fig = plt.figure(figsize=(8, 4.8))
    for method in methods:
        subset = [row for row in rows if str(row.get("method")) == method and np.isfinite(_maybe_float(row.get("deltaf_mean")))]
        if not subset:
            continue
        xs = finite_or_nan([row.get("sample_size") for row in subset])
        ys = finite_or_nan([row.get("deltaf_mean") for row in subset])
        order = np.argsort(xs)
        plt.plot(xs[order], ys[order], marker="o", label=method)
    if np.isfinite(raw_ref):
        plt.axhline(raw_ref, color=COLORS["neutral"], linestyle="--", linewidth=1.5, label="raw full reference")
    plt.xlabel("held-out samples per direction")
    plt.ylabel("mean deltaf (kT)")
    plt.title("Convergence of deltaf mean")
    plt.legend()
    savefig(fig, out_path, dpi)


def work_arrays_from_file(path: Path) -> Optional[Dict[str, np.ndarray]]:
    if not path.exists():
        return None
    with np.load(path) as arrays:
        return {name: np.asarray(arrays[name]) for name in arrays.files}


def load_work_arrays(analysis_dir: Path) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[str]]:
    holdout_npz = analysis_dir / "validation_work_arrays.npz"
    if holdout_npz.exists():
        arrays = work_arrays_from_file(holdout_npz)
        if arrays is not None and (
            "raw_reweighted_bootstrap_deltaf" in arrays
            or "tfep_reweighted_bootstrap_deltaf" in arrays
        ):
            return arrays, None

    pooled_npz = analysis_dir / "pooled_work_arrays.npz"
    if pooled_npz.exists():
        return work_arrays_from_file(pooled_npz), None

    if holdout_npz.exists():
        return work_arrays_from_file(holdout_npz), None

    return None, "pooled_work_arrays.npz and validation_work_arrays.npz missing: work and bootstrap histogram plots skipped"


def array_with_fallback(arrays: Dict[str, np.ndarray], primary: str, fallback: Optional[str] = None) -> Optional[np.ndarray]:
    if primary in arrays:
        return np.asarray(arrays[primary])
    if fallback is not None and fallback in arrays:
        return np.asarray(arrays[fallback])
    return None


def plot_bootstrap_hist(arrays: Optional[Dict[str, np.ndarray]], out_path: Path, dpi: int) -> None:
    if arrays is None:
        return
    has_reweighted = (
        "raw_reweighted_bootstrap_deltaf" in arrays
        and "tfep_reweighted_bootstrap_deltaf" in arrays
        and len(finite_values(arrays.get("raw_reweighted_bootstrap_deltaf", []))) > 0
        and len(finite_values(arrays.get("tfep_reweighted_bootstrap_deltaf", []))) > 0
    )
    if has_reweighted:
        series = [
            ("raw_reweighted_bootstrap_deltaf", "raw reweighted BAR", COLORS["raw"]),
            ("tfep_reweighted_bootstrap_deltaf", "TFEP reweighted BAR", COLORS["tfep"]),
            ("snf_bootstrap_deltaf", "stochastic path TFEP", COLORS["snf"]),
        ]
        title = "Reweighted bootstrap deltaf distributions"
    else:
        series = [
            ("raw_bootstrap_deltaf", "raw BAR", COLORS["raw"]),
            ("tfep_bootstrap_deltaf", "TFEP BAR", COLORS["tfep"]),
            ("snf_bootstrap_deltaf", "stochastic path TFEP", COLORS["snf"]),
        ]
        title = "Bootstrap deltaf distributions"
    finite_series = [
        (label, finite_values(arrays.get(key, [])), color)
        for key, label, color in series
        if len(finite_values(arrays.get(key, []))) > 0
    ]
    if len(finite_series) < 2:
        return

    fig = plt.figure(figsize=(8, 4.8))
    for label, values, color in finite_series:
        plt.hist(values, bins=50, histtype="step", density=True, linewidth=1.8, color=color, label=label)
    plt.xlabel("deltaf (kT)")
    plt.ylabel("density")
    plt.title(title)
    plt.legend()
    savefig(fig, out_path, dpi)


def plot_pooled_work_hists(arrays: Optional[Dict[str, np.ndarray]], out_path: Path, dpi: int) -> None:
    if arrays is None:
        return
    required = ["raw_w01", "raw_w10", "tfep_w01", "tfep_w10"]
    if any(name not in arrays for name in required):
        return
    has_snf = "snf_w01" in arrays and "snf_w10" in arrays
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    axes[0].hist(finite_values(arrays["raw_w01"]), bins=80, histtype="step", density=True, color=COLORS["raw"], label="raw")
    axes[0].hist(finite_values(arrays["tfep_w01"]), bins=80, histtype="step", density=True, color=COLORS["tfep"], label="tfep")
    if has_snf:
        axes[0].hist(finite_values(arrays["snf_w01"]), bins=80, histtype="step", density=True, color=COLORS["snf"], label="stochastic")
    axes[0].set_title("Forward work: state0->state1")
    axes[0].set_xlabel("work (kT)")
    axes[0].set_ylabel("density")
    axes[0].legend()

    axes[1].hist(finite_values(arrays["raw_w10"]), bins=80, histtype="step", density=True, color=COLORS["raw"], label="raw")
    axes[1].hist(finite_values(arrays["tfep_w10"]), bins=80, histtype="step", density=True, color=COLORS["tfep"], label="tfep")
    if has_snf:
        axes[1].hist(finite_values(arrays["snf_w10"]), bins=80, histtype="step", density=True, color=COLORS["snf"], label="stochastic")
    axes[1].set_title("Reverse work: state1->state0")
    axes[1].set_xlabel("work (kT)")
    axes[1].set_ylabel("density")
    axes[1].legend()
    savefig(fig, out_path, dpi)


def plot_pooled_work_vs_index(arrays: Optional[Dict[str, np.ndarray]], out_path: Path, dpi: int, max_points: int) -> None:
    if arrays is None:
        return
    raw_01_idx = array_with_fallback(arrays, "raw_state0_state1_traj_idx", "state0_state1_traj_idx")
    raw_10_idx = array_with_fallback(arrays, "raw_state1_state0_traj_idx", "state1_state0_traj_idx")
    tfep_01_idx = array_with_fallback(arrays, "tfep_state0_state1_traj_idx", "state0_state1_traj_idx")
    tfep_10_idx = array_with_fallback(arrays, "tfep_state1_state0_traj_idx", "state1_state0_traj_idx")
    snf_01_idx = array_with_fallback(arrays, "snf_state0_state1_traj_idx", "state0_state1_traj_idx")
    snf_10_idx = array_with_fallback(arrays, "snf_state1_state0_traj_idx", "state1_state0_traj_idx")
    if any(item is None for item in (raw_01_idx, raw_10_idx, tfep_01_idx, tfep_10_idx)):
        return

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=False)
    configs = [
        (axes[0], np.asarray(raw_01_idx, dtype=float), "raw_w01", np.asarray(tfep_01_idx, dtype=float), "tfep_w01", "state0->state1"),
        (axes[1], np.asarray(raw_10_idx, dtype=float), "raw_w10", np.asarray(tfep_10_idx, dtype=float), "tfep_w10", "state1->state0"),
    ]
    snf_configs = [
        (np.asarray(snf_01_idx, dtype=float) if snf_01_idx is not None else None, "snf_w01"),
        (np.asarray(snf_10_idx, dtype=float) if snf_10_idx is not None else None, "snf_w10"),
    ]
    for (ax, raw_idx, raw_work_key, tfep_idx, tfep_work_key, title), (snf_idx, snf_work_key) in zip(configs, snf_configs):
        raw_work = np.asarray(arrays[raw_work_key], dtype=float)
        tfep_work = np.asarray(arrays[tfep_work_key], dtype=float)
        raw_order = np.argsort(raw_idx)
        tfep_order = np.argsort(tfep_idx)
        raw_idx, raw_work = evenly_spaced_subset(raw_idx[raw_order], raw_work[raw_order], max_points)
        tfep_idx, tfep_work = evenly_spaced_subset(tfep_idx[tfep_order], tfep_work[tfep_order], max_points)
        ax.scatter(raw_idx, raw_work, s=4, alpha=0.45, color=COLORS["raw"], label="raw")
        ax.scatter(tfep_idx, tfep_work, s=4, alpha=0.45, color=COLORS["tfep"], label="tfep")
        if snf_idx is not None and snf_work_key in arrays:
            snf_work = np.asarray(arrays[snf_work_key], dtype=float)
            snf_order = np.argsort(snf_idx)
            snf_idx_plot, snf_work_plot = evenly_spaced_subset(snf_idx[snf_order], snf_work[snf_order], max_points)
            ax.scatter(snf_idx_plot, snf_work_plot, s=4, alpha=0.45, color=COLORS["snf"], label="stochastic")
        ax.set_title(f"Work vs trajectory index: {title}")
        ax.set_xlabel("trajectory index")
        ax.set_ylabel("work (kT)")
        ax.legend()
    savefig(fig, out_path, dpi)


def plot_fold_bootstrap_hists(analysis_dir: Path, out_path: Path, dpi: int, allowed_fold_ids: Optional[Sequence[int]] = None) -> None:
    fold_dirs = select_fold_dirs(analysis_dir, allowed_fold_ids)
    if not fold_dirs:
        return

    fig, axes = plt.subplots(len(fold_dirs), 1, figsize=(8, max(3.2 * len(fold_dirs), 4.2)), sharex=True)
    if len(fold_dirs) == 1:
        axes = [axes]
    plotted = False
    for ax, fold_dir in zip(axes, fold_dirs):
        arrays_path = fold_dir / "validation_work_arrays.npz"
        if not arrays_path.exists():
            ax.axis("off")
            ax.text(0.02, 0.5, f"{fold_dir.name}: validation_work_arrays.npz missing", va="center")
            continue
        arrays = np.load(arrays_path)
        raw = finite_values(arrays.get("raw_bootstrap_deltaf", []))
        tfep = finite_values(arrays.get("tfep_bootstrap_deltaf", []))
        snf = finite_values(arrays.get("snf_bootstrap_deltaf", []))
        if len(raw) == 0 or len(tfep) == 0:
            ax.axis("off")
            ax.text(0.02, 0.5, f"{fold_dir.name}: no finite bootstrap deltaf samples", va="center")
            continue
        ax.hist(raw, bins=35, histtype="step", density=True, color=COLORS["raw"], label="raw")
        ax.hist(tfep, bins=35, histtype="step", density=True, color=COLORS["tfep"], label="tfep")
        if len(snf) > 0:
            ax.hist(snf, bins=35, histtype="step", density=True, color=COLORS["snf"], label="stochastic")
        ax.set_ylabel(fold_dir.name)
        ax.legend(loc="upper right")
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    axes[-1].set_xlabel("deltaf (kT)")
    savefig(fig, out_path, dpi)


def plot_training_losses(analysis_dir: Path, out_path: Path, dpi: int, allowed_fold_ids: Optional[Sequence[int]] = None) -> None:
    fold_dirs = select_fold_dirs(analysis_dir, allowed_fold_ids)
    metrics = []
    for fold_dir in fold_dirs:
        metrics_path = latest_metrics_csv(fold_dir)
        if metrics_path is None:
            continue
        rows = load_csv_rows(metrics_path)
        if rows:
            metrics.append((fold_dir.name, rows))
    if not metrics:
        root_metrics = latest_metrics_csv(analysis_dir)
        if root_metrics is not None:
            rows = load_csv_rows(root_metrics)
            if rows:
                metrics.append(("holdout", rows))
    if not metrics:
        return

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=False)
    columns = [("loss", "total loss"), ("loss_0_1", "loss 0->1"), ("loss_1_0", "loss 1->0")]
    for ax, (column, ylabel) in zip(axes, columns):
        for label, rows in metrics:
            xs = finite_or_nan([row.get("step") for row in rows])
            ys = finite_or_nan([row.get(column) for row in rows])
            ok = np.isfinite(xs) & np.isfinite(ys)
            if not np.any(ok):
                continue
            order = np.argsort(xs[ok])
            ax.plot(xs[ok][order], ys[ok][order], linewidth=1.2, alpha=0.8, label=label)
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
    axes[0].set_title("Training losses by fold")
    if len(metrics) <= 8:
        axes[0].legend(ncol=2)
    savefig(fig, out_path, dpi)


def plot_fold_final_loss(analysis_dir: Path, out_path: Path, dpi: int, allowed_fold_ids: Optional[Sequence[int]] = None) -> None:
    fold_dirs = select_fold_dirs(analysis_dir, allowed_fold_ids)
    rows_out = []
    for fold_dir in fold_dirs:
        metrics_path = latest_metrics_csv(fold_dir)
        if metrics_path is None:
            continue
        rows = load_csv_rows(metrics_path)
        if not rows:
            continue
        last = rows[-1]
        rows_out.append((fold_dir.name, _maybe_float(last.get("loss")), _maybe_float(last.get("loss_0_1")), _maybe_float(last.get("loss_1_0"))))
    if not rows_out:
        root_metrics = latest_metrics_csv(analysis_dir)
        if root_metrics is not None:
            rows = load_csv_rows(root_metrics)
            if rows:
                last = rows[-1]
                rows_out.append(("holdout", _maybe_float(last.get("loss")), _maybe_float(last.get("loss_0_1")), _maybe_float(last.get("loss_1_0"))))
    if not rows_out:
        return

    labels = [item[0] for item in rows_out]
    total = finite_or_nan([item[1] for item in rows_out])
    dir01 = finite_or_nan([item[2] for item in rows_out])
    dir10 = finite_or_nan([item[3] for item in rows_out])
    x = np.arange(len(labels), dtype=float)
    width = 0.25

    fig = plt.figure(figsize=(10, 4.8))
    plt.bar(x - width, total, width=width, color=COLORS["neutral"], label="loss")
    plt.bar(x, dir01, width=width, color=COLORS["forward"], label="loss_0_1")
    plt.bar(x + width, dir10, width=width, color=COLORS["reverse"], label="loss_1_0")
    plt.xticks(x, labels, rotation=30)
    plt.ylabel("final logged loss")
    plt.title("Final training loss by fold")
    plt.legend()
    savefig(fig, out_path, dpi)


def plot_fold_shift(fold_rows: List[Dict[str, Any]], out_path: Path, dpi: int) -> None:
    if not fold_rows:
        return
    fold_ids = np.array([int(row["fold"]) for row in fold_rows], dtype=int)
    shift = finite_or_nan([_maybe_float(row.get("tfep_deltaf")) - _maybe_float(row.get("raw_deltaf")) for row in fold_rows])
    fig = plt.figure(figsize=(8, 4.8))
    plt.axhline(0.0, color=COLORS["neutral"], linestyle="--", linewidth=1.2)
    plt.bar(fold_ids, shift, color=COLORS["tfep"], alpha=0.85)
    plt.xlabel("fold")
    plt.ylabel("TFEP deltaf - raw deltaf (kT)")
    plt.title("Held-out deltaf shift by fold")
    plt.xticks(fold_ids)
    savefig(fig, out_path, dpi)


def plot_fold_deltaf_raw_vs_tfep(comp_rows: List[Dict[str, Any]], out_path: Path, dpi: int) -> None:
    if not comp_rows:
        return
    raw = finite_or_nan([row.get("raw_deltaf") for row in comp_rows])
    tfep = finite_or_nan([row.get("tfep_deltaf") for row in comp_rows])
    valid = np.isfinite(raw) & np.isfinite(tfep)
    if not np.any(valid):
        return
    raw = raw[valid]
    tfep = tfep[valid]

    lo = float(min(np.min(raw), np.min(tfep)))
    hi = float(max(np.max(raw), np.max(tfep)))
    margin = 0.08 * max(1e-6, hi - lo)

    fig = plt.figure(figsize=(6.2, 6.0))
    plt.scatter(raw, tfep, s=40, alpha=0.85, color=COLORS["tfep"])
    plt.plot([lo - margin, hi + margin], [lo - margin, hi + margin], linestyle="--", color=COLORS["neutral"], linewidth=1.2, label="y = x")
    plt.xlim(lo - margin, hi + margin)
    plt.ylim(lo - margin, hi + margin)
    plt.xlabel("raw BAR deltaf (kT)")
    plt.ylabel("TFEP BAR deltaf (kT)")
    plt.title("Paired held-out deltaf: mapped vs non-mapped")
    plt.legend()
    savefig(fig, out_path, dpi)


def plot_fold_mapped_vs_raw_dashboard(comp_rows: List[Dict[str, Any]], out_path: Path, dpi: int) -> None:
    if not comp_rows:
        return
    fold_ids = np.array([int(row["fold"]) for row in comp_rows], dtype=int)
    abs_shift = finite_or_nan([row.get("abs_deltaf_shift") for row in comp_rows])
    boot_ratio = finite_or_nan([row.get("bootstrap_std_ratio_tfep_over_raw") for row in comp_rows])
    sigma_ratio = finite_or_nan([row.get("sigma_ratio_tfep_over_raw") for row in comp_rows])
    overlap_gain = finite_or_nan([row.get("overlap_gain_tfep_minus_raw") for row in comp_rows])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].bar(fold_ids, abs_shift, color=COLORS["tfep"], alpha=0.85)
    axes[0, 0].set_title("|TFEP - raw| deltaf")
    axes[0, 0].set_xlabel("fold")
    axes[0, 0].set_ylabel("kT")

    axes[0, 1].plot(fold_ids, boot_ratio, "o-", color=COLORS["forward"], label="bootstrap std ratio")
    axes[0, 1].plot(fold_ids, sigma_ratio, "s-", color=COLORS["reverse"], label="sigma ratio")
    axes[0, 1].axhline(1.0, linestyle="--", color=COLORS["neutral"], linewidth=1.1)
    axes[0, 1].set_title("Uncertainty ratios (TFEP/raw)")
    axes[0, 1].set_xlabel("fold")
    axes[0, 1].set_ylabel("ratio")
    axes[0, 1].legend()

    axes[1, 0].bar(fold_ids, overlap_gain, color=COLORS["raw"], alpha=0.85)
    axes[1, 0].axhline(0.0, linestyle="--", color=COLORS["neutral"], linewidth=1.1)
    axes[1, 0].set_title("BAR-overlap gain (TFEP - raw)")
    axes[1, 0].set_xlabel("fold")
    axes[1, 0].set_ylabel("gain")

    scatter = axes[1, 1].scatter(abs_shift, boot_ratio, c=overlap_gain, cmap="coolwarm", s=60, alpha=0.9)
    axes[1, 1].axhline(1.0, linestyle="--", color=COLORS["neutral"], linewidth=1.1)
    axes[1, 1].set_title("Shift vs uncertainty ratio")
    axes[1, 1].set_xlabel("|TFEP - raw| deltaf (kT)")
    axes[1, 1].set_ylabel("bootstrap std ratio (TFEP/raw)")
    cbar = fig.colorbar(scatter, ax=axes[1, 1], fraction=0.045, pad=0.03)
    cbar.set_label("overlap gain")

    savefig(fig, out_path, dpi)


def plot_fold_rank(comp_rows: List[Dict[str, Any]], out_path: Path, dpi: int) -> None:
    if not comp_rows:
        return
    ranked = sorted(comp_rows, key=lambda row: (_maybe_float(row.get("rank_smaller_is_better")), int(row.get("fold", 0))))
    labels = [f"fold {int(row['fold'])}" for row in ranked]
    scores = finite_or_nan([row.get("comparison_score_smaller_is_better") for row in ranked])
    y = np.arange(len(labels), dtype=float)

    fig = plt.figure(figsize=(9.0, max(4.5, 0.35 * len(labels) + 1.8)))
    plt.barh(y, scores, color=COLORS["tfep"], alpha=0.85)
    plt.yticks(y, labels)
    plt.gca().invert_yaxis()
    plt.xlabel("comparison score (smaller is better)")
    plt.title("Fold ranking for mapped-vs-non-mapped comparison")
    savefig(fig, out_path, dpi)


def main() -> None:
    args = build_argparser().parse_args()
    analysis_dir = resolve_analysis_dir(args.outdir, args.analysis_subdir, args.analysis_dir)
    plots_dir = args.plots.resolve() if args.plots is not None else (analysis_dir / "plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    notes: List[str] = []

    summary = load_json(analysis_dir / "summary.json")
    manifest = load_json(analysis_dir / "split_manifest.json")
    fold_rows = load_csv_rows(analysis_dir / "fold_metrics.csv")
    convergence_rows = load_csv_rows(analysis_dir / "convergence.csv")
    work_arrays, array_note = load_work_arrays(analysis_dir)
    if array_note is not None:
        notes.append(array_note)

    if summary is not None:
        plot_text_summary(summary, plots_dir / "summary_text.png", args.dpi)
    else:
        notes.append("summary.json missing: summary_text and some convergence reference plots skipped")

    if manifest is not None:
        plot_manifest_coverage(analysis_dir, manifest, plots_dir / "fold_coverage.png", args.dpi)
    else:
        notes.append("split_manifest.json missing: fold coverage plot skipped")

    if fold_rows:
        allowed_fold_ids = [int(row["fold"]) for row in fold_rows if np.isfinite(_maybe_float(row.get("fold")))]
        plot_fold_deltaf(fold_rows, plots_dir / "fold_deltaf.png", args.dpi)
        plot_fold_overlap(fold_rows, plots_dir / "fold_overlap.png", args.dpi)
        plot_fold_direct_overlap(fold_rows, plots_dir / "fold_direct_overlap.png", args.dpi)
        plot_fold_bootstrap_std(fold_rows, plots_dir / "fold_bootstrap_std.png", args.dpi)
        plot_fold_work_stds(fold_rows, plots_dir / "fold_work_std.png", args.dpi)
        plot_fold_shift(fold_rows, plots_dir / "fold_deltaf_shift.png", args.dpi)
        comparison_rows = build_fold_comparison_rows(fold_rows)
        write_csv_rows(comparison_rows, plots_dir / "fold_mapped_vs_raw_comparison.csv")
        (plots_dir / "fold_mapped_vs_raw_summary.json").write_text(json.dumps(build_fold_comparison_summary(comparison_rows), indent=2))
        plot_fold_deltaf_raw_vs_tfep(comparison_rows, plots_dir / "fold_deltaf_raw_vs_tfep_scatter.png", args.dpi)
        plot_fold_mapped_vs_raw_dashboard(comparison_rows, plots_dir / "fold_mapped_vs_raw_dashboard.png", args.dpi)
        plot_fold_rank(comparison_rows, plots_dir / "fold_mapped_vs_raw_rank.png", args.dpi)
    else:
        allowed_fold_ids = None
        comparison_rows = []
        notes.append("fold_metrics.csv missing or empty: fold comparison plots skipped")

    if convergence_rows:
        plot_convergence(convergence_rows, "deltaf_std", "std(deltaf)", "Held-out convergence speed", plots_dir / "convergence_std.png", args.dpi)
        error_key = "mean_abs_error_vs_raw_full"
        if not any(np.isfinite(_maybe_float(row.get(error_key))) for row in convergence_rows):
            error_key = "mean_abs_error_vs_raw_validation"
        label = "mean abs error vs raw full BAR (kT)" if error_key == "mean_abs_error_vs_raw_full" else "mean abs error vs raw held-out BAR (kT)"
        plot_convergence(convergence_rows, error_key, label, "Held-out convergence accuracy proxy", plots_dir / "convergence_abs_error.png", args.dpi)
        if summary is not None:
            plot_convergence_mean(convergence_rows, summary, plots_dir / "convergence_deltaf_mean.png", args.dpi)
    else:
        notes.append("convergence.csv missing or empty: convergence plots skipped")

    plot_bootstrap_hist(work_arrays, plots_dir / "bootstrap_deltaf.png", args.dpi)
    plot_pooled_work_hists(work_arrays, plots_dir / "pooled_work_histograms.png", args.dpi)
    plot_pooled_work_vs_index(work_arrays, plots_dir / "pooled_work_vs_trajidx.png", args.dpi, int(args.max_scatter_points))
    plot_fold_bootstrap_hists(analysis_dir, plots_dir / "fold_bootstrap_histograms.png", args.dpi, allowed_fold_ids=allowed_fold_ids)
    plot_training_losses(analysis_dir, plots_dir / "training_losses_by_fold.png", args.dpi, allowed_fold_ids=allowed_fold_ids)
    plot_fold_final_loss(analysis_dir, plots_dir / "training_final_loss_by_fold.png", args.dpi, allowed_fold_ids=allowed_fold_ids)
    try:
        from tfep.stochastic.plotting import plot_stochastic_diagnostics

        snf_report = plot_stochastic_diagnostics(analysis_dir, plots_dir, dpi=int(args.dpi))
        if not snf_report.get("generated_plots"):
            notes.append("SNF plotting found no complete stochastic path arrays; stochastic plots skipped")
    except Exception as exc:
        snf_report = {"error": str(exc)}
        notes.append(f"SNF plotting failed: {exc}")

    report = {
        "analysis_dir": str(analysis_dir),
        "plots_dir": str(plots_dir),
        "notes": notes,
        "generated": sorted([p.name for p in plots_dir.glob("*.png")]),
        "generated_csv": sorted([p.name for p in plots_dir.glob("*.csv")]),
        "generated_json": sorted([p.name for p in plots_dir.glob("*.json")]),
        "comparison_rows": int(len(comparison_rows)),
        "stochastic_plot_report": snf_report,
    }
    (plots_dir / "plot_report.json").write_text(json.dumps(report, indent=2))
    print(f"[done] plots in: {plots_dir}")
    if notes:
        for note in notes:
            print(f"[note] {note}")


if __name__ == "__main__":
    main()
