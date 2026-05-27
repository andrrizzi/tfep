"""Rich diagnostic plots for stochastic path-weighted TFEP runs.

The plotting layer is intentionally read-only: it consumes saved work/path
terms and never recomputes or mutates deterministic or stochastic estimators.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {
    "raw": "#1f77b4",
    "tfep": "#d95f02",
    "snf": "#009E73",
    "forward": "#2ca02c",
    "reverse": "#9467bd",
    "neutral": "#444444",
    "warn": "#b2182b",
}


def _finite(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        return {}
    with np.load(path) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            converted: dict[str, Any] = {}
            for key, value in row.items():
                try:
                    converted[key] = float(value)
                except (TypeError, ValueError):
                    converted[key] = value
            rows.append(converted)
    return rows


def _savefig(fig, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _hist(ax, values: Any, *, label: str, color: str, bins: int = 80) -> bool:
    vals = _finite(values)
    if vals.size == 0:
        return False
    ax.hist(vals, bins=bins, histtype="step", density=True, linewidth=1.8, color=color, label=label)
    return True


def _scatter(ax, x: Any, y: Any, *, label: str, color: str, s: float = 6.0, alpha: float = 0.45) -> bool:
    xx = np.asarray(x, dtype=float).reshape(-1)
    yy = np.asarray(y, dtype=float).reshape(-1)
    n = min(xx.size, yy.size)
    if n == 0:
        return False
    xx = xx[:n]
    yy = yy[:n]
    ok = np.isfinite(xx) & np.isfinite(yy)
    if not ok.any():
        return False
    ax.scatter(xx[ok], yy[ok], s=s, alpha=alpha, color=color, label=label)
    return True


def _path_csv_dir(analysis_dir: Path, summary: Mapping[str, Any]) -> Optional[Path]:
    info = summary.get("stochastic_path_tfep", {}) if isinstance(summary, Mapping) else {}
    output_dir = info.get("output_dir") if isinstance(info, Mapping) else None
    if output_dir:
        p = Path(str(output_dir))
        if not p.is_absolute():
            p = analysis_dir / p
        if p.exists():
            return p
    candidates = [analysis_dir / "stochastic_tfep_outputs", analysis_dir / "train" / "stochastic_tfep_outputs"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_work_arrays(analysis_dir: Path) -> tuple[dict[str, np.ndarray], Optional[Path]]:
    # Prefer validation arrays because holdout runs store richer SNF terms there.
    for name in ("validation_work_arrays.npz", "pooled_work_arrays.npz"):
        path = analysis_dir / name
        arrays = _load_npz(path)
        if arrays:
            return arrays, path
    return {}, None


def _nonfinite_count(arrays: Mapping[str, np.ndarray], keys: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for key in keys:
        if key not in arrays:
            continue
        arr = np.asarray(arrays[key], dtype=float)
        out[key] = int(np.size(arr) - np.count_nonzero(np.isfinite(arr)))
    return out


def plot_snf_work_histograms(arrays: Mapping[str, np.ndarray], out: Path, dpi: int) -> bool:
    if "snf_w01" not in arrays or "snf_w10" not in arrays:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    plotted = False
    plotted |= _hist(axes[0], arrays.get("raw_w01", []), label="raw", color=COLORS["raw"])
    plotted |= _hist(axes[0], arrays.get("tfep_w01", []), label="deterministic TFEP", color=COLORS["tfep"])
    plotted |= _hist(axes[0], arrays.get("snf_w01", []), label="stochastic path TFEP", color=COLORS["snf"])
    axes[0].set_title("Forward work state0->state1")
    axes[0].set_xlabel("work (kT)")
    axes[0].set_ylabel("density")
    axes[0].legend()
    plotted |= _hist(axes[1], arrays.get("raw_w10", []), label="raw", color=COLORS["raw"])
    plotted |= _hist(axes[1], arrays.get("tfep_w10", []), label="deterministic TFEP", color=COLORS["tfep"])
    plotted |= _hist(axes[1], arrays.get("snf_w10", []), label="stochastic path TFEP", color=COLORS["snf"])
    axes[1].set_title("Reverse work state1->state0")
    axes[1].set_xlabel("work (kT)")
    axes[1].legend()
    if plotted:
        _savefig(fig, out, dpi)
    else:
        plt.close(fig)
    return plotted


def plot_snf_bootstrap(arrays: Mapping[str, np.ndarray], out: Path, dpi: int) -> bool:
    if "snf_bootstrap_deltaf" not in arrays:
        return False
    fig = plt.figure(figsize=(8, 4.8))
    plotted = False
    plotted |= _hist(plt.gca(), arrays.get("raw_bootstrap_deltaf", []), label="raw", color=COLORS["raw"], bins=60)
    plotted |= _hist(plt.gca(), arrays.get("tfep_bootstrap_deltaf", []), label="deterministic TFEP", color=COLORS["tfep"], bins=60)
    plotted |= _hist(plt.gca(), arrays.get("snf_bootstrap_deltaf", []), label="stochastic path TFEP", color=COLORS["snf"], bins=60)
    plt.xlabel("BAR deltaf bootstrap (kT)")
    plt.ylabel("density")
    plt.title("Bootstrap free-energy distributions")
    plt.legend()
    if plotted:
        _savefig(fig, out, dpi)
    else:
        plt.close(fig)
    return plotted


def plot_snf_overlap_crooks(arrays: Mapping[str, np.ndarray], out: Path, dpi: int) -> bool:
    if "snf_w01" not in arrays or "snf_w10" not in arrays:
        return False
    fig = plt.figure(figsize=(8.4, 5.2))
    plotted = False
    plotted |= _hist(plt.gca(), arrays["snf_w01"], label="SNF forward w01", color=COLORS["forward"], bins=80)
    plotted |= _hist(plt.gca(), -np.asarray(arrays["snf_w10"], dtype=float), label="SNF -reverse w10", color=COLORS["reverse"], bins=80)
    plt.xlabel("BAR common work axis (kT)")
    plt.ylabel("density")
    plt.title("Stochastic path BAR/Crooks overlap")
    plt.legend()
    if plotted:
        _savefig(fig, out, dpi)
    else:
        plt.close(fig)
    return plotted


def plot_snf_vs_tfep_scatter(arrays: Mapping[str, np.ndarray], out: Path, dpi: int) -> bool:
    if "snf_w01" not in arrays or "tfep_w01" not in arrays:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    plotted = False
    plotted |= _scatter(axes[0], arrays.get("tfep_w01", []), arrays.get("snf_w01", []), label="forward", color=COLORS["snf"])
    axes[0].set_title("Forward stochastic vs deterministic work")
    axes[0].set_xlabel("deterministic TFEP work (kT)")
    axes[0].set_ylabel("SNF path work (kT)")
    plotted |= _scatter(axes[1], arrays.get("tfep_w10", []), arrays.get("snf_w10", []), label="reverse", color=COLORS["reverse"])
    axes[1].set_title("Reverse stochastic vs deterministic work")
    axes[1].set_xlabel("deterministic TFEP work (kT)")
    axes[1].set_ylabel("SNF path work (kT)")
    for ax in axes:
        xlim = ax.get_xlim(); ylim = ax.get_ylim()
        lo = min(xlim[0], ylim[0]); hi = max(xlim[1], ylim[1])
        ax.plot([lo, hi], [lo, hi], "--", color="0.35", lw=1)
        ax.legend()
    if plotted:
        _savefig(fig, out, dpi)
    else:
        plt.close(fig)
    return plotted


def plot_snf_log_terms(arrays: Mapping[str, np.ndarray], out: Path, dpi: int) -> bool:
    keys = [
        "snf_logJ01", "snf_logJ10",
        "snf_sum_logq_forward_01", "snf_sum_logq_reverse_01",
        "snf_sum_logq_forward_10", "snf_sum_logq_reverse_10",
    ]
    if not any(key in arrays for key in keys):
        return False
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    plotted = False
    plotted |= _hist(axes[0, 0], arrays.get("snf_logJ01", []), label="logJ01", color=COLORS["tfep"], bins=70)
    plotted |= _hist(axes[0, 0], arrays.get("snf_logJ10", []), label="logJ10", color=COLORS["reverse"], bins=70)
    axes[0, 0].set_title("Deterministic log-Jacobian terms")
    axes[0, 0].legend()
    plotted |= _hist(axes[0, 1], arrays.get("snf_sum_logq_forward_01", []), label="logqF 01", color=COLORS["forward"], bins=70)
    plotted |= _hist(axes[0, 1], arrays.get("snf_sum_logq_reverse_01", []), label="logqR 01", color=COLORS["reverse"], bins=70)
    axes[0, 1].set_title("Forward-direction stochastic logq terms")
    axes[0, 1].legend()
    plotted |= _hist(axes[1, 0], arrays.get("snf_sum_logq_forward_10", []), label="logqF 10", color=COLORS["forward"], bins=70)
    plotted |= _hist(axes[1, 0], arrays.get("snf_sum_logq_reverse_10", []), label="logqR 10", color=COLORS["reverse"], bins=70)
    axes[1, 0].set_title("Reverse-direction stochastic logq terms")
    axes[1, 0].legend()
    if "snf_sum_logq_forward_01" in arrays and "snf_sum_logq_reverse_01" in arrays:
        imbalance01 = np.asarray(arrays["snf_sum_logq_forward_01"], dtype=float) - np.asarray(arrays["snf_sum_logq_reverse_01"], dtype=float)
        plotted |= _hist(axes[1, 1], imbalance01, label="logqF-logqR 01", color=COLORS["snf"], bins=70)
    if "snf_sum_logq_forward_10" in arrays and "snf_sum_logq_reverse_10" in arrays:
        imbalance10 = np.asarray(arrays["snf_sum_logq_forward_10"], dtype=float) - np.asarray(arrays["snf_sum_logq_reverse_10"], dtype=float)
        plotted |= _hist(axes[1, 1], imbalance10, label="logqF-logqR 10", color=COLORS["warn"], bins=70)
    axes[1, 1].set_title("Stochastic log-probability imbalance")
    axes[1, 1].legend()
    for ax in axes.ravel():
        ax.set_xlabel("reduced units")
        ax.set_ylabel("density")
    if plotted:
        _savefig(fig, out, dpi)
    else:
        plt.close(fig)
    return plotted


def plot_snf_path_decomposition(arrays: Mapping[str, np.ndarray], out: Path, dpi: int) -> bool:
    required = ["snf_w01", "snf_u_from_01", "snf_u_to_snf_01", "snf_logJ01", "snf_sum_logq_forward_01", "snf_sum_logq_reverse_01"]
    if not any(key in arrays for key in required):
        return False
    labels = []
    values = []
    for key, label in [
        ("snf_u_from_01", "u_source 01"),
        ("snf_u_to_snf_01", "u_target 01"),
        ("snf_logJ01", "logJ 01"),
        ("snf_sum_logq_forward_01", "logqF 01"),
        ("snf_sum_logq_reverse_01", "logqR 01"),
        ("snf_w01", "work 01"),
        ("snf_u_from_10", "u_source 10"),
        ("snf_u_to_snf_10", "u_target 10"),
        ("snf_logJ10", "logJ 10"),
        ("snf_sum_logq_forward_10", "logqF 10"),
        ("snf_sum_logq_reverse_10", "logqR 10"),
        ("snf_w10", "work 10"),
    ]:
        if key in arrays:
            vals = _finite(arrays[key])
            if vals.size:
                labels.append(label)
                values.append(vals)
    if not values:
        return False
    fig = plt.figure(figsize=(max(10.0, 0.65 * len(values) + 3), 5.6))
    try:
        plt.boxplot(values, tick_labels=labels, showfliers=False)
    except TypeError:
        plt.boxplot(values, labels=labels, showfliers=False)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("reduced units")
    plt.title("SNF path-work term decomposition")
    _savefig(fig, out, dpi)
    return True


def plot_snf_displacements(arrays: Mapping[str, np.ndarray], out: Path, dpi: int) -> bool:
    keys = ["snf_mapped_to_snf_rmsd_01", "snf_mapped_to_snf_rmsd_10", "snf_mapped_to_snf_max_disp_01", "snf_mapped_to_snf_max_disp_10"]
    if not any(key in arrays for key in keys):
        return False
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    plotted = False
    plotted |= _hist(axes[0], arrays.get("snf_mapped_to_snf_rmsd_01", []), label="RMSD 01", color=COLORS["forward"], bins=70)
    plotted |= _hist(axes[0], arrays.get("snf_mapped_to_snf_rmsd_10", []), label="RMSD 10", color=COLORS["reverse"], bins=70)
    axes[0].set_title("Mapped-to-stochastic RMSD")
    axes[0].set_xlabel("Angstrom")
    axes[0].legend()
    plotted |= _hist(axes[1], arrays.get("snf_mapped_to_snf_max_disp_01", []), label="max disp 01", color=COLORS["forward"], bins=70)
    plotted |= _hist(axes[1], arrays.get("snf_mapped_to_snf_max_disp_10", []), label="max disp 10", color=COLORS["reverse"], bins=70)
    axes[1].set_title("Mapped-to-stochastic max displacement")
    axes[1].set_xlabel("Angstrom")
    axes[1].legend()
    if plotted:
        _savefig(fig, out, dpi)
    else:
        plt.close(fig)
    return plotted


def _latest_metrics_csv(analysis_dir: Path) -> Optional[Path]:
    candidates = sorted((analysis_dir / "train" / "logs" / "tfep").glob("version_*/metrics.csv"))
    return candidates[-1] if candidates else None


def plot_snf_training_metrics(analysis_dir: Path, out: Path, dpi: int) -> bool:
    path = _latest_metrics_csv(analysis_dir)
    rows = _read_csv(path) if path else []
    if not rows:
        return False
    keys = [
        "snf_work_01_mean", "snf_work_10_mean",
        "snf_logq_forward_01_mean", "snf_logq_reverse_01_mean",
        "snf_logq_forward_10_mean", "snf_logq_reverse_10_mean",
        "snf_mapped_to_snf_rmsd_01_mean", "snf_mapped_to_snf_rmsd_10_mean",
        "snf_bar_obj", "snf_df_bar_obj",
    ]
    present = [key for key in keys if any(np.isfinite(row.get(key, np.nan)) for row in rows)]
    if not present:
        return False
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.0), sharex=True)
    step = np.asarray([row.get("step", i) for i, row in enumerate(rows)], dtype=float)
    for key in present:
        vals = np.asarray([row.get(key, np.nan) for row in rows], dtype=float)
        ax = axes[1] if "rmsd" in key else axes[0]
        ax.plot(step, vals, label=key, linewidth=1.4)
    axes[0].set_ylabel("reduced units")
    axes[0].set_title("SNF train-time work/logq/objective metrics")
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].set_ylabel("Angstrom")
    axes[1].set_xlabel("training step")
    axes[1].set_title("SNF train-time displacement metrics")
    axes[1].legend(fontsize=8, ncol=2)
    _savefig(fig, out, dpi)
    return True


def plot_stochastic_diagnostics(analysis_dir: Path, plots_dir: Optional[Path] = None, *, dpi: int = 180) -> dict[str, Any]:
    analysis_dir = Path(analysis_dir).expanduser().resolve()
    plots_dir = Path(plots_dir).expanduser().resolve() if plots_dir is not None else analysis_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    summary = _read_json(analysis_dir / "summary.json")
    arrays, arrays_path = _load_work_arrays(analysis_dir)
    path_dir = _path_csv_dir(analysis_dir, summary)
    path_forward_rows = _read_csv(path_dir / "paths_forward.csv") if path_dir else []
    path_reverse_rows = _read_csv(path_dir / "paths_reverse.csv") if path_dir else []

    generated: list[str] = []
    skipped: list[str] = []
    plotters = [
        ("snf_work_histograms.png", plot_snf_work_histograms),
        ("snf_bootstrap_deltaf.png", plot_snf_bootstrap),
        ("snf_overlap_crooks.png", plot_snf_overlap_crooks),
        ("snf_vs_tfep_work_scatter.png", plot_snf_vs_tfep_scatter),
        ("snf_logq_logj_histograms.png", plot_snf_log_terms),
        ("snf_path_work_decomposition.png", plot_snf_path_decomposition),
        ("snf_displacement_diagnostics.png", plot_snf_displacements),
    ]
    for filename, fn in plotters:
        ok = fn(arrays, plots_dir / filename, dpi)
        (generated if ok else skipped).append(filename)
    if plot_snf_training_metrics(analysis_dir, plots_dir / "snf_training_metrics.png", dpi):
        generated.append("snf_training_metrics.png")
    else:
        skipped.append("snf_training_metrics.png")

    keys = [key for key in arrays if key.startswith("snf_")]
    report = {
        "analysis_dir": str(analysis_dir),
        "arrays_path": str(arrays_path) if arrays_path else None,
        "path_dir": str(path_dir) if path_dir else None,
        "n_snf_array_keys": len(keys),
        "snf_array_keys": sorted(keys),
        "n_forward_path_rows": len(path_forward_rows),
        "n_reverse_path_rows": len(path_reverse_rows),
        "generated_plots": generated,
        "skipped_plots": skipped,
        "nonfinite_counts": _nonfinite_count(arrays, keys),
        "warning": "Stochastic path plots are diagnostics for path-weighted SNF outputs; deterministic TFEP arrays are not modified.",
    }
    (plots_dir / "stochastic_plot_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--plots", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=180)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    report = plot_stochastic_diagnostics(args.analysis_dir, args.plots, dpi=int(args.dpi))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
