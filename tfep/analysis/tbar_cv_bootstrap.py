#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import gc
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from tfep.app import small_molecule_tmbar as train_mod


def parse_fraction_list(text: str) -> List[float]:
    out: List[float] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        value = float(token)
        if value <= 0.0 or value > 1.0:
            raise ValueError(f"Invalid fraction {value}; expected 0 < f <= 1")
        out.append(value)
    if not out:
        raise ValueError("At least one convergence fraction is required")
    return out


def build_argparser() -> argparse.ArgumentParser:
    parser = train_mod.build_argparser()
    parser.description = (
        "Blocked K-fold cross-validation and bootstrap analysis for the local "
        "bromomethane TFEP/TMBAR workflow."
    )
    parser.add_argument("--cv-folds", type=int, default=5,
                        help="Number of folds for train/validation splitting")
    parser.add_argument("--cv-mode", choices=["blocked", "interleaved", "random"], default="blocked",
                        help="How folds are built from trajectory order")
    parser.add_argument("--cv-seed-offset", type=int, default=1000,
                        help="Seed offset added to the training seed for each fold")
    parser.add_argument("--analysis-subdir", type=str, default="cv_bootstrap",
                        help="Subdirectory created under --outdir for fold outputs and reports")
    parser.add_argument("--analysis-batch-size", type=int, default=None,
                        help="Batch size used during held-out evaluation; defaults to --batch-size")
    parser.add_argument("--bootstrap-replicates", type=int, default=200,
                        help="Number of bootstrap replicates for each fold and pooled result")
    parser.add_argument("--bootstrap-block-size", type=int, default=1,
                        help="Optional contiguous block size for bootstrap/subsample operations")
    parser.add_argument("--bootstrap-ci", type=float, default=0.95,
                        help="Central confidence interval reported from bootstrap estimates")
    parser.add_argument("--convergence-fractions", type=str, default="0.05,0.1,0.2,0.5,1.0",
                        help="Comma-separated fractions of the held-out sample count used for convergence curves")
    parser.add_argument("--convergence-replicates", type=int, default=100,
                        help="Number of repeated subsamples per convergence point")
    parser.add_argument("--convergence-min-samples", type=int, default=100,
                        help="Minimum number of forward/reverse samples used in convergence analysis")
    parser.add_argument("--save-work-arrays", action="store_true",
                        help="Save per-fold and pooled raw/TFEP work arrays as NPZ files")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only build and save the fold split manifest without training")
    return parser


def json_ready(value: Any):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def make_folds(n_samples: int, n_folds: int, mode: str, seed: int) -> List[np.ndarray]:
    if n_folds < 2:
        raise ValueError("--cv-folds must be >= 2 to produce a train/validation split")
    if n_folds > n_samples:
        raise ValueError(f"--cv-folds={n_folds} exceeds dataset length {n_samples}")

    indices = np.arange(n_samples, dtype=int)
    if mode == "blocked":
        return [chunk.astype(int) for chunk in np.array_split(indices, n_folds)]
    if mode == "interleaved":
        return [indices[i::n_folds].astype(int) for i in range(n_folds)]
    if mode == "random":
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(indices)
        return [chunk.astype(int) for chunk in np.array_split(shuffled, n_folds)]
    raise ValueError(f"Unsupported cv mode: {mode}")


def complement_indices(n_samples: int, val_indices: np.ndarray) -> np.ndarray:
    mask = np.ones(n_samples, dtype=bool)
    mask[np.asarray(val_indices, dtype=int)] = False
    return np.nonzero(mask)[0].astype(int)


def resample_with_replacement(arr: np.ndarray, rng: np.random.Generator, block_size: int) -> np.ndarray:
    arr = np.asarray(arr)
    n = len(arr)
    if n == 0:
        return arr.copy()
    if block_size <= 1:
        return arr[rng.integers(0, n, size=n)]

    starts = np.arange(0, n, int(block_size), dtype=int)
    chunks: List[np.ndarray] = []
    total = 0
    while total < n:
        start = int(starts[rng.integers(0, len(starts))])
        chunk = arr[start:min(start + int(block_size), n)]
        chunks.append(chunk)
        total += len(chunk)
    return np.concatenate(chunks, axis=0)[:n]


def subsample_without_replacement(arr: np.ndarray, n_pick: int, rng: np.random.Generator, block_size: int) -> np.ndarray:
    arr = np.asarray(arr)
    if n_pick > len(arr):
        raise ValueError(f"Cannot pick {n_pick} samples from array of length {len(arr)}")
    if block_size <= 1:
        sel = rng.choice(len(arr), size=n_pick, replace=False)
        return arr[np.asarray(sel, dtype=int)]

    starts = np.arange(0, len(arr), int(block_size), dtype=int)
    perm = rng.permutation(starts)
    chunks: List[np.ndarray] = []
    total = 0
    for start in perm:
        chunk = arr[int(start):min(int(start) + int(block_size), len(arr))]
        chunks.append(chunk)
        total += len(chunk)
        if total >= n_pick:
            break
    return np.concatenate(chunks, axis=0)[:n_pick]


def bootstrap_bar_summary(
    w_forward: np.ndarray,
    w_reverse: np.ndarray,
    *,
    n_boot: int,
    block_size: int,
    ci: float,
    seed: int,
) -> Tuple[Dict[str, Any], np.ndarray]:
    estimates = np.empty(int(n_boot), dtype=np.float64)
    rng = np.random.default_rng(seed)
    for i in range(int(n_boot)):
        wf = resample_with_replacement(w_forward, rng, block_size)
        wr = resample_with_replacement(w_reverse, rng, block_size)
        estimates[i] = train_mod.bar_deltaf(wf, wr)[0]

    estimates = estimates[np.isfinite(estimates)]
    alpha = 1.0 - float(ci)
    if len(estimates) == 0:
        summary = {
            "n_boot": int(n_boot),
            "n_finite": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }
    else:
        summary = {
            "n_boot": int(n_boot),
            "n_finite": int(len(estimates)),
            "mean": float(np.mean(estimates)),
            "std": float(np.std(estimates, ddof=1)) if len(estimates) > 1 else float("nan"),
            "ci_low": float(np.quantile(estimates, alpha / 2.0)),
            "ci_high": float(np.quantile(estimates, 1.0 - alpha / 2.0)),
        }
    return summary, estimates


def convergence_curve(
    method_name: str,
    w_forward: np.ndarray,
    w_reverse: np.ndarray,
    *,
    sample_sizes: Sequence[int],
    n_repeats: int,
    reference_deltaf: float,
    block_size: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []
    for sample_size in sample_sizes:
        estimates = np.empty(int(n_repeats), dtype=np.float64)
        for i in range(int(n_repeats)):
            wf = subsample_without_replacement(w_forward, int(sample_size), rng, block_size)
            wr = subsample_without_replacement(w_reverse, int(sample_size), rng, block_size)
            estimates[i] = train_mod.bar_deltaf(wf, wr)[0]
        estimates = estimates[np.isfinite(estimates)]
        if len(estimates) == 0:
            rows.append({
                "method": method_name,
                "sample_size": int(sample_size),
                "n_finite": 0,
                "deltaf_mean": float("nan"),
                "deltaf_std": float("nan"),
                "mean_abs_error_vs_raw_full": float("nan"),
            })
            continue
        rows.append({
            "method": method_name,
            "sample_size": int(sample_size),
            "n_finite": int(len(estimates)),
            "deltaf_mean": float(np.mean(estimates)),
            "deltaf_std": float(np.std(estimates, ddof=1)) if len(estimates) > 1 else float("nan"),
            "mean_abs_error_vs_raw_full": float(np.mean(np.abs(estimates - float(reference_deltaf)))),
        })
    return rows


def save_rows_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
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
            writer.writerow({key: json_ready(value) for key, value in row.items()})


def maybe_plot_bootstrap(raw_samples: np.ndarray, tfep_samples: np.ndarray, out_path: Path) -> None:
    if plt is None or len(raw_samples) == 0 or len(tfep_samples) == 0:
        return
    fig = plt.figure()
    plt.hist(raw_samples, bins=40, histtype="step", label="raw BAR", density=True)
    plt.hist(tfep_samples, bins=40, histtype="step", label="TFEP BAR", density=True)
    plt.xlabel("deltaf (kT)")
    plt.ylabel("density")
    plt.title("Bootstrap deltaf comparison")
    plt.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def maybe_plot_convergence(rows: Sequence[Dict[str, Any]], value_key: str, ylabel: str, title: str, out_path: Path) -> None:
    if plt is None or len(rows) == 0:
        return
    methods = sorted({row["method"] for row in rows})
    fig = plt.figure()
    for method in methods:
        subset = [row for row in rows if row["method"] == method and np.isfinite(row[value_key])]
        if not subset:
            continue
        xs = [row["sample_size"] for row in subset]
        ys = [row[value_key] for row in subset]
        plt.plot(xs, ys, marker="o", label=method)
    plt.xlabel("held-out samples per direction")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def pooled_summary(raw_eval_rows: Sequence[Dict[str, Any]], tfep_eval_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    raw_sigma_ratio = []
    raw_forward_std_ratio = []
    raw_reverse_std_ratio = []
    for raw_row, tfep_row in zip(raw_eval_rows, tfep_eval_rows):
        if np.isfinite(raw_row.get("sigma", np.nan)) and raw_row["sigma"] != 0.0 and np.isfinite(tfep_row.get("sigma", np.nan)):
            raw_sigma_ratio.append(tfep_row["sigma"] / raw_row["sigma"])
        if np.isfinite(raw_row.get("w_forward_std", np.nan)) and raw_row["w_forward_std"] != 0.0 and np.isfinite(tfep_row.get("w_forward_std", np.nan)):
            raw_forward_std_ratio.append(tfep_row["w_forward_std"] / raw_row["w_forward_std"])
        if np.isfinite(raw_row.get("w_reverse_std", np.nan)) and raw_row["w_reverse_std"] != 0.0 and np.isfinite(tfep_row.get("w_reverse_std", np.nan)):
            raw_reverse_std_ratio.append(tfep_row["w_reverse_std"] / raw_row["w_reverse_std"])
    return {
        "mean_sigma_ratio_tfep_over_raw": float(np.nanmean(raw_sigma_ratio)) if raw_sigma_ratio else float("nan"),
        "mean_forward_std_ratio_tfep_over_raw": float(np.nanmean(raw_forward_std_ratio)) if raw_forward_std_ratio else float("nan"),
        "mean_reverse_std_ratio_tfep_over_raw": float(np.nanmean(raw_reverse_std_ratio)) if raw_reverse_std_ratio else float("nan"),
    }


def write_report(
    args: argparse.Namespace,
    analysis_dir: Path,
    summary: Dict[str, Any],
    fold_rows: Sequence[Dict[str, Any]],
    convergence_rows: Sequence[Dict[str, Any]],
) -> None:
    raw_full = summary["pooled"]["raw_full"]
    tfep_oof = summary["pooled"]["tfep_oof"]
    comparison = summary["comparison"]
    lines = [
        "# Bromomethane TFEP/TMBAR Cross-Validation Report",
        "",
        "## Setup",
        "",
        f"- Fold count: {int(args.cv_folds)}",
        f"- Fold mode: `{args.cv_mode}`",
        f"- Bootstrap replicates: {int(args.bootstrap_replicates)}",
        f"- Bootstrap block size: {int(args.bootstrap_block_size)}",
        f"- Training objective: `{args.objective}`",
        f"- Flow space: `{args.flow_space}`",
        "",
        "## Reference Comparison",
        "",
        f"- Standard BAR on pooled held-out raw works: `deltaf = {raw_full['deltaf']:.6f} kT`, `sigma = {raw_full['sigma']}`",
        f"- TFEP BAR on pooled held-out transformed works: `deltaf = {tfep_oof['deltaf']:.6f} kT`, `sigma = {tfep_oof['sigma']}`",
        f"- Pooled raw BAR-consistent overlap: `{raw_full.get('overlap', np.nan):.4f}`",
        f"- Pooled TFEP BAR-consistent overlap: `{tfep_oof.get('overlap', np.nan):.4f}`",
        f"- Pooled raw direct overlap (w_F vs w_R): `{raw_full.get('direct_overlap', np.nan):.4f}`",
        f"- Pooled TFEP direct overlap (w_F vs w_R): `{tfep_oof.get('direct_overlap', np.nan):.4f}`",
        f"- TFEP minus raw full-reference: `{comparison['tfep_minus_raw_full_deltaf']:.6f} kT`",
        f"- Mean fold sigma ratio TFEP/raw: `{comparison['mean_sigma_ratio_tfep_over_raw']}`",
        f"- Mean forward work std ratio TFEP/raw: `{comparison['mean_forward_std_ratio_tfep_over_raw']}`",
        f"- Mean reverse work std ratio TFEP/raw: `{comparison['mean_reverse_std_ratio_tfep_over_raw']}`",
        "",
        "Interpretation:",
        "Standard BAR is the baseline on the original cross-state works. TFEP-TMBAR trains an invertible map and then applies BAR to transformed works. If the held-out TFEP works are narrower and the held-out bootstrap spread is smaller without drifting far from raw full-data BAR, TFEP is converging faster at roughly comparable accuracy. If the TFEP estimate shifts away from the raw full-data baseline or only looks good in-training, the map is likely overfitting.",
        "",
        "## Fold Summaries",
        "",
    ]
    for row in fold_rows:
        lines.append(
            f"- Fold {row['fold']}: raw deltaf `{row['raw_deltaf']:.6f}`, tfep deltaf `{row['tfep_deltaf']:.6f}`, "
            f"raw sigma `{row['raw_sigma']}`, tfep sigma `{row['tfep_sigma']}`, "
            f"raw BAR overlap `{row['raw_overlap']:.4f}`, tfep BAR overlap `{row['tfep_overlap']:.4f}`, "
            f"raw direct overlap `{row.get('raw_direct_overlap', np.nan):.4f}`, tfep direct overlap `{row.get('tfep_direct_overlap', np.nan):.4f}`"
        )
    if convergence_rows:
        lines.extend([
            "",
            "## Convergence Data",
            "",
            "`convergence.csv` reports repeated held-out subsampling curves for raw BAR and TFEP BAR. "
            "Use `deltaf_std` as a sample-efficiency proxy and `mean_abs_error_vs_raw_full` as a practical accuracy proxy relative to the full raw-BAR baseline.",
        ])
    (analysis_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = build_argparser().parse_args()
    analysis_root = Path(args.outdir).resolve()
    analysis_dir = analysis_root / args.analysis_subdir
    analysis_dir.mkdir(parents=True, exist_ok=True)
    folds_dir = analysis_dir / "folds"
    folds_dir.mkdir(parents=True, exist_ok=True)

    probe_args = copy.deepcopy(args)
    probe_args.overwrite = False
    probe_args.outdir = str(analysis_dir / "_probe")
    probe_model, _ = train_mod.create_model_from_args(probe_args, outdir_override=probe_args.outdir)
    n_state0 = len(probe_model.make_dataset(0))
    n_state1 = len(probe_model.make_dataset(1))

    folds0 = make_folds(n_state0, int(args.cv_folds), args.cv_mode, int(args.seed))
    folds1 = make_folds(n_state1, int(args.cv_folds), args.cv_mode, int(args.seed) + 17)

    manifest = {
        "analysis_dir": str(analysis_dir),
        "n_state0": int(n_state0),
        "n_state1": int(n_state1),
        "cv_folds": int(args.cv_folds),
        "cv_mode": args.cv_mode,
        "seed": int(args.seed),
        "folds": [
            {
                "fold": int(i),
                "state0_val_count": int(len(folds0[i])),
                "state1_val_count": int(len(folds1[i])),
                "state0_val_start": int(folds0[i][0]),
                "state0_val_stop": int(folds0[i][-1]),
                "state1_val_start": int(folds1[i][0]),
                "state1_val_stop": int(folds1[i][-1]),
            }
            for i in range(int(args.cv_folds))
        ],
    }
    (analysis_dir / "split_manifest.json").write_text(json.dumps(json_ready(manifest), indent=2))
    if args.dry_run:
        print(f"[dry-run] wrote split manifest to {analysis_dir / 'split_manifest.json'}")
        return

    eval_batch_size = int(args.analysis_batch_size) if args.analysis_batch_size is not None else int(args.batch_size)
    fold_rows: List[Dict[str, Any]] = []
    raw_fold_summaries: List[Dict[str, Any]] = []
    tfep_fold_summaries: List[Dict[str, Any]] = []
    raw_w01_all: List[np.ndarray] = []
    raw_w10_all: List[np.ndarray] = []
    tfep_w01_all: List[np.ndarray] = []
    tfep_w10_all: List[np.ndarray] = []
    raw_idx0_all: List[np.ndarray] = []
    raw_idx1_all: List[np.ndarray] = []
    tfep_idx0_all: List[np.ndarray] = []
    tfep_idx1_all: List[np.ndarray] = []

    for fold_index in range(int(args.cv_folds)):
        fold_dir = folds_dir / f"fold_{fold_index:02d}"
        train_dir = fold_dir / "train"
        fold_dir.mkdir(parents=True, exist_ok=True)

        val0 = np.asarray(folds0[fold_index], dtype=int)
        val1 = np.asarray(folds1[fold_index], dtype=int)
        train0 = complement_indices(n_state0, val0)
        train1 = complement_indices(n_state1, val1)
        np.save(fold_dir / "state0_train_indices.npy", train0)
        np.save(fold_dir / "state1_train_indices.npy", train1)
        np.save(fold_dir / "state0_val_indices.npy", val0)
        np.save(fold_dir / "state1_val_indices.npy", val1)

        fold_args = copy.deepcopy(args)
        fold_args.outdir = str(train_dir)
        fold_args.overwrite = True
        fold_args.seed = int(args.seed) + int(args.cv_seed_offset) + int(fold_index)
        print(f"[fold {fold_index}] train0={len(train0)} val0={len(val0)} train1={len(train1)} val1={len(val1)}")
        artifacts = train_mod.run_training(
            fold_args,
            outdir_override=train_dir,
            train_indices_0=train0,
            train_indices_1=train1,
        )
        try:
            eval_result = train_mod.evaluate_bidirectional_map(
                artifacts.model,
                subset_indices_0=val0,
                subset_indices_1=val1,
                batch_size=eval_batch_size,
                include_tfep=True,
            )
        finally:
            train_mod.close_model_openmm_workers(artifacts.model)

        raw_summary = dict(eval_result["raw"])
        tfep_summary = dict(eval_result["tfep"])
        raw_bootstrap, raw_bootstrap_samples = bootstrap_bar_summary(
            eval_result["state0_state1"]["raw_work"],
            eval_result["state1_state0"]["raw_work"],
            n_boot=int(args.bootstrap_replicates),
            block_size=int(args.bootstrap_block_size),
            ci=float(args.bootstrap_ci),
            seed=int(args.seed) + 5000 + fold_index,
        )
        tfep_bootstrap, tfep_bootstrap_samples = bootstrap_bar_summary(
            eval_result["state0_state1"]["tfep_work"],
            eval_result["state1_state0"]["tfep_work"],
            n_boot=int(args.bootstrap_replicates),
            block_size=int(args.bootstrap_block_size),
            ci=float(args.bootstrap_ci),
            seed=int(args.seed) + 9000 + fold_index,
        )

        fold_summary = {
            "fold": int(fold_index),
            "train0": int(len(train0)),
            "val0": int(len(val0)),
            "train1": int(len(train1)),
            "val1": int(len(val1)),
            "raw": raw_summary,
            "tfep": tfep_summary,
            "raw_bootstrap": raw_bootstrap,
            "tfep_bootstrap": tfep_bootstrap,
        }
        (fold_dir / "summary.json").write_text(json.dumps(json_ready(fold_summary), indent=2))
        if args.save_work_arrays:
            np.savez_compressed(
                fold_dir / "validation_work_arrays.npz",
                state0_state1_traj_idx=eval_result["state0_state1"]["trajectory_sample_index"],
                state1_state0_traj_idx=eval_result["state1_state0"]["trajectory_sample_index"],
                raw_w01=eval_result["state0_state1"]["raw_work"],
                raw_w10=eval_result["state1_state0"]["raw_work"],
                tfep_w01=eval_result["state0_state1"]["tfep_work"],
                tfep_w10=eval_result["state1_state0"]["tfep_work"],
                raw_bootstrap_deltaf=raw_bootstrap_samples,
                tfep_bootstrap_deltaf=tfep_bootstrap_samples,
            )

        fold_rows.append({
            "fold": int(fold_index),
            "train0": int(len(train0)),
            "val0": int(len(val0)),
            "train1": int(len(train1)),
            "val1": int(len(val1)),
            "raw_deltaf": float(raw_summary["deltaf"]),
            "raw_sigma": float(raw_summary["sigma"]),
            "raw_overlap": float(raw_summary["overlap"]),
            "raw_direct_overlap": float(raw_summary.get("direct_overlap", np.nan)),
            "tfep_deltaf": float(tfep_summary["deltaf"]),
            "tfep_sigma": float(tfep_summary["sigma"]),
            "tfep_overlap": float(tfep_summary["overlap"]),
            "tfep_direct_overlap": float(tfep_summary.get("direct_overlap", np.nan)),
            "raw_bootstrap_std": float(raw_bootstrap["std"]),
            "tfep_bootstrap_std": float(tfep_bootstrap["std"]),
            "raw_forward_std": float(raw_summary["w_forward_std"]),
            "raw_reverse_std": float(raw_summary["w_reverse_std"]),
            "tfep_forward_std": float(tfep_summary["w_forward_std"]),
            "tfep_reverse_std": float(tfep_summary["w_reverse_std"]),
        })
        raw_fold_summaries.append(raw_summary)
        tfep_fold_summaries.append(tfep_summary)

        raw_w01_all.append(np.asarray(eval_result["state0_state1"]["raw_work"], dtype=np.float64))
        raw_w10_all.append(np.asarray(eval_result["state1_state0"]["raw_work"], dtype=np.float64))
        tfep_w01_all.append(np.asarray(eval_result["state0_state1"]["tfep_work"], dtype=np.float64))
        tfep_w10_all.append(np.asarray(eval_result["state1_state0"]["tfep_work"], dtype=np.float64))
        raw_idx0_all.append(np.asarray(eval_result["state0_state1"]["trajectory_sample_index"], dtype=int))
        raw_idx1_all.append(np.asarray(eval_result["state1_state0"]["trajectory_sample_index"], dtype=int))
        tfep_idx0_all.append(np.asarray(eval_result["state0_state1"]["trajectory_sample_index"], dtype=int))
        tfep_idx1_all.append(np.asarray(eval_result["state1_state0"]["trajectory_sample_index"], dtype=int))

        del artifacts
        gc.collect()

    raw_idx0 = np.concatenate(raw_idx0_all, axis=0)
    raw_idx1 = np.concatenate(raw_idx1_all, axis=0)
    raw_w01 = np.concatenate(raw_w01_all, axis=0)
    raw_w10 = np.concatenate(raw_w10_all, axis=0)
    tfep_idx0 = np.concatenate(tfep_idx0_all, axis=0)
    tfep_idx1 = np.concatenate(tfep_idx1_all, axis=0)
    tfep_w01 = np.concatenate(tfep_w01_all, axis=0)
    tfep_w10 = np.concatenate(tfep_w10_all, axis=0)

    raw_order0 = np.argsort(raw_idx0)
    raw_order1 = np.argsort(raw_idx1)
    tfep_order0 = np.argsort(tfep_idx0)
    tfep_order1 = np.argsort(tfep_idx1)
    raw_w01 = raw_w01[raw_order0]
    raw_w10 = raw_w10[raw_order1]
    raw_idx0 = raw_idx0[raw_order0]
    raw_idx1 = raw_idx1[raw_order1]
    tfep_w01 = tfep_w01[tfep_order0]
    tfep_w10 = tfep_w10[tfep_order1]
    tfep_idx0 = tfep_idx0[tfep_order0]
    tfep_idx1 = tfep_idx1[tfep_order1]

    raw_full = train_mod.summarize_work_pair(raw_w01, raw_w10)
    tfep_oof = train_mod.summarize_work_pair(tfep_w01, tfep_w10)
    raw_bootstrap_all, raw_bootstrap_samples = bootstrap_bar_summary(
        raw_w01,
        raw_w10,
        n_boot=int(args.bootstrap_replicates),
        block_size=int(args.bootstrap_block_size),
        ci=float(args.bootstrap_ci),
        seed=int(args.seed) + 15000,
    )
    tfep_bootstrap_all, tfep_bootstrap_samples = bootstrap_bar_summary(
        tfep_w01,
        tfep_w10,
        n_boot=int(args.bootstrap_replicates),
        block_size=int(args.bootstrap_block_size),
        ci=float(args.bootstrap_ci),
        seed=int(args.seed) + 19000,
    )

    n_common = min(len(raw_w01), len(raw_w10), len(tfep_w01), len(tfep_w10))
    fractions = parse_fraction_list(args.convergence_fractions)
    sample_sizes = sorted({
        max(int(args.convergence_min_samples), int(round(frac * n_common)))
        for frac in fractions
        if int(round(frac * n_common)) > 0
    })
    sample_sizes = [n for n in sample_sizes if n <= n_common]
    convergence_rows = []
    if sample_sizes:
        convergence_rows.extend(convergence_curve(
            "raw_bar",
            raw_w01,
            raw_w10,
            sample_sizes=sample_sizes,
            n_repeats=int(args.convergence_replicates),
            reference_deltaf=float(raw_full["deltaf"]),
            block_size=int(args.bootstrap_block_size),
            seed=int(args.seed) + 23000,
        ))
        convergence_rows.extend(convergence_curve(
            "tfep_bar",
            tfep_w01,
            tfep_w10,
            sample_sizes=sample_sizes,
            n_repeats=int(args.convergence_replicates),
            reference_deltaf=float(raw_full["deltaf"]),
            block_size=int(args.bootstrap_block_size),
            seed=int(args.seed) + 27000,
        ))

    comparison = pooled_summary(raw_fold_summaries, tfep_fold_summaries)
    comparison["tfep_minus_raw_full_deltaf"] = float(tfep_oof["deltaf"] - raw_full["deltaf"])

    summary = {
        "analysis": {
            "analysis_dir": str(analysis_dir),
            "cv_folds": int(args.cv_folds),
            "cv_mode": args.cv_mode,
            "bootstrap_replicates": int(args.bootstrap_replicates),
            "bootstrap_block_size": int(args.bootstrap_block_size),
            "bootstrap_ci": float(args.bootstrap_ci),
            "analysis_batch_size": int(eval_batch_size),
            "overlap_definition": "BAR-consistent histogram overlap of forward work w_F and sign-flipped reverse work -w_R",
            "direct_overlap_definition": "Direct histogram overlap of forward work w_F and reverse work w_R on their native axes",
        },
        "pooled": {
            "raw_full": raw_full,
            "tfep_oof": tfep_oof,
            "raw_bootstrap": raw_bootstrap_all,
            "tfep_bootstrap": tfep_bootstrap_all,
        },
        "comparison": comparison,
        "folds": fold_rows,
    }
    (analysis_dir / "summary.json").write_text(json.dumps(json_ready(summary), indent=2))
    save_rows_csv(fold_rows, analysis_dir / "fold_metrics.csv")
    save_rows_csv(convergence_rows, analysis_dir / "convergence.csv")

    if args.save_work_arrays:
        np.savez_compressed(
            analysis_dir / "pooled_work_arrays.npz",
            raw_state0_state1_traj_idx=raw_idx0,
            raw_state1_state0_traj_idx=raw_idx1,
            raw_w01=raw_w01,
            raw_w10=raw_w10,
            tfep_state0_state1_traj_idx=tfep_idx0,
            tfep_state1_state0_traj_idx=tfep_idx1,
            tfep_w01=tfep_w01,
            tfep_w10=tfep_w10,
            raw_bootstrap_deltaf=raw_bootstrap_samples,
            tfep_bootstrap_deltaf=tfep_bootstrap_samples,
        )

    maybe_plot_bootstrap(raw_bootstrap_samples, tfep_bootstrap_samples, analysis_dir / "bootstrap_deltaf.png")
    maybe_plot_convergence(
        convergence_rows,
        value_key="deltaf_std",
        ylabel="std(deltaf)",
        title="Held-out convergence speed",
        out_path=analysis_dir / "convergence_std.png",
    )
    maybe_plot_convergence(
        convergence_rows,
        value_key="mean_abs_error_vs_raw_full",
        ylabel="mean abs error vs raw full BAR (kT)",
        title="Held-out convergence accuracy proxy",
        out_path=analysis_dir / "convergence_abs_error.png",
    )
    write_report(args, analysis_dir, summary, fold_rows, convergence_rows)
    print(f"[done] analysis written to: {analysis_dir}")


if __name__ == "__main__":
    main()
