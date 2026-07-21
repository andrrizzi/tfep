#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import gc
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from tfep.analysis import reweighting as rwlib
from tfep.analysis import tbar_cv_bootstrap as cv_mod


train_mod = cv_mod.train_mod


def build_argparser() -> argparse.ArgumentParser:
    parser = train_mod.build_argparser()
    parser.description = (
        "Single train/validation holdout analysis for the local bromomethane "
        "TFEP/TMBAR workflow. Trains on a small subset and evaluates on the "
        "large held-out remainder."
    )
    parser.add_argument(
        "--train-count",
        type=int,
        default=1000,
        help="Number of training frames per state. The remainder is held out for validation.",
    )
    parser.add_argument(
        "--val-count",
        type=int,
        default=None,
        help=(
            "Optional number of held-out validation frames per state. If omitted, "
            "all frames not selected for training are used, preserving historical behavior."
        ),
    )
    parser.add_argument(
        "--large-val-count",
        type=int,
        default=None,
        help="Optional disjoint final-validation frame count per state.",
    )
    parser.add_argument(
        "--reserve-count",
        type=int,
        default=0,
        help="Optional disjoint reserve frame count per state.",
    )
    parser.add_argument(
        "--split-mode",
        choices=["random", "blocked", "time-stratified"],
        default="random",
        help="How the small training subset is chosen from each trajectory.",
    )
    parser.add_argument(
        "--split-seed-offset",
        type=int,
        default=2000,
        help="Seed offset added to --seed before selecting the holdout split.",
    )
    parser.add_argument(
        "--analysis-subdir",
        type=str,
        default="holdout_small_train",
        help="Subdirectory created under --outdir for the holdout outputs.",
    )
    parser.add_argument(
        "--analysis-batch-size",
        type=int,
        default=None,
        help="Batch size used during held-out evaluation; defaults to --batch-size.",
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=200,
        help="Number of bootstrap replicates for the held-out raw and TFEP BAR estimates.",
    )
    parser.add_argument(
        "--bootstrap-block-size",
        type=int,
        default=1,
        help="Optional contiguous block size for bootstrap and convergence subsampling.",
    )
    parser.add_argument(
        "--bootstrap-ci",
        type=float,
        default=0.95,
        help="Central confidence interval reported from bootstrap estimates.",
    )
    parser.add_argument(
        "--convergence-fractions",
        type=str,
        default="0.05,0.1,0.2,0.5,1.0",
        help="Comma-separated fractions of the held-out sample count used for convergence curves.",
    )
    parser.add_argument(
        "--convergence-replicates",
        type=int,
        default=100,
        help="Number of repeated held-out subsamples per convergence point.",
    )
    parser.add_argument(
        "--convergence-min-samples",
        type=int,
        default=100,
        help="Minimum number of forward/reverse held-out samples used in convergence analysis.",
    )
    parser.add_argument(
        "--save-work-arrays",
        action="store_true",
        help="Save held-out raw/TFEP work arrays and bootstrap samples as NPZ files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only build and save the holdout split manifest without training.",
    )
    return parser


def choose_train_indices(n_samples: int, n_pick: int, mode: str, seed: int) -> np.ndarray:
    if n_pick < 1:
        raise ValueError("--train-count must be >= 1")
    if n_pick >= n_samples:
        raise ValueError(f"--train-count={n_pick} must be smaller than dataset length {n_samples}")

    indices = np.arange(n_samples, dtype=int)
    rng = np.random.default_rng(seed)

    if mode == "random":
        return np.sort(rng.choice(indices, size=int(n_pick), replace=False).astype(int))
    if mode == "blocked":
        start = int(rng.integers(0, n_samples - n_pick + 1))
        return indices[start:start + int(n_pick)].astype(int)
    raise ValueError(f"Unsupported split mode: {mode}")


def complement_indices(n_samples: int, selected_indices: np.ndarray) -> np.ndarray:
    mask = np.ones(n_samples, dtype=bool)
    mask[np.asarray(selected_indices, dtype=int)] = False
    return np.nonzero(mask)[0].astype(int)


def choose_validation_indices(
    candidate_indices: np.ndarray,
    n_pick: Optional[int],
    mode: str,
    seed: int,
) -> np.ndarray:
    """Select a validation subset from the non-training candidates."""
    candidates = np.asarray(candidate_indices, dtype=int)
    if n_pick is None:
        return candidates
    n_pick = int(n_pick)
    if n_pick < 1:
        raise ValueError("--val-count must be >= 1 when provided")
    if n_pick > len(candidates):
        raise ValueError(
            f"--val-count={n_pick} exceeds available non-training frames "
            f"({len(candidates)})."
        )
    if n_pick == len(candidates):
        return candidates
    rng = np.random.default_rng(seed)
    if mode == "random":
        return np.sort(rng.choice(candidates, size=n_pick, replace=False).astype(int))
    if mode == "blocked":
        start = int(rng.integers(0, len(candidates) - n_pick + 1))
        return candidates[start:start + n_pick].astype(int)
    raise ValueError(f"Unsupported split mode: {mode}")


def choose_time_stratified_partitions(
    n_samples: int,
    counts: Dict[str, int],
    *,
    seed: int,
    n_strata: int = 100,
) -> Dict[str, np.ndarray]:
    """Split a trajectory exactly while spreading every partition over time."""
    if any(int(value) < 0 for value in counts.values()):
        raise ValueError("Four-way split counts must be non-negative")
    if sum(int(value) for value in counts.values()) != int(n_samples):
        raise ValueError(
            f"Four-way split counts sum to {sum(counts.values())}, expected {n_samples}"
        )
    labels = list(counts)
    strata = [np.asarray(x, dtype=int) for x in np.array_split(np.arange(n_samples, dtype=int), min(n_strata, n_samples))]
    remaining = {label: int(counts[label]) for label in labels}
    allocations: list[dict[str, int]] = []
    remaining_samples = int(n_samples)
    for stratum_idx, stratum in enumerate(strata):
        size = int(len(stratum))
        alloc: dict[str, int] = {}
        slots = size
        for label in labels[:-1]:
            if stratum_idx == len(strata) - 1:
                take = min(remaining[label], slots)
            else:
                target = remaining[label] * size / max(remaining_samples, 1)
                take = min(remaining[label], slots, int(round(target)))
            alloc[label] = int(take)
            remaining[label] -= int(take)
            slots -= int(take)
        last = labels[-1]
        take_last = min(remaining[last], slots)
        alloc[last] = int(take_last)
        remaining[last] -= int(take_last)
        slots -= int(take_last)
        # Fill rounding slack from partitions with the largest unmet fraction.
        while slots > 0:
            candidates = [label for label in labels if remaining[label] > 0]
            if not candidates:
                raise RuntimeError("Could not fill time-stratified split allocation")
            label = max(candidates, key=lambda x: remaining[x])
            alloc[label] += 1
            remaining[label] -= 1
            slots -= 1
        allocations.append(alloc)
        remaining_samples -= size
    if any(remaining.values()):
        raise RuntimeError(f"Time-stratified allocation left unassigned counts: {remaining}")

    rng = np.random.default_rng(seed)
    selected: dict[str, list[np.ndarray]] = {label: [] for label in labels}
    for stratum, alloc in zip(strata, allocations):
        shuffled = rng.permutation(stratum)
        start = 0
        for label in labels:
            stop = start + alloc[label]
            selected[label].append(shuffled[start:stop])
            start = stop
    return {
        label: np.sort(np.concatenate(chunks).astype(int)) if chunks else np.empty(0, dtype=int)
        for label, chunks in selected.items()
    }


def write_report(
    args: argparse.Namespace,
    analysis_dir: Path,
    summary: Dict[str, Any],
    convergence_rows: List[Dict[str, Any]],
) -> None:
    reweighted_primary = bool(summary.get("analysis", {}).get("primary_estimator") == "reweighted")
    raw_validation = summary["held_out"]["raw_reweighted"] if reweighted_primary else summary["held_out"]["raw"]
    tfep_validation = summary["held_out"]["tfep_reweighted"] if reweighted_primary else summary["held_out"]["tfep"]
    raw_diagnostics = summary["held_out"]["raw"]
    tfep_diagnostics = summary["held_out"]["tfep"]
    snf_validation = summary["held_out"].get("stochastic_path_tfep")
    comparison = summary["comparison"]
    split = summary["split"]
    estimator_label = "reweighted BAR" if reweighted_primary else "BAR"

    lines = [
        "# Bromomethane TFEP/TMBAR Small-Train Holdout Report",
        "",
        "## Setup",
        "",
        f"- Split mode: `{split['split_mode']}`",
        f"- Training frames per state: `{split['train_count_per_state']}`",
        f"- Validation frames state0/state1: `{split['val0_count']}` / `{split['val1_count']}`",
        f"- Training fraction state0/state1: `{split['train_fraction_state0']:.4f}` / `{split['train_fraction_state1']:.4f}`",
        f"- Split seeds state0/state1: `{split['state0_split_seed']}` / `{split['state1_split_seed']}`",
        f"- Bootstrap replicates: `{int(args.bootstrap_replicates)}`",
        f"- Bootstrap block size: `{int(args.bootstrap_block_size)}`",
        f"- Training objective: `{args.objective}`",
        f"- Flow space: `{args.flow_space}`",
        f"- Primary validation estimator: `{estimator_label}`",
        "",
        "## Held-Out Validation",
        "",
        f"- Raw {estimator_label} on held-out validation: `deltaf = {raw_validation['deltaf']:.6f} kT`, `sigma = {raw_validation.get('sigma', np.nan)}`",
        f"- TFEP {estimator_label} on held-out validation: `deltaf = {tfep_validation['deltaf']:.6f} kT`, `sigma = {tfep_validation.get('sigma', np.nan)}`",
        f"- Raw BAR-consistent overlap diagnostic: `{raw_diagnostics.get('overlap', np.nan):.4f}`",
        f"- TFEP BAR-consistent overlap diagnostic: `{tfep_diagnostics.get('overlap', np.nan):.4f}`",
        f"- Raw direct overlap diagnostic (w_F vs w_R): `{raw_diagnostics.get('direct_overlap', np.nan):.4f}`",
        f"- TFEP direct overlap diagnostic (w_F vs w_R): `{tfep_diagnostics.get('direct_overlap', np.nan):.4f}`",
        f"- TFEP minus raw held-out deltaf: `{comparison['tfep_minus_raw_validation_deltaf']:.6f} kT`",
        f"- Held-out sigma ratio TFEP/raw: `{comparison['mean_sigma_ratio_tfep_over_raw']}`",
        f"- Held-out forward work std ratio TFEP/raw: `{comparison['mean_forward_std_ratio_tfep_over_raw']}`",
        f"- Held-out reverse work std ratio TFEP/raw: `{comparison['mean_reverse_std_ratio_tfep_over_raw']}`",
        "",
        "Interpretation:",
        "This run is a low-data training stress test: the map is fitted on a small subset and judged only on the large held-out remainder. If TFEP narrows held-out work distributions and lowers held-out bootstrap spread without shifting far from raw held-out BAR, the map is helping under severe data scarcity. If TFEP only improves the training objective but drifts on this held-out set, it is overfitting.",
    ]
    if snf_validation is not None:
        lines.extend(
            [
                "",
                "## Experimental Stochastic Path TFEP",
                "",
                f"- SNF/path-weighted BAR: `deltaf = {snf_validation['deltaf']:.6f} kT`, `sigma = {snf_validation['sigma']}`",
                f"- SNF minus raw held-out deltaf: `{comparison.get('snf_minus_raw_validation_deltaf', np.nan):.6f} kT`",
                f"- SNF minus deterministic TFEP deltaf: `{comparison.get('snf_minus_tfep_validation_deltaf', np.nan):.6f} kT`",
                f"- SNF BAR-consistent overlap: `{snf_validation.get('overlap', np.nan):.4f}`",
                "",
                "These are experimental path-weighted stochastic works. They include the deterministic log-Jacobian and explicit stochastic transition-density terms, and are saved separately from deterministic TFEP arrays.",
            ]
        )

    if convergence_rows:
        lines.extend(
            [
                "",
                "## Convergence Data",
                "",
                f"`convergence.csv` reports repeated held-out subsampling curves for raw {estimator_label} and TFEP {estimator_label}. "
                "Use `deltaf_std` as a sample-efficiency proxy and `mean_abs_error_vs_raw_validation` as a practical "
                "accuracy proxy relative to the full held-out raw-BAR baseline.",
            ]
        )

    (analysis_dir / "report.md").write_text("\n".join(lines) + "\n")


def write_plotter_compat_arrays(
    analysis_dir: Path,
    eval_result: Dict[str, Any],
    raw_w01: np.ndarray,
    raw_w10: np.ndarray,
    tfep_w01: np.ndarray,
    tfep_w10: np.ndarray,
    raw_bootstrap_samples: np.ndarray,
    tfep_bootstrap_samples: np.ndarray,
    extra_arrays: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    arrays = {
        "raw_state0_state1_traj_idx": np.asarray(eval_result["state0_state1"]["trajectory_sample_index"], dtype=np.int64),
        "raw_state1_state0_traj_idx": np.asarray(eval_result["state1_state0"]["trajectory_sample_index"], dtype=np.int64),
        "tfep_state0_state1_traj_idx": np.asarray(eval_result["state0_state1"]["trajectory_sample_index"], dtype=np.int64),
        "tfep_state1_state0_traj_idx": np.asarray(eval_result["state1_state0"]["trajectory_sample_index"], dtype=np.int64),
        "raw_w01": np.asarray(raw_w01, dtype=np.float64),
        "raw_w10": np.asarray(raw_w10, dtype=np.float64),
        "tfep_w01": np.asarray(tfep_w01, dtype=np.float64),
        "tfep_w10": np.asarray(tfep_w10, dtype=np.float64),
        "raw_bootstrap_deltaf": np.asarray(raw_bootstrap_samples, dtype=np.float64),
        "tfep_bootstrap_deltaf": np.asarray(tfep_bootstrap_samples, dtype=np.float64),
    }
    if extra_arrays:
        arrays.update(extra_arrays)
    np.savez_compressed(analysis_dir / "pooled_work_arrays.npz", **arrays)


def build_holdout_fold_row(
    manifest: Dict[str, Any],
    raw_summary: Dict[str, Any],
    tfep_summary: Dict[str, Any],
    raw_bootstrap: Dict[str, Any],
    tfep_bootstrap: Dict[str, Any],
    *,
    estimator_mode: str = "unweighted",
    raw_diagnostics: Optional[Dict[str, Any]] = None,
    tfep_diagnostics: Optional[Dict[str, Any]] = None,
    snf_summary: Optional[Dict[str, Any]] = None,
    snf_bootstrap: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    raw_diag = raw_diagnostics or raw_summary
    tfep_diag = tfep_diagnostics or tfep_summary
    row = {
        "fold": 0,
        "estimator_mode": estimator_mode,
        "train0": int(manifest["train_count_per_state"]),
        "val0": int(manifest["val0_count"]),
        "train1": int(manifest["train_count_per_state"]),
        "val1": int(manifest["val1_count"]),
        "raw_deltaf": float(raw_summary["deltaf"]),
        "raw_sigma": float(raw_summary.get("sigma", np.nan)),
        "raw_overlap": float(raw_diag.get("overlap", np.nan)),
        "raw_direct_overlap": float(raw_diag.get("direct_overlap", np.nan)),
        "tfep_deltaf": float(tfep_summary["deltaf"]),
        "tfep_sigma": float(tfep_summary.get("sigma", np.nan)),
        "tfep_overlap": float(tfep_diag.get("overlap", np.nan)),
        "tfep_direct_overlap": float(tfep_diag.get("direct_overlap", np.nan)),
        "raw_bootstrap_std": float(raw_bootstrap["std"]),
        "tfep_bootstrap_std": float(tfep_bootstrap["std"]),
        "raw_forward_std": float(raw_diag["w_forward_std"]),
        "raw_reverse_std": float(raw_diag["w_reverse_std"]),
        "tfep_forward_std": float(tfep_diag["w_forward_std"]),
        "tfep_reverse_std": float(tfep_diag["w_reverse_std"]),
    }
    if snf_summary is not None:
        row.update(
            {
                "snf_deltaf": float(snf_summary["deltaf"]),
                "snf_sigma": float(snf_summary["sigma"]),
                "snf_overlap": float(snf_summary.get("overlap", np.nan)),
                "snf_direct_overlap": float(snf_summary.get("direct_overlap", np.nan)),
                "snf_bootstrap_std": float((snf_bootstrap or {}).get("std", np.nan)),
                "snf_forward_std": float(snf_summary["w_forward_std"]),
                "snf_reverse_std": float(snf_summary["w_reverse_std"]),
            }
        )
    return row


def weighted_convergence_curve(
    method: str,
    w01: np.ndarray,
    w10: np.ndarray,
    logw01: Optional[np.ndarray],
    logw10: Optional[np.ndarray],
    *,
    sample_sizes: List[int],
    n_repeats: int,
    reference_deltaf: float,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    w01 = np.asarray(w01, dtype=np.float64)
    w10 = np.asarray(w10, dtype=np.float64)
    lw01 = None if logw01 is None else np.asarray(logw01, dtype=np.float64)
    lw10 = None if logw10 is None else np.asarray(logw10, dtype=np.float64)
    rows: List[Dict[str, Any]] = []
    for n in sample_sizes:
        n_eff = int(min(n, len(w01), len(w10)))
        if n_eff < 2:
            continue
        estimates: List[float] = []
        for _ in range(int(n_repeats)):
            idx01 = rng.choice(len(w01), size=n_eff, replace=False)
            idx10 = rng.choice(len(w10), size=n_eff, replace=False)
            df, _ = rwlib.weighted_bar_deltaf(
                w01[idx01],
                w10[idx10],
                None if lw01 is None else lw01[idx01],
                None if lw10 is None else lw10[idx10],
            )
            if np.isfinite(df):
                estimates.append(float(df))
        arr = np.asarray(estimates, dtype=np.float64)
        rows.append({
            "method": method,
            "sample_size": int(n_eff),
            "n_repeats": int(n_repeats),
            "n_finite": int(arr.size),
            "deltaf_mean": float(np.mean(arr)) if arr.size else float("nan"),
            "deltaf_std": float(np.std(arr, ddof=1)) if arr.size > 1 else float("nan"),
            "mean_abs_error_vs_raw_validation": float(np.mean(np.abs(arr - float(reference_deltaf)))) if arr.size else float("nan"),
        })
    return rows


def main() -> None:
    args = build_argparser().parse_args()
    analysis_root = Path(args.outdir).resolve()
    analysis_dir = analysis_root / args.analysis_subdir
    analysis_dir.mkdir(parents=True, exist_ok=True)

    probe_args = copy.deepcopy(args)
    probe_args.overwrite = False
    probe_args.outdir = str(analysis_dir / "_probe")
    probe_model, _ = train_mod.create_model_from_args(probe_args, outdir_override=probe_args.outdir)
    try:
        n_state0 = len(probe_model.make_dataset(0))
        n_state1 = len(probe_model.make_dataset(1))
    finally:
        train_mod.close_model_openmm_workers(probe_model)

    train_count = int(args.train_count)
    if train_count < int(args.batch_size):
        raise ValueError(
            f"--train-count={train_count} is smaller than --batch-size={args.batch_size}; "
            "training would produce zero dropped-last batches."
        )
    if train_count >= n_state0 or train_count >= n_state1:
        raise ValueError(
            f"--train-count={train_count} must be smaller than both dataset lengths "
            f"(state0={n_state0}, state1={n_state1})."
        )

    seed0 = int(args.seed) + int(args.split_seed_offset)
    seed1 = int(args.seed) + int(args.split_seed_offset) + 17
    val_seed0 = int(args.seed) + int(args.split_seed_offset) + 101
    val_seed1 = int(args.seed) + int(args.split_seed_offset) + 118
    use_four_way = args.large_val_count is not None or int(args.reserve_count) > 0
    if use_four_way:
        if args.val_count is None or args.large_val_count is None:
            raise ValueError("Four-way splitting requires --val-count and --large-val-count")
        counts = {
            "train": train_count,
            "small_validation": int(args.val_count),
            "large_validation": int(args.large_val_count),
            "reserve": int(args.reserve_count),
        }
        if args.split_mode != "time-stratified":
            raise ValueError("Explicit four-way splitting currently requires --split-mode time-stratified")
        split0 = choose_time_stratified_partitions(n_state0, counts, seed=seed0)
        split1 = choose_time_stratified_partitions(n_state1, counts, seed=seed1)
        train0, train1 = split0["train"], split1["train"]
        val0, val1 = split0["small_validation"], split1["small_validation"]
        large0, large1 = split0["large_validation"], split1["large_validation"]
        reserve0, reserve1 = split0["reserve"], split1["reserve"]
    else:
        train0 = choose_train_indices(n_state0, train_count, args.split_mode, seed0)
        train1 = choose_train_indices(n_state1, train_count, args.split_mode, seed1)
        val0_candidates = complement_indices(n_state0, train0)
        val1_candidates = complement_indices(n_state1, train1)
        val0 = choose_validation_indices(val0_candidates, args.val_count, args.split_mode, val_seed0)
        val1 = choose_validation_indices(val1_candidates, args.val_count, args.split_mode, val_seed1)
        large0 = large1 = reserve0 = reserve1 = np.empty(0, dtype=int)

    np.save(analysis_dir / "state0_train_indices.npy", train0)
    np.save(analysis_dir / "state1_train_indices.npy", train1)
    np.save(analysis_dir / "state0_val_indices.npy", val0)
    np.save(analysis_dir / "state1_val_indices.npy", val1)
    if use_four_way:
        np.save(analysis_dir / "state0_large_val_indices.npy", large0)
        np.save(analysis_dir / "state1_large_val_indices.npy", large1)
        np.save(analysis_dir / "state0_reserve_indices.npy", reserve0)
        np.save(analysis_dir / "state1_reserve_indices.npy", reserve1)

    manifest = {
        "analysis_dir": str(analysis_dir),
        "n_state0": int(n_state0),
        "n_state1": int(n_state1),
        "train_count_per_state": int(train_count),
        "val0_count": int(len(val0)),
        "val1_count": int(len(val1)),
        "val_count_requested": None if args.val_count is None else int(args.val_count),
        "large_val0_count": int(len(large0)),
        "large_val1_count": int(len(large1)),
        "reserve0_count": int(len(reserve0)),
        "reserve1_count": int(len(reserve1)),
        "large_val_count_requested": None if args.large_val_count is None else int(args.large_val_count),
        "reserve_count_requested": int(args.reserve_count),
        "partitions_disjoint": bool(use_four_way),
        "split_mode": args.split_mode,
        "state0_split_seed": int(seed0),
        "state1_split_seed": int(seed1),
        "state0_val_seed": int(val_seed0),
        "state1_val_seed": int(val_seed1),
        "train_fraction_state0": float(len(train0) / n_state0),
        "train_fraction_state1": float(len(train1) / n_state1),
    }
    (analysis_dir / "split_manifest.json").write_text(json.dumps(cv_mod.json_ready(manifest), indent=2))
    if args.dry_run:
        print(f"[dry-run] wrote holdout split manifest to {analysis_dir / 'split_manifest.json'}")
        return

    train_dir = analysis_dir / "train"
    train_args = copy.deepcopy(args)
    train_args.outdir = str(train_dir)
    train_args.overwrite = True
    print(
        f"[holdout] train0={len(train0)} val0={len(val0)} "
        f"train1={len(train1)} val1={len(val1)}"
    )
    artifacts = train_mod.run_training(
        train_args,
        outdir_override=train_dir,
        train_indices_0=train0,
        train_indices_1=train1,
    )
    eval_batch_size = int(args.analysis_batch_size) if args.analysis_batch_size is not None else int(args.batch_size)
    relaxation_summary = None
    stochastic_result = None
    try:
        eval_result = train_mod.evaluate_bidirectional_map(
            artifacts.model,
            subset_indices_0=val0,
            subset_indices_1=val1,
            batch_size=eval_batch_size,
            include_tfep=True,
        )
        relaxation_summary = train_mod.run_relaxation_diagnostics_for_model(
            artifacts,
            args,
            analysis_dir=analysis_dir,
            subset_indices_0=val0,
            subset_indices_1=val1,
            batch_size=eval_batch_size,
            eval_result=eval_result,
        )
        stochastic_result = train_mod.run_stochastic_path_evaluation_for_model(
            artifacts,
            args,
            analysis_dir=analysis_dir,
            subset_indices_0=val0,
            subset_indices_1=val1,
            batch_size=eval_batch_size,
        )
    finally:
        train_mod.close_model_openmm_workers(artifacts.model)

    raw_summary = dict(eval_result["raw"])
    tfep_summary = dict(eval_result["tfep"])
    raw_w01 = np.asarray(eval_result["state0_state1"]["raw_work"], dtype=np.float64)
    raw_w10 = np.asarray(eval_result["state1_state0"]["raw_work"], dtype=np.float64)
    tfep_w01 = np.asarray(eval_result["state0_state1"]["tfep_work"], dtype=np.float64)
    tfep_w10 = np.asarray(eval_result["state1_state0"]["tfep_work"], dtype=np.float64)
    state0_log_weights_arr = np.asarray(eval_result["state0_state1"].get("log_weights", []), dtype=np.float64)
    state1_log_weights_arr = np.asarray(eval_result["state1_state0"].get("log_weights", []), dtype=np.float64)
    state0_log_weights = state0_log_weights_arr if len(state0_log_weights_arr) == len(raw_w01) and len(state0_log_weights_arr) > 0 else None
    state1_log_weights = state1_log_weights_arr if len(state1_log_weights_arr) == len(raw_w10) and len(state1_log_weights_arr) > 0 else None
    reweighting_enabled = state0_log_weights is not None or state1_log_weights is not None
    raw_reweighted_summary = dict(eval_result.get("raw_reweighted", {})) if reweighting_enabled else None
    tfep_reweighted_summary = dict(eval_result.get("tfep_reweighted", {})) if reweighting_enabled else None
    raw_reweighted_bootstrap = None
    tfep_reweighted_bootstrap = None
    raw_reweighted_bootstrap_samples = None
    tfep_reweighted_bootstrap_samples = None
    snf_w01 = None
    snf_w10 = None
    snf_summary = None
    snf_bootstrap = None
    snf_bootstrap_samples = None
    snf_forward = None
    snf_reverse = None
    if stochastic_result is not None:
        snf_forward = stochastic_result.get("state0_state1")
        snf_reverse = stochastic_result.get("state1_state0")
        if snf_forward is not None and snf_reverse is not None:
            snf_w01 = np.asarray(snf_forward["snf_work"], dtype=np.float64)
            snf_w10 = np.asarray(snf_reverse["snf_work"], dtype=np.float64)
            snf_summary = dict(stochastic_result.get("summary", {}))

    raw_bootstrap, raw_bootstrap_samples = cv_mod.bootstrap_bar_summary(
        raw_w01,
        raw_w10,
        n_boot=int(args.bootstrap_replicates),
        block_size=int(args.bootstrap_block_size),
        ci=float(args.bootstrap_ci),
        seed=int(args.seed) + 5000,
    )
    tfep_bootstrap, tfep_bootstrap_samples = cv_mod.bootstrap_bar_summary(
        tfep_w01,
        tfep_w10,
        n_boot=int(args.bootstrap_replicates),
        block_size=int(args.bootstrap_block_size),
        ci=float(args.bootstrap_ci),
        seed=int(args.seed) + 9000,
    )
    if reweighting_enabled:
        raw_weighted_boot = rwlib.weighted_bootstrap_bar_summary(
            raw_w01,
            raw_w10,
            state0_log_weights,
            state1_log_weights,
            n_boot=int(args.bootstrap_replicates),
            ci=float(args.bootstrap_ci),
            seed=int(args.seed) + 15000,
            block_size=int(args.bootstrap_block_size),
        )
        raw_reweighted_bootstrap = raw_weighted_boot.summary
        raw_reweighted_bootstrap_samples = raw_weighted_boot.samples
        tfep_weighted_boot = rwlib.weighted_bootstrap_bar_summary(
            tfep_w01,
            tfep_w10,
            state0_log_weights,
            state1_log_weights,
            n_boot=int(args.bootstrap_replicates),
            ci=float(args.bootstrap_ci),
            seed=int(args.seed) + 19000,
            block_size=int(args.bootstrap_block_size),
        )
        tfep_reweighted_bootstrap = tfep_weighted_boot.summary
        tfep_reweighted_bootstrap_samples = tfep_weighted_boot.samples
    if snf_w01 is not None and snf_w10 is not None:
        snf_bootstrap, snf_bootstrap_samples = cv_mod.bootstrap_bar_summary(
            snf_w01,
            snf_w10,
            n_boot=int(args.bootstrap_replicates),
            block_size=int(args.bootstrap_block_size),
            ci=float(args.bootstrap_ci),
            seed=int(args.seed) + 13000,
        )

    n_common = min(len(raw_w01), len(raw_w10), len(tfep_w01), len(tfep_w10))
    sample_sizes: List[int] = []
    convergence_rows: List[Dict[str, Any]] = []
    fractions = cv_mod.parse_fraction_list(args.convergence_fractions)
    if n_common > 0:
        sample_sizes = sorted(
            {
                max(int(args.convergence_min_samples), int(round(frac * n_common)))
                for frac in fractions
                if int(round(frac * n_common)) > 0
            }
        )
        sample_sizes = [n for n in sample_sizes if n <= n_common]

    if sample_sizes:
        if reweighting_enabled:
            convergence_rows.extend(
                weighted_convergence_curve(
                    "raw_reweighted_bar",
                    raw_w01,
                    raw_w10,
                    state0_log_weights,
                    state1_log_weights,
                    sample_sizes=sample_sizes,
                    n_repeats=int(args.convergence_replicates),
                    reference_deltaf=float(raw_reweighted_summary["deltaf"]),
                    seed=int(args.seed) + 23000,
                )
            )
            convergence_rows.extend(
                weighted_convergence_curve(
                    "tfep_reweighted_bar",
                    tfep_w01,
                    tfep_w10,
                    state0_log_weights,
                    state1_log_weights,
                    sample_sizes=sample_sizes,
                    n_repeats=int(args.convergence_replicates),
                    reference_deltaf=float(raw_reweighted_summary["deltaf"]),
                    seed=int(args.seed) + 27000,
                )
            )
        else:
            convergence_rows.extend(
                cv_mod.convergence_curve(
                    "raw_bar",
                    raw_w01,
                    raw_w10,
                    sample_sizes=sample_sizes,
                    n_repeats=int(args.convergence_replicates),
                    reference_deltaf=float(raw_summary["deltaf"]),
                    block_size=int(args.bootstrap_block_size),
                    seed=int(args.seed) + 23000,
                )
            )
            convergence_rows.extend(
                cv_mod.convergence_curve(
                    "tfep_bar",
                    tfep_w01,
                    tfep_w10,
                    sample_sizes=sample_sizes,
                    n_repeats=int(args.convergence_replicates),
                    reference_deltaf=float(raw_summary["deltaf"]),
                    block_size=int(args.bootstrap_block_size),
                    seed=int(args.seed) + 27000,
                )
            )
            for row in convergence_rows:
                row["mean_abs_error_vs_raw_validation"] = row.pop("mean_abs_error_vs_raw_full")

    primary_estimator = "reweighted" if reweighting_enabled else "unweighted"
    primary_raw_summary = raw_reweighted_summary if reweighting_enabled and raw_reweighted_summary else raw_summary
    primary_tfep_summary = tfep_reweighted_summary if reweighting_enabled and tfep_reweighted_summary else tfep_summary
    primary_raw_bootstrap = raw_reweighted_bootstrap if reweighting_enabled and raw_reweighted_bootstrap else raw_bootstrap
    primary_tfep_bootstrap = tfep_reweighted_bootstrap if reweighting_enabled and tfep_reweighted_bootstrap else tfep_bootstrap

    comparison = cv_mod.pooled_summary([raw_summary], [tfep_summary])
    comparison["estimator_mode"] = primary_estimator
    comparison["tfep_minus_raw_validation_deltaf"] = float(primary_tfep_summary["deltaf"] - primary_raw_summary["deltaf"])
    comparison["unweighted_tfep_minus_raw_validation_deltaf"] = float(tfep_summary["deltaf"] - raw_summary["deltaf"])
    if reweighting_enabled:
        comparison["reweighted_tfep_minus_raw_validation_deltaf"] = float(
            primary_tfep_summary["deltaf"] - primary_raw_summary["deltaf"]
        )
        comparison["mean_sigma_ratio_tfep_over_raw"] = float(
            primary_tfep_bootstrap["std"] / primary_raw_bootstrap["std"]
        ) if float(primary_raw_bootstrap["std"]) != 0.0 else float("nan")
    if snf_summary is not None:
        comparison["snf_minus_raw_validation_deltaf"] = float(snf_summary["deltaf"] - primary_raw_summary["deltaf"])
        comparison["snf_minus_tfep_validation_deltaf"] = float(snf_summary["deltaf"] - primary_tfep_summary["deltaf"])
        comparison["sigma_ratio_snf_over_raw"] = float(snf_summary["sigma"] / raw_summary["sigma"]) if float(raw_summary["sigma"]) != 0.0 else float("nan")

    summary = {
        "analysis": {
            "analysis_dir": str(analysis_dir),
            "analysis_batch_size": int(eval_batch_size),
            "bootstrap_replicates": int(args.bootstrap_replicates),
            "bootstrap_block_size": int(args.bootstrap_block_size),
            "bootstrap_ci": float(args.bootstrap_ci),
            "primary_estimator": primary_estimator,
            "primary_estimator_note": (
                "Enhanced-sampling log weights are present, so plots and fold metrics use raw_reweighted/tfep_reweighted estimates."
                if reweighting_enabled else
                "No enhanced-sampling log weights were present, so plots and fold metrics use ordinary unweighted BAR estimates."
            ),
            "overlap_definition": "BAR-consistent histogram overlap of forward work w_F and sign-flipped reverse work -w_R",
            "direct_overlap_definition": "Direct histogram overlap of forward work w_F and reverse work w_R on their native axes",
        },
        "split": manifest,
        "held_out": {
            "raw": raw_summary,
            "tfep": tfep_summary,
            "raw_bootstrap": raw_bootstrap,
            "tfep_bootstrap": tfep_bootstrap,
        },
        "comparison": comparison,
    }
    if reweighting_enabled:
        summary["reweighting"] = {
            "enabled": True,
            "state0": rwlib.log_weight_diagnostics(state0_log_weights),
            "state1": rwlib.log_weight_diagnostics(state1_log_weights),
            "statistical_note": (
                "Unweighted raw/TFEP estimates from biased enhanced-sampling trajectories "
                "are diagnostics only; use raw_reweighted/tfep_reweighted for equilibrium estimates."
            ),
        }
        summary["held_out"]["raw_reweighted"] = raw_reweighted_summary
        summary["held_out"]["tfep_reweighted"] = tfep_reweighted_summary
        summary["held_out"]["raw_reweighted_bootstrap"] = raw_reweighted_bootstrap
        summary["held_out"]["tfep_reweighted_bootstrap"] = tfep_reweighted_bootstrap
    if snf_summary is not None:
        summary["held_out"]["stochastic_path_tfep"] = snf_summary
        summary["held_out"]["stochastic_path_tfep_bootstrap"] = snf_bootstrap
        summary["stochastic_path_tfep"] = {
            "output_dir": stochastic_result.get("output_dir"),
            "metadata": stochastic_result.get("metadata", {}),
            "statistical_note": (
                "Experimental path-weighted stochastic TFEP. Work includes "
                "+logq_forward-logq_reverse and is saved separately from "
                "deterministic TFEP arrays."
            ),
        }
    if relaxation_summary is not None:
        summary["relaxation_diagnostic"] = relaxation_summary
    (analysis_dir / "summary.json").write_text(json.dumps(cv_mod.json_ready(summary), indent=2))
    cv_mod.save_rows_csv(
        [
            build_holdout_fold_row(
                manifest,
                primary_raw_summary,
                primary_tfep_summary,
                primary_raw_bootstrap,
                primary_tfep_bootstrap,
                estimator_mode=primary_estimator,
                raw_diagnostics=raw_summary,
                tfep_diagnostics=tfep_summary,
                snf_summary=snf_summary,
                snf_bootstrap=snf_bootstrap,
            )
        ],
        analysis_dir / "fold_metrics.csv",
    )
    cv_mod.save_rows_csv(convergence_rows, analysis_dir / "convergence.csv")

    if args.save_work_arrays:
        snf_arrays: Dict[str, np.ndarray] = {}
        reweighted_arrays: Dict[str, np.ndarray] = {}
        arrays = {
            "state0_state1_traj_idx": eval_result["state0_state1"]["trajectory_sample_index"],
            "state1_state0_traj_idx": eval_result["state1_state0"]["trajectory_sample_index"],
            "raw_w01": raw_w01,
            "raw_w10": raw_w10,
            "tfep_w01": tfep_w01,
            "tfep_w10": tfep_w10,
            "raw_bootstrap_deltaf": raw_bootstrap_samples,
            "tfep_bootstrap_deltaf": tfep_bootstrap_samples,
        }
        if reweighting_enabled:
            if state0_log_weights is not None:
                arrays["state0_log_weights"] = np.asarray(state0_log_weights, dtype=np.float64)
            if state1_log_weights is not None:
                arrays["state1_log_weights"] = np.asarray(state1_log_weights, dtype=np.float64)
            arrays["raw_reweighted_deltaf"] = np.asarray(
                [float(raw_reweighted_summary.get("deltaf", np.nan)) if raw_reweighted_summary else np.nan],
                dtype=np.float64,
            )
            arrays["tfep_reweighted_deltaf"] = np.asarray(
                [float(tfep_reweighted_summary.get("deltaf", np.nan)) if tfep_reweighted_summary else np.nan],
                dtype=np.float64,
            )
            arrays["raw_reweighted_bootstrap_deltaf"] = np.asarray(raw_reweighted_bootstrap_samples, dtype=np.float64)
            arrays["tfep_reweighted_bootstrap_deltaf"] = np.asarray(tfep_reweighted_bootstrap_samples, dtype=np.float64)
            reweighted_arrays = {
                key: np.asarray(value)
                for key, value in arrays.items()
                if key.startswith("state") and key.endswith("_log_weights")
                or key.startswith("raw_reweighted")
                or key.startswith("tfep_reweighted")
            }
        if snf_w01 is not None and snf_w10 is not None:
            snf_arrays.update({
                "snf_w01": snf_w01,
                "snf_w10": snf_w10,
                "snf_bootstrap_deltaf": np.asarray(snf_bootstrap_samples, dtype=np.float64),
            })
            for suffix, records in (("01", snf_forward), ("10", snf_reverse)):
                if records is None:
                    continue
                key_map = {
                    "log_det_J": f"snf_logJ{suffix}",
                    "sum_logq_forward": f"snf_sum_logq_forward_{suffix}",
                    "sum_logq_reverse": f"snf_sum_logq_reverse_{suffix}",
                    "u_from": f"snf_u_from_{suffix}",
                    "u_to_snf": f"snf_u_to_snf_{suffix}",
                    "u_to_mapped": f"snf_u_to_mapped_{suffix}",
                    "path_log_weight": f"snf_path_log_weight_{suffix}",
                    "mapped_to_snf_rmsd_angstrom": f"snf_mapped_to_snf_rmsd_{suffix}",
                    "mapped_to_snf_max_disp_angstrom": f"snf_mapped_to_snf_max_disp_{suffix}",
                    "trajectory_sample_index": f"snf_state{0 if suffix == '01' else 1}_state{1 if suffix == '01' else 0}_traj_idx",
                    "dataset_sample_index": f"snf_state{0 if suffix == '01' else 1}_state{1 if suffix == '01' else 0}_dataset_idx",
                }
                for src_key, dst_key in key_map.items():
                    if src_key in records:
                        dtype = np.int64 if src_key.endswith("sample_index") else np.float64
                        snf_arrays[dst_key] = np.asarray(records[src_key], dtype=dtype)
            arrays.update(snf_arrays)
        np.savez_compressed(analysis_dir / "validation_work_arrays.npz", **arrays)
        write_plotter_compat_arrays(
            analysis_dir,
            eval_result,
            raw_w01,
            raw_w10,
            tfep_w01,
            tfep_w10,
            raw_bootstrap_samples,
            tfep_bootstrap_samples,
            extra_arrays={**reweighted_arrays, **snf_arrays},
        )

    plot_raw_bootstrap_samples = raw_reweighted_bootstrap_samples if reweighting_enabled else raw_bootstrap_samples
    plot_tfep_bootstrap_samples = tfep_reweighted_bootstrap_samples if reweighting_enabled else tfep_bootstrap_samples
    cv_mod.maybe_plot_bootstrap(plot_raw_bootstrap_samples, plot_tfep_bootstrap_samples, analysis_dir / "bootstrap_deltaf.png")
    cv_mod.maybe_plot_convergence(
        convergence_rows,
        value_key="deltaf_std",
        ylabel="std(deltaf)",
        title="Held-out convergence speed",
        out_path=analysis_dir / "convergence_std.png",
    )
    cv_mod.maybe_plot_convergence(
        convergence_rows,
        value_key="mean_abs_error_vs_raw_validation",
        ylabel="mean abs error vs raw held-out BAR (kT)",
        title="Held-out convergence accuracy proxy",
        out_path=analysis_dir / "convergence_abs_error.png",
    )
    write_report(args, analysis_dir, summary, convergence_rows)

    del artifacts
    gc.collect()
    print(f"[done] holdout analysis written to: {analysis_dir}")


if __name__ == "__main__":
    main()
