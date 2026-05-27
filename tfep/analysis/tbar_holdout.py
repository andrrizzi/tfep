#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import gc
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

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
        "--split-mode",
        choices=["random", "blocked"],
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


def write_report(
    args: argparse.Namespace,
    analysis_dir: Path,
    summary: Dict[str, Any],
    convergence_rows: List[Dict[str, Any]],
) -> None:
    raw_validation = summary["held_out"]["raw"]
    tfep_validation = summary["held_out"]["tfep"]
    snf_validation = summary["held_out"].get("stochastic_path_tfep")
    comparison = summary["comparison"]
    split = summary["split"]

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
        "",
        "## Held-Out Validation",
        "",
        f"- Raw BAR on held-out validation: `deltaf = {raw_validation['deltaf']:.6f} kT`, `sigma = {raw_validation['sigma']}`",
        f"- TFEP BAR on held-out validation: `deltaf = {tfep_validation['deltaf']:.6f} kT`, `sigma = {tfep_validation['sigma']}`",
        f"- Raw BAR-consistent overlap: `{raw_validation.get('overlap', np.nan):.4f}`",
        f"- TFEP BAR-consistent overlap: `{tfep_validation.get('overlap', np.nan):.4f}`",
        f"- Raw direct overlap (w_F vs w_R): `{raw_validation.get('direct_overlap', np.nan):.4f}`",
        f"- TFEP direct overlap (w_F vs w_R): `{tfep_validation.get('direct_overlap', np.nan):.4f}`",
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
                "`convergence.csv` reports repeated held-out subsampling curves for raw BAR and TFEP BAR. "
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
    snf_summary: Optional[Dict[str, Any]] = None,
    snf_bootstrap: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    row = {
        "fold": 0,
        "train0": int(manifest["train_count_per_state"]),
        "val0": int(manifest["val0_count"]),
        "train1": int(manifest["train_count_per_state"]),
        "val1": int(manifest["val1_count"]),
        "raw_deltaf": float(raw_summary["deltaf"]),
        "raw_sigma": float(raw_summary["sigma"]),
        "raw_overlap": float(raw_summary.get("overlap", np.nan)),
        "raw_direct_overlap": float(raw_summary.get("direct_overlap", np.nan)),
        "tfep_deltaf": float(tfep_summary["deltaf"]),
        "tfep_sigma": float(tfep_summary["sigma"]),
        "tfep_overlap": float(tfep_summary.get("overlap", np.nan)),
        "tfep_direct_overlap": float(tfep_summary.get("direct_overlap", np.nan)),
        "raw_bootstrap_std": float(raw_bootstrap["std"]),
        "tfep_bootstrap_std": float(tfep_bootstrap["std"]),
        "raw_forward_std": float(raw_summary["w_forward_std"]),
        "raw_reverse_std": float(raw_summary["w_reverse_std"]),
        "tfep_forward_std": float(tfep_summary["w_forward_std"]),
        "tfep_reverse_std": float(tfep_summary["w_reverse_std"]),
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
    train0 = choose_train_indices(n_state0, train_count, args.split_mode, seed0)
    train1 = choose_train_indices(n_state1, train_count, args.split_mode, seed1)
    val0 = complement_indices(n_state0, train0)
    val1 = complement_indices(n_state1, train1)

    np.save(analysis_dir / "state0_train_indices.npy", train0)
    np.save(analysis_dir / "state1_train_indices.npy", train1)
    np.save(analysis_dir / "state0_val_indices.npy", val0)
    np.save(analysis_dir / "state1_val_indices.npy", val1)

    manifest = {
        "analysis_dir": str(analysis_dir),
        "n_state0": int(n_state0),
        "n_state1": int(n_state1),
        "train_count_per_state": int(train_count),
        "val0_count": int(len(val0)),
        "val1_count": int(len(val1)),
        "split_mode": args.split_mode,
        "state0_split_seed": int(seed0),
        "state1_split_seed": int(seed1),
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

    comparison = cv_mod.pooled_summary([raw_summary], [tfep_summary])
    comparison["tfep_minus_raw_validation_deltaf"] = float(tfep_summary["deltaf"] - raw_summary["deltaf"])
    if snf_summary is not None:
        comparison["snf_minus_raw_validation_deltaf"] = float(snf_summary["deltaf"] - raw_summary["deltaf"])
        comparison["snf_minus_tfep_validation_deltaf"] = float(snf_summary["deltaf"] - tfep_summary["deltaf"])
        comparison["sigma_ratio_snf_over_raw"] = float(snf_summary["sigma"] / raw_summary["sigma"]) if float(raw_summary["sigma"]) != 0.0 else float("nan")

    summary = {
        "analysis": {
            "analysis_dir": str(analysis_dir),
            "analysis_batch_size": int(eval_batch_size),
            "bootstrap_replicates": int(args.bootstrap_replicates),
            "bootstrap_block_size": int(args.bootstrap_block_size),
            "bootstrap_ci": float(args.bootstrap_ci),
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
                raw_summary,
                tfep_summary,
                raw_bootstrap,
                tfep_bootstrap,
                snf_summary=snf_summary,
                snf_bootstrap=snf_bootstrap,
            )
        ],
        analysis_dir / "fold_metrics.csv",
    )
    cv_mod.save_rows_csv(convergence_rows, analysis_dir / "convergence.csv")

    if args.save_work_arrays:
        snf_arrays: Dict[str, np.ndarray] = {}
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
            extra_arrays=snf_arrays,
        )

    cv_mod.maybe_plot_bootstrap(raw_bootstrap_samples, tfep_bootstrap_samples, analysis_dir / "bootstrap_deltaf.png")
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
