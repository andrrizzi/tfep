#!/usr/bin/env python3
"""Holdout analysis for one-sided small-molecule TFEP.

This module is deliberately separate from the bidirectional TMBAR holdout
driver.  It trains only the ``state0 -> state1`` map with the standard TFEP
Boltzmann-KL objective and evaluates source-side FEP/TFEP log weights on held-out
state0 frames.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import MDAnalysis as mda
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional in lean envs.
    plt = None

from tfep.app import small_molecule_tfep as train_mod
from tfep.analysis import reweighting as rwlib
from tfep.analysis import tbar_cv_bootstrap as cv_mod
from tfep.analysis.tbar_holdout import choose_train_indices, complement_indices


def build_argparser() -> argparse.ArgumentParser:
    parser = train_mod.build_argparser()
    parser.description = (
        "One-sided TFEP holdout driver. Trains ref->tgt with the Boltzmann-KL "
        "objective and evaluates one-sided raw FEP vs one-sided TFEP FEP."
    )
    parser.add_argument("--train-count", type=int, default=9000)
    parser.add_argument("--split-mode", choices=["random", "blocked"], default="random")
    parser.add_argument("--split-seed-offset", type=int, default=2000)
    parser.add_argument("--analysis-subdir", type=str, default="holdout_small_train")
    parser.add_argument("--analysis-batch-size", type=int, default=None)
    parser.add_argument("--bootstrap-replicates", type=int, default=200)
    parser.add_argument("--bootstrap-block-size", type=int, default=1)
    parser.add_argument("--bootstrap-ci", type=float, default=0.95)
    parser.add_argument("--convergence-fractions", type=str, default="0.05,0.1,0.2,0.5,1.0")
    parser.add_argument("--convergence-replicates", type=int, default=100)
    parser.add_argument("--convergence-min-samples", type=int, default=100)
    parser.add_argument("--save-work-arrays", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _fep_bootstrap_summary(
    work: np.ndarray,
    *,
    n_boot: int,
    block_size: int,
    ci: float,
    seed: int,
) -> Tuple[Dict[str, Any], np.ndarray]:
    rng = np.random.default_rng(seed)
    estimates = np.empty(int(n_boot), dtype=np.float64)
    work = np.asarray(work, dtype=np.float64)
    for i in range(int(n_boot)):
        sample = cv_mod.resample_with_replacement(work, rng, int(block_size))
        estimates[i] = train_mod._stable_fep_deltaf(sample)
    estimates = estimates[np.isfinite(estimates)]
    alpha = 1.0 - float(ci)
    if estimates.size == 0:
        return {
            "n_boot": int(n_boot),
            "n_finite": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }, estimates
    return {
        "n_boot": int(n_boot),
        "n_finite": int(estimates.size),
        "mean": float(np.mean(estimates)),
        "std": float(np.std(estimates, ddof=1)) if estimates.size > 1 else float("nan"),
        "ci_low": float(np.quantile(estimates, alpha / 2.0)),
        "ci_high": float(np.quantile(estimates, 1.0 - alpha / 2.0)),
    }, estimates


def _weighted_fep_bootstrap_summary(
    work: np.ndarray,
    log_weights: np.ndarray,
    *,
    n_boot: int,
    block_size: int,
    ci: float,
    seed: int,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Bootstrap weighted one-sided FEP by resampling work/log-weight pairs."""
    rng = np.random.default_rng(seed)
    work = np.asarray(work, dtype=np.float64)
    log_weights = np.asarray(log_weights, dtype=np.float64)
    if work.shape != log_weights.shape:
        raise ValueError("work and log_weights must have the same shape")
    finite = np.isfinite(work) & np.isfinite(log_weights)
    work = work[finite]
    log_weights = log_weights[finite]
    indices = np.arange(work.size, dtype=int)
    estimates = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        sample_idx = cv_mod.resample_with_replacement(indices, rng, int(block_size))
        estimates[i] = rwlib.weighted_fep_deltaf(work[sample_idx], log_weights[sample_idx])
    estimates = estimates[np.isfinite(estimates)]
    alpha = 1.0 - float(ci)
    if estimates.size == 0:
        return {
            "n_boot": int(n_boot),
            "n_finite": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }, estimates
    return {
        "n_boot": int(n_boot),
        "n_finite": int(estimates.size),
        "mean": float(np.mean(estimates)),
        "std": float(np.std(estimates, ddof=1)) if estimates.size > 1 else float("nan"),
        "ci_low": float(np.quantile(estimates, alpha / 2.0)),
        "ci_high": float(np.quantile(estimates, 1.0 - alpha / 2.0)),
    }, estimates


def _one_sided_convergence_curve(
    method_name: str,
    work: np.ndarray,
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
            sample = cv_mod.subsample_without_replacement(work, int(sample_size), rng, int(block_size))
            estimates[i] = train_mod._stable_fep_deltaf(sample)
        estimates = estimates[np.isfinite(estimates)]
        if estimates.size == 0:
            rows.append({
                "method": method_name,
                "sample_size": int(sample_size),
                "n_finite": 0,
                "deltaf_mean": float("nan"),
                "deltaf_std": float("nan"),
                "mean_abs_error_vs_raw_validation": float("nan"),
            })
            continue
        rows.append({
            "method": method_name,
            "sample_size": int(sample_size),
            "n_finite": int(estimates.size),
            "deltaf_mean": float(np.mean(estimates)),
            "deltaf_std": float(np.std(estimates, ddof=1)) if estimates.size > 1 else float("nan"),
            "mean_abs_error_vs_raw_validation": float(np.mean(np.abs(estimates - float(reference_deltaf)))),
        })
    return rows


def _safe_ratio(num: Any, den: Any) -> float:
    try:
        n = float(num)
        d = float(den)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(n) or not np.isfinite(d) or abs(d) < 1.0e-15:
        return float("nan")
    return n / d


def _plot_bootstrap(raw_samples: np.ndarray, tfep_samples: np.ndarray, out_path: Path) -> None:
    if plt is None or len(raw_samples) == 0 or len(tfep_samples) == 0:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.hist(raw_samples, bins=40, histtype="step", label="raw one-sided FEP", density=True)
    ax.hist(tfep_samples, bins=40, histtype="step", label="TFEP one-sided FEP", density=True)
    ax.set_xlabel("Delta f (kT)")
    ax.set_ylabel("density")
    ax.set_title("One-sided bootstrap delta f")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_work_hist(raw_w: np.ndarray, tfep_w: np.ndarray, out_path: Path) -> None:
    if plt is None or len(raw_w) == 0 or len(tfep_w) == 0:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.hist(raw_w[np.isfinite(raw_w)], bins=60, histtype="step", label="raw work", density=True)
    ax.hist(tfep_w[np.isfinite(tfep_w)], bins=60, histtype="step", label="TFEP mapped work", density=True)
    ax.set_xlabel("work (kT)")
    ax.set_ylabel("density")
    ax.set_title("Held-out one-sided work distributions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_logj(logj: np.ndarray, out_path: Path) -> None:
    if plt is None or len(logj) == 0:
        return
    finite = logj[np.isfinite(logj)]
    if finite.size == 0:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.hist(finite, bins=60, histtype="stepfilled", alpha=0.55)
    ax.set_xlabel("log |det J|")
    ax.set_ylabel("count")
    ax.set_title("Held-out log-Jacobian distribution")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _write_report(
    args: argparse.Namespace,
    analysis_dir: Path,
    summary: Dict[str, Any],
    convergence_rows: Sequence[Dict[str, Any]],
) -> None:
    raw = summary["held_out"]["raw"]
    tfep = summary["held_out"]["tfep"]
    comparison = summary["comparison"]
    split = summary["split"]
    reweighting = summary.get("reweighting", {"enabled": False})
    lines = [
        "# One-Sided TFEP Holdout Report",
        "",
        "## Setup",
        "",
        "- Estimator: `one-sided TFEP`",
        "- Direction: `ref_gaff_am1bcc -> tgt_openff_nagl` (`state0 -> state1`)",
        f"- Training objective: `{args.objective}`",
        f"- Flow space: `{args.flow_space}`",
        f"- Training frames from source state: `{split['train_count_per_state']}`",
        f"- Validation source frames: `{split['val0_count']}`",
        f"- Batch size: `{args.batch_size}`",
        f"- Epochs: `{args.epochs}`",
        "",
        "## Held-Out One-Sided Estimates",
        "",
        f"- Raw one-sided FEP: `deltaf = {raw['deltaf']:.6f} kT`, `sigma = {raw['sigma']}`, `ESS ratio = {raw['ess']}`",
        f"- TFEP one-sided FEP: `deltaf = {tfep['deltaf']:.6f} kT`, `sigma = {tfep['sigma']}`, `ESS ratio = {tfep['ess']}`",
        f"- TFEP minus raw: `{comparison['tfep_minus_raw_validation_deltaf']:.6f} kT`",
        f"- Sigma ratio TFEP/raw: `{comparison['mean_sigma_ratio_tfep_over_raw']}`",
        f"- Bootstrap std ratio TFEP/raw: `{comparison['bootstrap_ratio_tfep_over_raw']}`",
        "",
        "Interpretation:",
        "This is a one-sided importance-sampling estimator. It is intentionally not BAR/TMBAR: no reverse work is required, and the free energy is computed as `-logmeanexp(-w)` on source-state held-out frames. The TFEP estimate uses the mapped work `u_tgt(M(x_ref)) - u_ref(x_ref) - logJ`.",
    ]
    if reweighting.get("enabled", False):
        raw_rw = summary["held_out"].get("raw_reweighted", {})
        tfep_rw = summary["held_out"].get("tfep_reweighted", {})
        source_diag = summary["reweighting"].get("source", {})
        lines.extend([
            "",
            "## Enhanced-Sampling Reweighting",
            "",
            "The source trajectory carries explicit dimensionless log weights. Unweighted one-sided estimates from a biased trajectory are diagnostics only; the equilibrium estimates are the reweighted entries.",
            f"- Raw reweighted FEP: `deltaf = {raw_rw.get('deltaf', float('nan')):.6f} kT`",
            f"- TFEP reweighted FEP: `deltaf = {tfep_rw.get('deltaf', float('nan')):.6f} kT`",
            f"- Source weight ESS/N: `{source_diag.get('ess_ratio', float('nan'))}`",
        ])
    if convergence_rows:
        lines.extend([
            "",
            "## Convergence",
            "",
            "`convergence.csv` reports repeated subsampling estimates for raw one-sided FEP and TFEP one-sided FEP.",
        ])
    (analysis_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_argparser().parse_args()
    analysis_root = Path(args.outdir).resolve()
    analysis_dir = analysis_root / args.analysis_subdir
    analysis_dir.mkdir(parents=True, exist_ok=True)

    if str(args.objective).lower() != "kl":
        raise ValueError("one-sided holdout requires --objective kl")
    if any(int(getattr(args, name, 0) or 0) for name in ("conditioning_shell_k1", "conditioning_shell_k2")):
        raise ValueError("one-sided standard TFEP disables shell conditioning; conditioning_shell_k1/k2 must be 0")

    probe_args = copy.deepcopy(args)
    probe_args.overwrite = False
    probe_args.outdir = str(analysis_dir / "_probe")
    probe_model, probe_ctx = train_mod.create_model_from_args(probe_args, outdir_override=probe_args.outdir)
    try:
        n_state0 = len(probe_model.make_dataset())
    finally:
        train_mod.close_model_openmm_workers(probe_model)
    n_state1 = len(mda.Universe(str(probe_ctx["topo_path"]), str(probe_ctx["traj1_path"])).trajectory)

    train_count = int(args.train_count)
    if train_count < int(args.batch_size):
        raise ValueError(
            f"--train-count={train_count} is smaller than --batch-size={args.batch_size}; "
            "training would produce zero dropped-last batches."
        )
    if train_count >= n_state0:
        raise ValueError(f"--train-count={train_count} must be smaller than source dataset length {n_state0}")

    seed0 = int(args.seed) + int(args.split_seed_offset)
    seed1 = int(args.seed) + int(args.split_seed_offset) + 17
    train0 = choose_train_indices(n_state0, train_count, args.split_mode, seed0)
    val0 = complement_indices(n_state0, train0)
    # State1 split files are compatibility metadata only for campaign tools.
    target_pick = min(train_count, n_state1 - 1)
    train1 = choose_train_indices(n_state1, target_pick, args.split_mode, seed1)
    val1 = complement_indices(n_state1, train1)

    np.save(analysis_dir / "state0_train_indices.npy", train0)
    np.save(analysis_dir / "state1_train_indices.npy", train1)
    np.save(analysis_dir / "state0_val_indices.npy", val0)
    np.save(analysis_dir / "state1_val_indices.npy", val1)

    manifest = {
        "analysis_dir": str(analysis_dir),
        "estimator": "onesided_tfep",
        "direction": "state0_to_state1",
        "n_state0": int(n_state0),
        "n_state1": int(n_state1),
        "train_count_per_state": int(train_count),
        "val0_count": int(len(val0)),
        "val1_count": int(len(val1)),
        "state1_split_used_for_training": False,
        "split_mode": args.split_mode,
        "state0_split_seed": int(seed0),
        "state1_split_seed": int(seed1),
        "train_fraction_state0": float(len(train0) / n_state0),
        "train_fraction_state1": float(len(train1) / n_state1),
    }
    (analysis_dir / "split_manifest.json").write_text(json.dumps(cv_mod.json_ready(manifest), indent=2), encoding="utf-8")
    if args.dry_run:
        print(f"[dry-run] wrote one-sided holdout split manifest to {analysis_dir / 'split_manifest.json'}")
        return

    train_dir = analysis_dir / "train"
    train_args = copy.deepcopy(args)
    train_args.outdir = str(train_dir)
    train_args.overwrite = True
    print(f"[onesided] train0={len(train0)} val0={len(val0)} target_metadata_train1={len(train1)} target_metadata_val1={len(val1)}")
    artifacts = train_mod.run_training(train_args, outdir_override=train_dir, train_indices_0=train0)
    eval_batch_size = int(args.analysis_batch_size) if args.analysis_batch_size is not None else int(args.batch_size)
    try:
        eval_result = train_mod.evaluate_one_sided_map(
            artifacts.model,
            subset_indices_0=val0,
            batch_size=eval_batch_size,
        )
    finally:
        train_mod.close_model_openmm_workers(artifacts.model)

    raw_summary = dict(eval_result["raw"])
    tfep_summary = dict(eval_result["tfep"])
    records = eval_result["state0_state1"]
    raw_w01 = np.asarray(records["raw_work"], dtype=np.float64)
    tfep_w01 = np.asarray(records["tfep_work"], dtype=np.float64)
    log_det = np.asarray(records["log_det_J"], dtype=np.float64)
    log_weights_arr = np.asarray(records.get("log_weights", []), dtype=np.float64)
    log_weights = log_weights_arr if len(log_weights_arr) == len(raw_w01) and len(log_weights_arr) > 0 else None
    reweighting_enabled = log_weights is not None
    raw_reweighted_summary = dict(eval_result.get("raw_reweighted", {})) if reweighting_enabled else None
    tfep_reweighted_summary = dict(eval_result.get("tfep_reweighted", {})) if reweighting_enabled else None
    raw_reweighted_bootstrap = None
    tfep_reweighted_bootstrap = None
    raw_reweighted_bootstrap_samples = None
    tfep_reweighted_bootstrap_samples = None

    raw_bootstrap, raw_bootstrap_samples = _fep_bootstrap_summary(
        raw_w01,
        n_boot=int(args.bootstrap_replicates),
        block_size=int(args.bootstrap_block_size),
        ci=float(args.bootstrap_ci),
        seed=int(args.seed) + 5000,
    )
    tfep_bootstrap, tfep_bootstrap_samples = _fep_bootstrap_summary(
        tfep_w01,
        n_boot=int(args.bootstrap_replicates),
        block_size=int(args.bootstrap_block_size),
        ci=float(args.bootstrap_ci),
        seed=int(args.seed) + 9000,
    )
    if reweighting_enabled:
        raw_reweighted_bootstrap, raw_reweighted_bootstrap_samples = _weighted_fep_bootstrap_summary(
            raw_w01,
            log_weights,
            n_boot=int(args.bootstrap_replicates),
            block_size=int(args.bootstrap_block_size),
            ci=float(args.bootstrap_ci),
            seed=int(args.seed) + 15000,
        )
        tfep_reweighted_bootstrap, tfep_reweighted_bootstrap_samples = _weighted_fep_bootstrap_summary(
            tfep_w01,
            log_weights,
            n_boot=int(args.bootstrap_replicates),
            block_size=int(args.bootstrap_block_size),
            ci=float(args.bootstrap_ci),
            seed=int(args.seed) + 19000,
        )

    n_common = min(len(raw_w01), len(tfep_w01))
    fractions = cv_mod.parse_fraction_list(args.convergence_fractions)
    sample_sizes = sorted({
        max(int(args.convergence_min_samples), int(round(frac * n_common)))
        for frac in fractions
        if int(round(frac * n_common)) > 0
    })
    sample_sizes = [n for n in sample_sizes if n <= n_common]
    convergence_rows: List[Dict[str, Any]] = []
    if sample_sizes:
        convergence_rows.extend(_one_sided_convergence_curve(
            "raw_fep",
            raw_w01,
            sample_sizes=sample_sizes,
            n_repeats=int(args.convergence_replicates),
            reference_deltaf=float(raw_summary["deltaf"]),
            block_size=int(args.bootstrap_block_size),
            seed=int(args.seed) + 23000,
        ))
        convergence_rows.extend(_one_sided_convergence_curve(
            "tfep_fep",
            tfep_w01,
            sample_sizes=sample_sizes,
            n_repeats=int(args.convergence_replicates),
            reference_deltaf=float(raw_summary["deltaf"]),
            block_size=int(args.bootstrap_block_size),
            seed=int(args.seed) + 27000,
        ))

    comparison = {
        "tfep_minus_raw_validation_deltaf": float(tfep_summary["deltaf"] - raw_summary["deltaf"]),
        "mean_sigma_ratio_tfep_over_raw": _safe_ratio(tfep_summary["sigma"], raw_summary["sigma"]),
        "mean_forward_std_ratio_tfep_over_raw": _safe_ratio(tfep_summary["w_forward_std"], raw_summary["w_forward_std"]),
        "mean_reverse_std_ratio_tfep_over_raw": float("nan"),
        "bootstrap_ratio_tfep_over_raw": _safe_ratio(tfep_bootstrap["std"], raw_bootstrap["std"]),
        "ess_ratio_tfep_over_raw": _safe_ratio(tfep_summary["ess"], raw_summary["ess"]),
    }
    summary = {
        "analysis": {
            "analysis_dir": str(analysis_dir),
            "estimator": "onesided_tfep",
            "direction": "state0_to_state1",
            "analysis_batch_size": int(eval_batch_size),
            "bootstrap_replicates": int(args.bootstrap_replicates),
            "bootstrap_block_size": int(args.bootstrap_block_size),
            "bootstrap_ci": float(args.bootstrap_ci),
            "free_energy_estimator": "-logmeanexp(-work)",
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
            "source": rwlib.log_weight_diagnostics(log_weights),
            "statistical_note": (
                "Unweighted raw/TFEP estimates from biased enhanced-sampling trajectories "
                "are diagnostics only; use raw_reweighted/tfep_reweighted for equilibrium estimates."
            ),
        }
        summary["held_out"]["raw_reweighted"] = raw_reweighted_summary
        summary["held_out"]["tfep_reweighted"] = tfep_reweighted_summary
        summary["held_out"]["raw_reweighted_bootstrap"] = raw_reweighted_bootstrap
        summary["held_out"]["tfep_reweighted_bootstrap"] = tfep_reweighted_bootstrap
    (analysis_dir / "summary.json").write_text(json.dumps(cv_mod.json_ready(summary), indent=2), encoding="utf-8")

    row = {
        "fold": 0,
        "train0": int(len(train0)),
        "val0": int(len(val0)),
        "train1": int(len(train1)),
        "val1": int(len(val1)),
        "raw_deltaf": float(raw_summary["deltaf"]),
        "raw_sigma": float(raw_summary["sigma"]),
        "raw_overlap": float(raw_summary.get("overlap", raw_summary.get("ess", np.nan))),
        "raw_direct_overlap": float("nan"),
        "tfep_deltaf": float(tfep_summary["deltaf"]),
        "tfep_sigma": float(tfep_summary["sigma"]),
        "tfep_overlap": float(tfep_summary.get("overlap", tfep_summary.get("ess", np.nan))),
        "tfep_direct_overlap": float("nan"),
        "raw_bootstrap_std": float(raw_bootstrap["std"]),
        "tfep_bootstrap_std": float(tfep_bootstrap["std"]),
        "raw_forward_std": float(raw_summary["w_forward_std"]),
        "raw_reverse_std": float("nan"),
        "tfep_forward_std": float(tfep_summary["w_forward_std"]),
        "tfep_reverse_std": float("nan"),
        "raw_ess": float(raw_summary["ess"]),
        "tfep_ess": float(tfep_summary["ess"]),
    }
    if reweighting_enabled:
        row.update({
            "raw_reweighted_deltaf": float(raw_reweighted_summary.get("deltaf", np.nan)) if raw_reweighted_summary else float("nan"),
            "tfep_reweighted_deltaf": float(tfep_reweighted_summary.get("deltaf", np.nan)) if tfep_reweighted_summary else float("nan"),
            "raw_reweighted_bootstrap_std": float(raw_reweighted_bootstrap.get("std", np.nan)) if raw_reweighted_bootstrap else float("nan"),
            "tfep_reweighted_bootstrap_std": float(tfep_reweighted_bootstrap.get("std", np.nan)) if tfep_reweighted_bootstrap else float("nan"),
            "source_weight_ess": float(summary["reweighting"]["source"].get("ess", np.nan)),
            "source_weight_ess_ratio": float(summary["reweighting"]["source"].get("ess_ratio", np.nan)),
        })
    cv_mod.save_rows_csv([row], analysis_dir / "fold_metrics.csv")
    cv_mod.save_rows_csv(convergence_rows, analysis_dir / "convergence.csv")

    if args.save_work_arrays:
        arrays = {
            "state0_state1_traj_idx": np.asarray(records["trajectory_sample_index"], dtype=np.int64),
            "state0_state1_dataset_idx": np.asarray(records["dataset_sample_index"], dtype=np.int64),
            "raw_w01": raw_w01,
            "tfep_w01": tfep_w01,
            "u_source_01": np.asarray(records["u_source"], dtype=np.float64),
            "u_target_raw_01": np.asarray(records["u_target_raw"], dtype=np.float64),
            "u_target_mapped_01": np.asarray(records["u_target_mapped"], dtype=np.float64),
            "log_det_J_01": log_det,
            "raw_bootstrap_deltaf": raw_bootstrap_samples,
            "tfep_bootstrap_deltaf": tfep_bootstrap_samples,
        }
        if reweighting_enabled:
            arrays["state0_log_weights"] = np.asarray(log_weights, dtype=np.float64)
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
        np.savez_compressed(analysis_dir / "validation_work_arrays.npz", **arrays)
        np.savez_compressed(analysis_dir / "pooled_work_arrays.npz", **arrays)

    _plot_bootstrap(raw_bootstrap_samples, tfep_bootstrap_samples, analysis_dir / "bootstrap_deltaf.png")
    if reweighting_enabled:
        _plot_bootstrap(
            raw_reweighted_bootstrap_samples,
            tfep_reweighted_bootstrap_samples,
            analysis_dir / "reweighted_bootstrap_deltaf.png",
        )
    cv_mod.maybe_plot_convergence(
        convergence_rows,
        value_key="deltaf_std",
        ylabel="std(delta f)",
        title="One-sided convergence speed",
        out_path=analysis_dir / "convergence_std.png",
    )
    cv_mod.maybe_plot_convergence(
        convergence_rows,
        value_key="mean_abs_error_vs_raw_validation",
        ylabel="mean abs error vs raw validation FEP (kT)",
        title="One-sided convergence accuracy proxy",
        out_path=analysis_dir / "convergence_abs_error.png",
    )
    _plot_work_hist(raw_w01, tfep_w01, analysis_dir / "work_histograms.png")
    _plot_logj(log_det, analysis_dir / "logJ_histogram.png")
    _write_report(args, analysis_dir, summary, convergence_rows)

    del artifacts
    gc.collect()
    print(f"[done] one-sided TFEP holdout analysis written to: {analysis_dir}")


if __name__ == "__main__":
    main()
