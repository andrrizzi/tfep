#!/usr/bin/env python3
"""examples/ozone_tmbar_bidirectional_contrib.py

Pipeline (same as Flow_ozone.py), but wired through tfep.contrib modules:

1) Run MD in state0 (reference) and state1 (target)
2) Plot DOF overlap (r01, r12, angle)
3) Baseline cross-energy evaluation (FEP fwd/rev + BAR) + work overlap plot
4) Bidirectional TFEP training (TMBARMapBase)
5) Plot ΔF vs epoch + loss curves
6) Plot final TFEP generalized-work overlap
7) Optional: bootstrap CI on final works
8) Optional: K-fold out-of-sample evaluation (avoids in-sample optimism)

Notes
-----
- OpenMM is optional for tfep, but required to actually run this example.
- This script assumes you have installed the tfep package (or are running from a checkout)
  and that the contrib package is available (tfep.contrib.*).
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np

import torch

# Optional dependencies with friendly errors
try:
    import pint
except Exception as e:
    raise RuntimeError(
        "This example requires 'pint' (used by tfep.potentials.OpenMMPotential). "
        "Install it with: pip install pint"
    ) from e

try:
    from lightning.pytorch import Trainer
    from lightning.pytorch.loggers import CSVLogger
except Exception as e:
    raise RuntimeError(
        "This example requires Lightning. Install it with: pip install lightning"
    ) from e

import tfep
from tfep.potentials.openmm import OpenMMPotential

from tfep.contrib.ozone.systems import (
    get_platform,
    system_reference_harmonic,
    system_target_morse_anharm,
    run_md,
)
from tfep.contrib.ozone.analysis import (
    plot_dofs,
    read_potential_energies_kjmol,
    eval_potential_on_frames_kjmol,
    fep_forward_df,
    fep_reverse_df,
    bar_df,
    plot_baseline_works,
    plot_losses_from_csv,
    plot_df_vs_epoch,
    plot_tfep_work_overlap,
    make_kfold_splits,
    compute_tfep_works_on_indices,
    bootstrap_tfep_final_epoch,
    plot_bootstrap_df,
)
from tfep.contrib.flows import FlowSpec
from tfep.contrib.ozone.map import OzoneBidirectionalTMBARMap


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="ozone_tmbar")
    ap.add_argument("--openmm-platform", type=str, default="OpenCL")
    ap.add_argument("--torch-accelerator", type=str, default="cpu", choices=["cpu", "gpu"])

    ap.add_argument("--steps", type=int, default=600000)
    ap.add_argument("--report-interval", type=int, default=50)

    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=0)

    ap.add_argument(
        "--flow-space",
        type=str,
        default="zmat",
        choices=["zmat", "cart"],
        help="'zmat' internal-coordinate flow (triatomic gas-phase); 'cart' Cartesian MAF.",
    )
    ap.add_argument("--n-maf-layers", type=int, default=6)
    ap.add_argument("--maf-hidden-layers", type=int, default=2)
    ap.add_argument("--maf-weight-norm", action="store_true")
    ap.add_argument("--maf-no-init-identity", action="store_true")

    # Z-matrix flow knobs
    ap.add_argument("--zmat-layers", type=int, default=6)
    ap.add_argument("--zmat-hidden", type=int, default=64)
    ap.add_argument("--zmat-n-hidden", type=int, default=2)
    ap.add_argument("--zmat-scale", type=float, default=0.8)

    # BAR-like regularizer
    ap.add_argument("--bar-reg-weight", type=float, default=0.0, help="0 disables; typical 1e-3..1e-1")

    ap.add_argument("--overwrite", action="store_true")

    # Optional bootstrap on final works
    ap.add_argument("--bootstrap", action="store_true")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--boot-ci", type=float, default=0.95)
    ap.add_argument("--boot-max-n", type=int, default=200000)
    ap.add_argument(
        "--boot-block-len",
        type=int,
        default=0,
        metavar="L",
        help="Moving-block bootstrap block length (0 => iid bootstrap).",
    )

    # Optional K-fold out-of-sample evaluation
    ap.add_argument(
        "--kfold",
        type=int,
        default=1,
        metavar="K",
        help="If K>1: K-fold CV (train on K-1 contiguous blocks, evaluate TFEP works on held-out fold).",
    )
    ap.add_argument("--kfold-seed", type=int, default=123)
    ap.add_argument("--kfold-shuffle", action="store_true")
    ap.add_argument("--kfold-contiguous", action="store_true", default=True)

    args = ap.parse_args()

    try:
        torch.set_float32_matmul_precision("medium")
    except Exception:
        pass

    # (No longer needed) device-safe geometry utilities are part of tfep core.

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    U = pint.UnitRegistry()
    TEMPERATURE = 298.15 * U.kelvin
    kT_kj = (TEMPERATURE * U.molar_gas_constant).to(U.kJ / U.mol).magnitude

    platform = get_platform(args.openmm_platform)

    # --- Systems ---
    sys0 = system_reference_harmonic()
    sys1 = system_target_morse_anharm()

    # --- MD ---
    md0 = outdir / "md_state0"
    md1 = outdir / "md_state1"

    dcd0 = md0 / "traj.dcd"
    csv0 = md0 / "energies.csv"
    pdb0 = md0 / "ozone.pdb"

    dcd1 = md1 / "traj.dcd"
    csv1 = md1 / "energies.csv"
    pdb1 = md1 / "ozone.pdb"

    if args.overwrite or not (dcd0.exists() and csv0.exists() and pdb0.exists()):
        print(f"[MD] Running state0 MD -> {md0}")
        run_md(sys0, md0, float(TEMPERATURE.magnitude), args.steps, args.report_interval, platform, seed=1)
    else:
        print(f"[MD] state0 exists -> {md0}")

    if args.overwrite or not (dcd1.exists() and csv1.exists() and pdb1.exists()):
        print(f"[MD] Running state1 MD -> {md1}")
        run_md(sys1, md1, float(TEMPERATURE.magnitude), args.steps, args.report_interval, platform, seed=2)
    else:
        print(f"[MD] state1 exists -> {md1}")

    # --- DOF overlap ---
    print("[PLOT] DOF overlap (state0 vs state1)")
    plot_dofs(outdir, dcd0, dcd1)

    # --- Baseline cross-energy evaluation ---
    print("[BASELINE] Evaluating cross energies ...")
    U0_x0 = read_potential_energies_kjmol(csv0)
    U1_x1 = read_potential_energies_kjmol(csv1)
    U1_x0 = eval_potential_on_frames_kjmol(sys1, dcd0, platform)
    U0_x1 = eval_potential_on_frames_kjmol(sys0, dcd1, platform)

    n0 = min(len(U0_x0), len(U1_x0))
    n1 = min(len(U1_x1), len(U0_x1))
    U0_x0, U1_x0 = U0_x0[:n0], U1_x0[:n0]
    U1_x1, U0_x1 = U1_x1[:n1], U0_x1[:n1]

    w01 = (U1_x0 - U0_x0) / kT_kj
    w10 = (U0_x1 - U1_x1) / kT_kj

    df_fep_01 = fep_forward_df(w01) * kT_kj
    df_fep_10 = fep_reverse_df(w10) * kT_kj
    df_bar_01 = bar_df(w01, w10) * kT_kj

    print(f"[BASELINE] ΔF FEP(0→1)={df_fep_01:.3f}  ΔF FEP(1→0)={df_fep_10:.3f}  BAR={df_bar_01:.3f}")
    plot_baseline_works(outdir, (U1_x0 - U0_x0), (U0_x1 - U1_x1), df_bar_01)

    # --- Potentials for TFEP ---
    pot0 = OpenMMPotential(
        system=sys0,
        platform=platform,
        positions_unit=U.angstrom,
        energy_unit=U.kJ / U.mol,
        system_name="ozone_state0",
        precompute_gradient=True,
    )
    pot1 = OpenMMPotential(
        system=sys1,
        platform=platform,
        positions_unit=U.angstrom,
        energy_unit=U.kJ / U.mol,
        system_name="ozone_state1",
        precompute_gradient=True,
    )

    # --- TFEP output directories ---
    tfep_dir = outdir / "tfep_bidirectional"
    if args.overwrite and tfep_dir.exists():
        shutil.rmtree(tfep_dir)
    tfep_logs = tfep_dir / "tfep_logs"

    # Clean stale preallocated arrays if asked.
    if args.overwrite:
        shutil.rmtree(tfep_logs, ignore_errors=True)
        shutil.rmtree(tfep_dir / "lightning_logs", ignore_errors=True)
    tfep_logs.mkdir(parents=True, exist_ok=True)

    accelerator = "gpu" if (args.torch_accelerator == "gpu" and torch.cuda.is_available()) else "cpu"

    # Precompute self energies in reduced units for work construction.
    U0_x0_dimless_full = U0_x0 / kT_kj
    U1_x1_dimless_full = U1_x1 / kT_kj
    n_frames = int(min(U0_x0_dimless_full.shape[0], U1_x1_dimless_full.shape[0]))

    # Flow selection through FlowSpec (developer control)
    if args.flow_space == "zmat":
        flow_spec = FlowSpec(
            name="triatomic_zmat",
            kwargs=dict(
                n_layers=int(args.zmat_layers),
                hidden_dim=int(args.zmat_hidden),
                n_hidden=int(args.zmat_n_hidden),
                scale=float(args.zmat_scale),
            ),
        )
        mapped_atoms = [0, 1, 2]
        conditioning_atoms = None
    else:
        flow_spec = FlowSpec(
            name="cartesian_maf",
            kwargs=dict(
                n_maf_layers=int(args.n_maf_layers),
                hidden_layers=int(args.maf_hidden_layers),
                weight_norm=bool(args.maf_weight_norm),
                initialize_identity=not bool(args.maf_no_init_identity),
            ),
        )
        # Cartesian baseline: map terminals, condition on center (same as Flow_ozone.py)
        mapped_atoms = [0, 2]
        conditioning_atoms = [1]

    df_bar_epochs: list[float] = []
    df_fwd_epochs: list[float] = []
    df_rev_epochs: list[float] = []

    last_w01: Optional[np.ndarray] = None
    last_w10: Optional[np.ndarray] = None
    boot: Optional[dict] = None

    if int(args.kfold) > 1:
        K = int(args.kfold)
        print(f"[TFEP] Running K-fold out-of-sample evaluation (K={K}) ...")
        splits = make_kfold_splits(
            n_frames,
            K,
            seed=int(args.kfold_seed),
            shuffle=bool(args.kfold_shuffle),
            contiguous=bool(args.kfold_contiguous),
        )

        w01_all, w10_all = [], []
        df_fold = []

        for fold, (train_idx, eval_idx) in enumerate(splits):
            fold_dir = outdir / f"tfep_kfold_fold{fold:02d}"
            tfep_logs_fold = fold_dir / "tfep_logs"
            if args.overwrite and fold_dir.exists():
                shutil.rmtree(fold_dir)
            tfep_logs_fold.mkdir(parents=True, exist_ok=True)

            model = OzoneBidirectionalTMBARMap(
                potential_0=pot0,
                potential_1=pot1,
                topology_file_path=str(pdb0),
                coordinates_file_path=str(dcd0),
                coordinates_file_path_2=str(dcd1),
                temperature=TEMPERATURE,
                batch_size=int(args.batch_size),
                mapped_atoms=mapped_atoms,
                conditioning_atoms=conditioning_atoms,
                origin_atom=1,
                axes_atoms=[0, 2],
                flow_spec=flow_spec,
                tfep_logger_dir_path=str(tfep_logs_fold),
                dataloader_kwargs={"num_workers": int(args.num_workers)},
                train_indices=train_idx.tolist(),
                bar_reg_weight=float(args.bar_reg_weight),
            )

            logger = CSVLogger(save_dir=str(fold_dir), name="lightning_logs")
            print(
                f"[TFEP] Fold {fold+1}/{K}: training on {train_idx.size} frames, evaluating on {eval_idx.size} frames"
            )
            trainer = Trainer(
                max_epochs=int(args.epochs),
                accelerator=accelerator,
                devices=1,
                default_root_dir=str(fold_dir),
                enable_checkpointing=False,
                logger=logger,
                log_every_n_steps=1,
            )
            trainer.fit(model)

            w01_eval, w10_eval = compute_tfep_works_on_indices(
                model,
                eval_idx,
                U0_x0_dimless_full=U0_x0_dimless_full,
                U1_x1_dimless_full=U1_x1_dimless_full,
                kT=kT_kj,
                batch_size=max(1, int(args.batch_size)),
                num_workers=int(args.num_workers),
                device="cuda" if accelerator == "gpu" else None,
            )

            w01_all.append(w01_eval)
            w10_all.append(w10_eval)

            df_fold_kj = bar_df(w01_eval, w10_eval) * kT_kj
            df_fold.append(df_fold_kj)
            print(f"[TFEP] Fold {fold+1}/{K}: ΔF_BAR (held-out) = {df_fold_kj:.4f} kJ/mol")

        last_w01 = np.concatenate(w01_all) if w01_all else np.asarray([], dtype=float)
        last_w10 = np.concatenate(w10_all) if w10_all else np.asarray([], dtype=float)

        df_fwd_cv = fep_forward_df(last_w01) * kT_kj
        df_rev_cv = fep_reverse_df(last_w10) * kT_kj
        df_bar_cv = bar_df(last_w01, last_w10) * kT_kj

        df_fwd_epochs.append(df_fwd_cv)
        df_rev_epochs.append(df_rev_cv)
        df_bar_epochs.append(df_bar_cv)

        with open(outdir / "kfold_summary.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["fold", "df_bar_kj"])
            for i, v in enumerate(df_fold):
                w.writerow([i, float(v)])
            w.writerow(["cv_fwd", float(df_fwd_cv)])
            w.writerow(["cv_rev", float(df_rev_cv)])
            w.writerow(["cv_bar", float(df_bar_cv)])

        print(
            f"[TFEP] K-fold (out-of-sample) ΔF: FWD={df_fwd_cv:.4f}  REV={df_rev_cv:.4f}  BAR={df_bar_cv:.4f} kJ/mol"
        )

    else:
        model = OzoneBidirectionalTMBARMap(
            potential_0=pot0,
            potential_1=pot1,
            topology_file_path=str(pdb0),
            coordinates_file_path=str(dcd0),
            coordinates_file_path_2=str(dcd1),
            temperature=TEMPERATURE,
            batch_size=int(args.batch_size),
            mapped_atoms=mapped_atoms,
            conditioning_atoms=conditioning_atoms,
            origin_atom=1,
            axes_atoms=[0, 2],
            flow_spec=flow_spec,
            tfep_logger_dir_path=str(tfep_logs),
            dataloader_kwargs={"num_workers": int(args.num_workers)},
            bar_reg_weight=float(args.bar_reg_weight),
        )

        logger = CSVLogger(save_dir=str(tfep_dir), name="lightning_logs")

        print("[TFEP] Bidirectional training started ...")
        trainer = Trainer(
            max_epochs=int(args.epochs),
            accelerator=accelerator,
            devices=1,
            default_root_dir=str(tfep_dir),
            enable_checkpointing=False,
            logger=logger,
            log_every_n_steps=1,
        )
        trainer.fit(model)

        tmb_logger = tfep.io.TMBARLogger(str(tfep_logs))

        for e in range(int(args.epochs)):
            d01 = tmb_logger.read_train_tensors(epoch_idx=e, state_mapping=(0, 1))
            d10 = tmb_logger.read_train_tensors(epoch_idx=e, state_mapping=(1, 0))

            idx01 = d01["trajectory_sample_index"].detach().cpu().numpy().astype(int)
            idx10 = d10["trajectory_sample_index"].detach().cpu().numpy().astype(int)

            w01_tfep = (d01["potential"] - d01["log_det_J"]).detach().cpu().numpy() - U0_x0_dimless_full[idx01]
            w10_tfep = (d10["potential"] - d10["log_det_J"]).detach().cpu().numpy() - U1_x1_dimless_full[idx10]

            df_fwd = fep_forward_df(w01_tfep) * kT_kj
            df_rev = fep_reverse_df(w10_tfep) * kT_kj
            df_bar = bar_df(w01_tfep, w10_tfep) * kT_kj

            df_fwd_epochs.append(df_fwd)
            df_rev_epochs.append(df_rev)
            df_bar_epochs.append(df_bar)

            last_w01 = w01_tfep
            last_w10 = w10_tfep

            print(
                f"[EPOCH {e}] TFEP-FWD={df_fwd:.4f} TFEP-REV={df_rev:.4f} TFEP-BAR={df_bar:.4f} kJ/mol",
                flush=True,
            )

    # --- Plots ---
    # Loss curve
    try:
        metrics_csv = Path(logger.log_dir) / "metrics.csv"
        if metrics_csv.exists():
            plot_losses_from_csv(outdir, metrics_csv)
    except Exception:
        pass

    plot_df_vs_epoch(
        outdir,
        df_bar_kj=df_bar_epochs,
        df_fwd_kj=df_fwd_epochs,
        df_rev_kj=df_rev_epochs,
        baseline={
            "baseline BAR": float(df_bar_01),
            "baseline FEP fwd": float(df_fep_01),
            "baseline FEP rev": float(df_fep_10),
        },
    )

    if last_w01 is not None and last_w10 is not None and len(df_bar_epochs) > 0:
        plot_tfep_work_overlap(outdir, last_w01 * kT_kj, last_w10 * kT_kj, df_bar_epochs[-1])

    # --- Bootstrap (final works) ---
    if args.bootstrap and last_w01 is not None and last_w10 is not None and len(df_bar_epochs) > 0:
        point = {
            "fwd": float(df_fwd_epochs[-1]) if df_fwd_epochs else float("nan"),
            "rev": float(df_rev_epochs[-1]) if df_rev_epochs else float("nan"),
            "bar": float(df_bar_epochs[-1]) if df_bar_epochs else float("nan"),
        }
        boot = bootstrap_tfep_final_epoch(
            last_w01,
            last_w10,
            kT_kj=kT_kj,
            n_boot=int(args.n_boot),
            ci=float(args.boot_ci),
            max_n=int(args.boot_max_n),
            block_len=(int(args.boot_block_len) if int(args.boot_block_len) > 0 else None),
            seed=123,
        )
        plot_bootstrap_df(outdir, boot, point)
        # Don't retain big arrays.
        boot.pop("df_fwd_boot", None)
        boot.pop("df_rev_boot", None)
        boot.pop("df_bar_boot", None)

    # --- Save compact results ---
    results_payload = dict(
        kT_kj=float(kT_kj),
        baseline_df_fep_01=float(df_fep_01),
        baseline_df_fep_10=float(df_fep_10),
        baseline_df_bar=float(df_bar_01),
        df_tfep_fwd_epochs=[float(x) for x in df_fwd_epochs],
        df_tfep_rev_epochs=[float(x) for x in df_rev_epochs],
        df_tfep_bar_epochs=[float(x) for x in df_bar_epochs],
    )
    if boot is not None:
        results_payload.update(
            dict(
                boot_ci=float(boot.get("ci", float("nan"))),
                boot_n_boot=int(boot.get("n_boot", -1)),
                boot_block_len=int(boot.get("block_len", 0)),
                boot_fwd=boot.get("fwd", {}),
                boot_rev=boot.get("rev", {}),
                boot_bar=boot.get("bar", {}),
            )
        )

    np.savez_compressed(outdir / "results_summary.npz", **results_payload)
    print(f"[DONE] Wrote outputs to: {outdir}")


if __name__ == "__main__":
    main()
