"""Ozone demo analysis utilities.

Contains:
- internal DOF histograms (MDAnalysis)
- baseline FEP/BAR estimators and overlap plots
- TFEP diagnostics: loss curves, ΔF vs epoch, final work overlap
- bootstrap CI for ΔF
- K-fold utilities and out-of-sample work evaluation
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
import MDAnalysis as mda


# =============================================================================
# Internal DOFs (MDAnalysis)
# =============================================================================

def internal_dofs_from_dcd(dcd_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (r01, r12, angle) from a triatomic DCD trajectory."""
    d01, d12, ang = [], [], []
    with mda.coordinates.DCD.DCDReader(str(dcd_path)) as trj:
        for ts in trj:
            x = ts.positions  # Angstrom
            r01 = x[0] - x[1]
            r12 = x[2] - x[1]
            dist01 = float(np.linalg.norm(r01))
            dist12 = float(np.linalg.norm(r12))
            c = float(np.dot(r01 / dist01, r12 / dist12))
            c = np.clip(c, -1.0, 1.0)
            theta = float(np.arccos(c))
            d01.append(dist01)
            d12.append(dist12)
            ang.append(theta)
    return np.asarray(d01), np.asarray(d12), np.asarray(ang)


def plot_dofs(outdir: Path, dcd0: Path, dcd1: Path) -> None:
    r01_0, r12_0, th_0 = internal_dofs_from_dcd(dcd0)
    r01_1, r12_1, th_1 = internal_dofs_from_dcd(dcd1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    axes[0].hist(r01_0, bins=80, alpha=0.7, label="state0/ref")
    axes[0].hist(r01_1, bins=80, alpha=0.5, label="state1/tgt")
    axes[1].hist(r12_0, bins=80, alpha=0.7, label="state0/ref")
    axes[1].hist(r12_1, bins=80, alpha=0.5, label="state1/tgt")
    axes[2].hist(th_0, bins=80, alpha=0.7, label="state0/ref")
    axes[2].hist(th_1, bins=80, alpha=0.5, label="state1/tgt")

    axes[0].set_xlabel("r(0-1) [Å]")
    axes[1].set_xlabel("r(1-2) [Å]")
    axes[2].set_xlabel("angle(0-1-2) [rad]")
    for ax in axes:
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(outdir / "fig_dofs_ref_vs_target.png", dpi=200)
    plt.close(fig)


# =============================================================================
# Energies, FEP/BAR
# =============================================================================

def _is_float(x: str) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


def read_potential_energies_kjmol(energies_csv: Path) -> np.ndarray:
    """Read potential energies [kJ/mol] from an OpenMM StateDataReporter CSV."""
    vals = []
    with open(energies_csv, "r", newline="") as f:
        reader = csv.reader(f)
        header = None
        for row in reader:
            if not row:
                continue
            if row[0].startswith("#"):
                header = [c.strip().strip('"') for c in row]
                continue
            if header is None and any(not _is_float(x) for x in row):
                header = [c.strip().strip('"') for c in row]
                continue

            if header is None:
                # Heuristic: second column is potential energy
                vals.append(float(row[1]))
            else:
                idx = None
                for i, name in enumerate(header):
                    if "Potential" in name and "Energy" in name:
                        idx = i
                        break
                if idx is None:
                    idx = 1
                vals.append(float(row[idx]))
    return np.asarray(vals, dtype=float)


def _require_openmm():
    try:
        import openmm
        import openmm.unit as unit
    except Exception as e:
        raise ImportError(
            "This function requires OpenMM (pip install openmm). "
            "OpenMM is an optional dependency of tfep."
        ) from e
    return openmm, unit


def eval_potential_on_frames_kjmol(system, dcd_path: Path, platform) -> np.ndarray:
    """Evaluate OpenMM potential energy on every frame of a DCD (kJ/mol)."""
    openmm, unit = _require_openmm()
    integrator = openmm.VerletIntegrator(2.0 * unit.femtosecond)
    context = openmm.Context(system, integrator, platform)
    out = []
    with mda.coordinates.DCD.DCDReader(str(dcd_path)) as trj:
        for ts in trj:
            pos_nm = unit.Quantity(ts.positions, unit.angstrom).in_units_of(unit.nanometer)
            context.setPositions(pos_nm)
            state = context.getState(getEnergy=True)
            out.append(state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole))
    return np.asarray(out, dtype=float)


def logmeanexp(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    m = np.max(a)
    return float(m + np.log(np.mean(np.exp(a - m))))


def fep_forward_df(work_dimless: np.ndarray) -> float:
    """Forward FEP Δf in kT units: -log <exp(-w)>."""
    return -logmeanexp(-np.asarray(work_dimless, dtype=float))


def fep_reverse_df(work_dimless: np.ndarray) -> float:
    """Reverse FEP Δf in kT units:  log <exp(-w)> (with w defined for 1→0)."""
    return logmeanexp(-np.asarray(work_dimless, dtype=float))


def bar_df(work_fwd: np.ndarray, work_rev: np.ndarray, tol: float = 1e-10, max_iter: int = 200) -> float:
    """Solve Bennett's acceptance ratio equation for Δf in kT units."""
    wf = np.asarray(work_fwd, dtype=float)
    wr = np.asarray(work_rev, dtype=float)

    lo = min(np.min(wf) - 50.0, -np.max(wr) - 50.0)
    hi = max(np.max(wf) + 50.0, -np.min(wr) + 50.0)

    def F(df):
        left = np.sum(1.0 / (1.0 + np.exp(wf - df)))
        right = np.sum(1.0 / (1.0 + np.exp(wr + df)))
        return left - right

    f_lo = F(lo)
    f_hi = F(hi)
    if f_lo * f_hi > 0:
        for _ in range(12):
            lo -= 50.0
            hi += 50.0
            f_lo = F(lo)
            f_hi = F(hi)
            if f_lo * f_hi <= 0:
                break
        else:
            return 0.5 * (lo + hi)

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = F(mid)
        if abs(f_mid) < tol or (hi - lo) < tol:
            return float(mid)
        if f_lo * f_mid <= 0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
    return float(0.5 * (lo + hi))


# =============================================================================
# Plotting helpers
# =============================================================================

def plot_baseline_works(outdir: Path, w01_kj: np.ndarray, w10_kj: np.ndarray, df_bar_kj: float) -> None:
    fig = plt.figure(figsize=(7, 3))
    plt.hist(w01_kj, bins=120, alpha=0.6, label="baseline work 0→1")
    plt.hist(-w10_kj, bins=120, alpha=0.6, label="baseline -work 1→0")
    plt.axvline(df_bar_kj, linestyle="--")
    plt.xlabel("work [kJ/mol]")
    plt.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(outdir / "fig_baseline_work_overlap.png", dpi=200)
    plt.close(fig)


def plot_losses_from_csv(outdir: Path, metrics_csv: Path) -> None:
    """Plot Lightning losses from metrics.csv (no pandas dependency)."""
    # metrics.csv has columns: epoch, step, loss, loss_0_1, loss_1_0, ... depending on logger
    with open(metrics_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader]

    if not rows:
        return

    def col_as_float(name: str) -> np.ndarray:
        out = []
        for r in rows:
            v = r.get(name, "")
            try:
                out.append(float(v))
            except Exception:
                out.append(np.nan)
        return np.asarray(out, dtype=float)

    x = col_as_float("step")
    fig = plt.figure(figsize=(7, 3))
    for col in ("loss", "loss_0_1", "loss_1_0"):
        if col in rows[0]:
            y = col_as_float(col)
            mask = np.isfinite(x) & np.isfinite(y)
            if np.any(mask):
                plt.plot(x[mask], y[mask], label=col)
    plt.xlabel("global step")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    fig.savefig(outdir / "fig_training_losses.png", dpi=200)
    plt.close(fig)


def plot_df_vs_epoch(
    outdir: Path,
    df_bar_kj: Sequence[float],
    df_fwd_kj: Sequence[float],
    df_rev_kj: Sequence[float],
    baseline: Dict[str, float],
) -> None:
    epochs = np.arange(len(df_bar_kj), dtype=int)
    fig = plt.figure(figsize=(7, 3))
    if len(df_bar_kj) > 0:
        plt.plot(epochs, np.asarray(df_bar_kj), marker="o", label="TFEP BAR")
    if len(df_fwd_kj) > 0:
        plt.plot(epochs, np.asarray(df_fwd_kj), marker=".", label="TFEP FEP fwd")
    if len(df_rev_kj) > 0:
        plt.plot(epochs, np.asarray(df_rev_kj), marker=".", label="TFEP FEP rev")
    for name, val in baseline.items():
        plt.axhline(float(val), linestyle="--", linewidth=1.0, label=name)
    plt.xlabel("epoch")
    plt.ylabel("ΔF [kJ/mol]")
    plt.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(outdir / "fig_df_vs_epoch.png", dpi=200)
    plt.close(fig)


def plot_tfep_work_overlap(outdir: Path, w01_kj: np.ndarray, w10_kj: np.ndarray, df_kj: float) -> None:
    fig = plt.figure(figsize=(7, 3))
    plt.hist(w01_kj, bins=120, alpha=0.6, label="TFEP W(0→1)")
    plt.hist(-w10_kj, bins=120, alpha=0.6, label="TFEP -W(1→0)")
    plt.axvline(df_kj, linestyle="--")
    plt.xlabel("generalized work [kJ/mol]")
    plt.legend()
    plt.tight_layout()
    fig.savefig(outdir / "fig_tfep_work_overlap_final.png", dpi=200)
    plt.close(fig)


# =============================================================================
# Bootstrap
# =============================================================================

def _bootstrap_resample_iid(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.integers(0, n, size=n)


def _bootstrap_resample_blocks(rng: np.random.Generator, n: int, block_len: int) -> np.ndarray:
    L = int(block_len)
    if L <= 1 or L >= n:
        return _bootstrap_resample_iid(rng, n)

    n_blocks = int(np.ceil(n / L))
    start_max = n - L
    starts = rng.integers(0, start_max + 1, size=n_blocks)

    idx = np.empty(n_blocks * L, dtype=int)
    p = 0
    for s in starts:
        idx[p:p + L] = np.arange(s, s + L, dtype=int)
        p += L
    return idx[:n]


def bootstrap_tfep_final_epoch(
    w01_dimless: np.ndarray,
    w10_dimless: np.ndarray,
    *,
    kT_kj: float,
    n_boot: int = 500,
    ci: float = 0.95,
    max_n: int = 200_000,
    block_len: Optional[int] = None,
    seed: int = 123,
) -> Dict[str, object]:
    """Bootstrap uncertainty for TFEP ΔF estimators using final-epoch generalized works."""
    w01 = np.asarray(w01_dimless, dtype=float).reshape(-1)
    w10 = np.asarray(w10_dimless, dtype=float).reshape(-1)

    if w01.size == 0 or w10.size == 0:
        raise ValueError("Empty work arrays.")

    n = int(min(w01.size, w10.size))
    w01 = w01[:n]
    w10 = w10[:n]

    if n > int(max_n):
        rng = np.random.default_rng(int(seed))
        sel = rng.choice(n, size=int(max_n), replace=False)
        sel.sort()
        w01 = w01[sel]
        w10 = w10[sel]
        n = w01.size

    rng = np.random.default_rng(int(seed))
    df_fwd_boot = np.empty(int(n_boot), dtype=float)
    df_rev_boot = np.empty(int(n_boot), dtype=float)
    df_bar_boot = np.empty(int(n_boot), dtype=float)

    for b in range(int(n_boot)):
        if block_len is not None and int(block_len) > 1:
            idx = _bootstrap_resample_blocks(rng, n, int(block_len))
        else:
            idx = _bootstrap_resample_iid(rng, n)
        wf = w01[idx]
        wr = w10[idx]
        df_fwd_boot[b] = fep_forward_df(wf) * float(kT_kj)
        df_rev_boot[b] = fep_reverse_df(wr) * float(kT_kj)
        df_bar_boot[b] = bar_df(wf, wr) * float(kT_kj)

    alpha = (1.0 - float(ci)) / 2.0
    lo_q, hi_q = alpha, 1.0 - alpha

    def summarize(x: np.ndarray) -> Dict[str, float]:
        x = np.asarray(x, dtype=float)
        return dict(
            mean=float(np.mean(x)),
            std=float(np.std(x, ddof=1)) if x.size > 1 else float("nan"),
            q_lo=float(np.quantile(x, lo_q)),
            q_hi=float(np.quantile(x, hi_q)),
        )

    return dict(
        n=int(n),
        n_boot=int(n_boot),
        ci=float(ci),
        block_len=(int(block_len) if block_len is not None else 0),
        df_fwd=summarize(df_fwd_boot),
        df_rev=summarize(df_rev_boot),
        df_bar=summarize(df_bar_boot),
        df_fwd_boot=df_fwd_boot,
        df_rev_boot=df_rev_boot,
        df_bar_boot=df_bar_boot,
    )


def plot_bootstrap_df(outdir: Path, boot: Dict[str, object], point: Dict[str, float]) -> None:
    """Histogram plots + CI annotation for bootstrap results."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))

    for ax, key, title in zip(
        axes,
        ("df_fwd_boot", "df_rev_boot", "df_bar_boot"),
        ("ΔF FWD", "ΔF REV", "ΔF BAR"),
    ):
        x = np.asarray(boot[key], dtype=float)
        ax.hist(x, bins=60, alpha=0.8)
        ax.axvline(point[title.split()[-1].lower()], linestyle="--")
        ax.set_title(title)

    fig.tight_layout()
    fig.savefig(outdir / "fig_bootstrap_df.png", dpi=200)
    plt.close(fig)


# =============================================================================
# K-fold utilities and out-of-sample works
# =============================================================================

def make_kfold_splits(n: int, k: int, *, seed: int = 123, shuffle: bool = False, contiguous: bool = True):
    if k <= 1:
        raise ValueError("k must be >= 2")
    n = int(n)
    idx = np.arange(n, dtype=int)
    if shuffle and not contiguous:
        rng = np.random.default_rng(int(seed))
        rng.shuffle(idx)
    folds = np.array_split(idx, k)
    splits = []
    all_set = set(idx.tolist())
    for f in folds:
        eval_idx = np.asarray(f, dtype=int)
        train_idx = np.asarray(sorted(all_set - set(eval_idx.tolist())), dtype=int)
        splits.append((train_idx, eval_idx))
    return splits


@torch.no_grad()
def compute_tfep_works_on_indices(
    model,
    indices: np.ndarray,
    *,
    U0_x0_dimless_full: np.ndarray,
    U1_x1_dimless_full: np.ndarray,
    kT: float,
    batch_size: int = 256,
    num_workers: int = 0,
    device: Optional[str] = None,
):
    """Compute out-of-sample generalized works for the given trajectory indices.

    Returns (w01_tfep, w10_tfep) in dimensionless kT units.
    """
    indices = np.asarray(indices, dtype=int)
    if indices.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    ds0 = getattr(model, "_dataset_full0", None) or getattr(model, "dataset", None)
    ds1 = getattr(model, "_dataset_full1", None) or getattr(model, "dataset_2", None)
    if ds0 is None or ds1 is None:
        raise RuntimeError("Model datasets are not initialized; did you call Trainer.fit()? ")

    dl0 = torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds0, indices.tolist()),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
    )
    dl1 = torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds1, indices.tolist()),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
    )

    if device is not None:
        model = model.to(device)

    two_pot = model._potential_energy_func
    kT = float(kT)

    w01_list = []
    for batch in dl0:
        result = model(batch)
        dims = batch.get("dimensions", None)
        pot = two_pot.energy(1, result["positions"], dims) / kT
        log_det_J = result["log_det_J"]
        traj_idx = batch["trajectory_sample_index"].detach().cpu().numpy().astype(int)
        w01 = (pot - log_det_J).detach().cpu().numpy() - U0_x0_dimless_full[traj_idx]
        w01_list.append(w01)

    w10_list = []
    for batch in dl1:
        result = model.inverse(batch)
        dims = batch.get("dimensions", None)
        pot = two_pot.energy(0, result["positions"], dims) / kT
        log_det_J = result["log_det_J"]
        traj_idx = batch["trajectory_sample_index"].detach().cpu().numpy().astype(int)
        w10 = (pot - log_det_J).detach().cpu().numpy() - U1_x1_dimless_full[traj_idx]
        w10_list.append(w10)

    w01_tfep = np.concatenate(w01_list) if w01_list else np.asarray([], dtype=float)
    w10_tfep = np.concatenate(w10_list) if w10_list else np.asarray([], dtype=float)
    return w01_tfep, w10_tfep
