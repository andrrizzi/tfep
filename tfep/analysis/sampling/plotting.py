"""Plot helpers for sampling diagnostics."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .core import finite_float


COLORS = {
    "full": "#999999",
    "train": "#1f77b4",
    "validation": "#d95f02",
    "tfep": "#d95f02",
    "raw": "#1f77b4",
}


def savefig(fig, path: Path, dpi: int = 180) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _finite(values) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def plot_feature_histograms(df: pd.DataFrame, feature_columns: Sequence[str], out: Path, *, dpi: int = 180) -> None:
    cols = [c for c in feature_columns if c in df.columns][:6]
    if not cols:
        return
    ncols = 2
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.6 * nrows), squeeze=False)
    for ax, col in zip(axes.ravel(), cols):
        full = _finite(df[col])
        train = _finite(df.loc[df.get("is_train", False), col]) if "is_train" in df.columns else np.asarray([])
        val = _finite(df.loc[df.get("is_validation", False), col]) if "is_validation" in df.columns else np.asarray([])
        if full.size == 0:
            ax.text(0.5, 0.5, "no finite data", ha="center", va="center")
            continue
        bins = np.histogram_bin_edges(full, bins=40)
        ax.hist(full, bins=bins, density=True, alpha=0.25, color=COLORS["full"], label="full")
        if train.size:
            ax.hist(train, bins=bins, density=True, histtype="step", linewidth=1.6, color=COLORS["train"], label="train")
        if val.size:
            ax.hist(val, bins=bins, density=True, histtype="step", linewidth=1.6, color=COLORS["validation"], label="validation")
        ax.set_title(col)
        ax.set_ylabel("density")
    for ax in axes.ravel()[len(cols):]:
        ax.axis("off")
    axes.ravel()[0].legend(frameon=False)
    savefig(fig, out, dpi=dpi)


def plot_feature_timeseries(df: pd.DataFrame, feature_columns: Sequence[str], out: Path, *, dpi: int = 180, max_points: int = 5000) -> None:
    cols = [c for c in feature_columns if c in df.columns][:4]
    if not cols or "frame_index" not in df.columns:
        return
    ordered = df.sort_values("frame_index")
    if len(ordered) > max_points:
        keep = np.linspace(0, len(ordered) - 1, int(max_points), dtype=int)
        ordered = ordered.iloc[keep]
    fig, axes = plt.subplots(len(cols), 1, figsize=(10, 2.7 * len(cols)), squeeze=False, sharex=True)
    x = pd.to_numeric(ordered["frame_index"], errors="coerce").to_numpy(dtype=float)
    for ax, col in zip(axes.ravel(), cols):
        y = pd.to_numeric(ordered[col], errors="coerce").to_numpy(dtype=float)
        ax.plot(x, y, color="#333333", linewidth=0.8, alpha=0.75, label="full")
        if "is_train" in ordered.columns:
            m = ordered["is_train"].to_numpy(dtype=bool)
            ax.scatter(x[m], y[m], s=5, color=COLORS["train"], alpha=0.35, label="train")
        if "is_validation" in ordered.columns:
            m = ordered["is_validation"].to_numpy(dtype=bool)
            ax.scatter(x[m], y[m], s=7, color=COLORS["validation"], alpha=0.55, label="validation")
        ax.set_ylabel(col)
    axes.ravel()[-1].set_xlabel("trajectory frame")
    axes.ravel()[0].legend(frameon=False, ncol=3)
    savefig(fig, out, dpi=dpi)


def plot_cluster_scatter(df: pd.DataFrame, out: Path, *, dpi: int = 180, max_points: int = 8000) -> None:
    if not {"pca1", "pca2", "cluster"}.issubset(df.columns):
        return
    data = df[np.isfinite(pd.to_numeric(df["pca1"], errors="coerce")) & np.isfinite(pd.to_numeric(df["pca2"], errors="coerce"))]
    if data.empty:
        return
    if len(data) > max_points:
        data = data.iloc[np.linspace(0, len(data) - 1, int(max_points), dtype=int)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    sc = axes[0].scatter(data["pca1"], data["pca2"], c=data["cluster"], cmap="tab10", s=8, alpha=0.65)
    axes[0].set_title("CV PCA clusters")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    fig.colorbar(sc, ax=axes[0], label="cluster")
    axes[1].scatter(data["pca1"], data["pca2"], c=COLORS["full"], s=8, alpha=0.20, label="full")
    if "is_train" in data.columns:
        train = data[data["is_train"].astype(bool)]
        axes[1].scatter(train["pca1"], train["pca2"], c=COLORS["train"], s=8, alpha=0.35, label="train")
    if "is_validation" in data.columns:
        val = data[data["is_validation"].astype(bool)]
        axes[1].scatter(val["pca1"], val["pca2"], c=COLORS["validation"], s=12, alpha=0.75, label="validation")
    axes[1].set_title("Train/validation coverage")
    axes[1].set_xlabel("PC1")
    axes[1].legend(frameon=False)
    savefig(fig, out, dpi=dpi)


def plot_work_vs_features(features: pd.DataFrame, work: pd.DataFrame, feature_columns: Sequence[str], out: Path, *, dpi: int = 180) -> None:
    if work.empty or "frame_index" not in features.columns or "frame_index" not in work.columns:
        return
    merged = work.merge(features, on="frame_index", how="inner")
    if merged.empty:
        return
    cols = [c for c in feature_columns if c in merged.columns][:4]
    if not cols:
        return
    fig, axes = plt.subplots(len(cols), 2, figsize=(10.5, 3.1 * len(cols)), squeeze=False)
    for i, col in enumerate(cols):
        x = pd.to_numeric(merged[col], errors="coerce").to_numpy(dtype=float)
        for j, target in enumerate(["raw_work", "tfep_work"]):
            if target not in merged.columns:
                axes[i, j].axis("off")
                continue
            y = pd.to_numeric(merged[target], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            axes[i, j].scatter(x[mask], y[mask], s=10, alpha=0.5, color=COLORS["raw" if target == "raw_work" else "tfep"])
            axes[i, j].set_xlabel(col)
            axes[i, j].set_ylabel(target)
    savefig(fig, out, dpi=dpi)


def plot_shell_residence(df: pd.DataFrame, out: Path, *, dpi: int = 180, max_points: int = 8000) -> None:
    if "nearest_water_resid" not in df.columns or "frame_index" not in df.columns:
        return
    ordered = df.sort_values("frame_index")
    if len(ordered) > max_points:
        ordered = ordered.iloc[np.linspace(0, len(ordered) - 1, int(max_points), dtype=int)]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(ordered["frame_index"], ordered["nearest_water_resid"], linewidth=0.7, color="#333333")
    axes[0].set_ylabel("nearest water resid")
    if "shell_nearest_distance_angstrom" in ordered.columns:
        axes[1].plot(ordered["frame_index"], ordered["shell_nearest_distance_angstrom"], linewidth=0.7, color="#0072B2")
        axes[1].set_ylabel("nearest O distance (A)")
    else:
        axes[1].axis("off")
    axes[-1].set_xlabel("trajectory frame")
    savefig(fig, out, dpi=dpi)


def plot_campaign_risk_heatmap(risk_df: pd.DataFrame, out: Path, *, dpi: int = 180) -> None:
    if risk_df.empty:
        return
    labels = {"low": 0, "moderate": 1, "high": 2, "critical": 3}
    rows = []
    for _, row in risk_df.iterrows():
        rows.append({
            "molecule_state": f"{row.get('compound_id')}/{row.get('leg')}/{row.get('state')}",
            "score": finite_float(row.get("sampling_risk_score"), 0.0),
            "label_value": labels.get(str(row.get("sampling_risk_label", "low")), 0),
        })
    data = pd.DataFrame(rows)
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(data)), 3.2))
    values = data[["score"]].T.to_numpy(dtype=float)
    im = ax.imshow(values, aspect="auto", cmap="YlOrRd")
    ax.set_yticks([0])
    ax.set_yticklabels(["risk score"])
    ax.set_xticks(np.arange(len(data)))
    ax.set_xticklabels(data["molecule_state"], rotation=60, ha="right")
    fig.colorbar(im, ax=ax, label="sampling risk score")
    savefig(fig, out, dpi=dpi)
