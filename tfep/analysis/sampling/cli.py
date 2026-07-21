"""CLI for TFEP/TBar sampling diagnostics."""
from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from tfep.solvation.mapping import minimum_image_displacements, resolve_solute_atom_indices

from .core import (
    SplitMasks,
    add_pca_and_clusters,
    build_split_masks,
    compute_cluster_coverage,
    compute_coverage_metrics,
    compute_sampling_risk,
    compute_timeseries_metrics,
    compute_work_cv_correlations,
    numeric_feature_columns,
    shell_residence_metrics,
    split_consistency_row,
    write_csv_rows,
)
from .plotting import (
    plot_campaign_risk_heatmap,
    plot_cluster_scatter,
    plot_feature_histograms,
    plot_feature_timeseries,
    plot_shell_residence,
    plot_work_vs_features,
)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--config", type=Path, required=True, help="YAML campaign config used by the TBar orchestrator.")
    ap.add_argument("--run-root", type=Path, default=None, help="Campaign run root; defaults to <campaign.root>/tbar_runs/<campaign.name>.")
    ap.add_argument("--output-dir", type=Path, default=None, help="Campaign-level sampling diagnostics directory.")
    ap.add_argument("--analysis-subdir", type=str, default=None, help="Per-run analysis subdirectory; defaults to training.analysis_subdir or holdout_small_train.")
    ap.add_argument("--feature-stride", type=int, default=1, help="Read every Nth trajectory frame for CV extraction.")
    ap.add_argument("--max-frames", type=int, default=None, help="Optional maximum number of frames per endpoint trajectory.")
    ap.add_argument("--max-distance-pairs", type=int, default=64, help="Maximum solute pair-distance features used for internal CVs/PCA.")
    ap.add_argument("--shell-cutoffs", type=str, default="3.5,4.5,6.0", help="Comma-separated shell O-distance cutoffs in Angstrom.")
    ap.add_argument("--shell-top-k", type=str, default="12,24", help="Comma-separated nearest-water counts for shell distance summaries.")
    ap.add_argument("--molecules", nargs="*", default=None, help="Optional molecule IDs to analyze.")
    ap.add_argument("--legs", nargs="*", default=None, help="Optional legs to analyze.")
    ap.add_argument("--seeds", nargs="*", type=int, default=None, help="Optional seeds to analyze.")
    ap.add_argument("--dpi", type=int, default=180, help="Plot resolution.")
    ap.add_argument("--skip-plots", action="store_true", help="Write CSV/JSON only.")
    ap.add_argument("--strict", action="store_true", help="Abort on the first run-level failure instead of recording it.")
    return ap


def load_config(path: Path) -> dict[str, Any]:
    with path.expanduser().resolve().open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config did not load as a mapping: {path}")
    return cfg


def _parse_floats(text: str) -> list[float]:
    return [float(x) for x in str(text).replace(",", " ").split() if x.strip()]


def _parse_ints(text: str) -> list[int]:
    return [int(x) for x in str(text).replace(",", " ").split() if x.strip()]


def _read_manifest(root: Path) -> list[dict[str, str]]:
    path = root / "manifest" / "selected_molecules.csv"
    if not path.is_file():
        return []
    return list(pd.read_csv(path).fillna("").astype(str).to_dict("records"))


def _selected_records(cfg: Mapping[str, Any], molecule_filters: Sequence[str] | None) -> list[dict[str, str]]:
    root = Path(cfg["campaign"]["root"]).expanduser().resolve()
    records = _read_manifest(root)
    if not records:
        selection = cfg.get("selection", {}).get("molecules", [])
        records = [{"compound_id": str(x), "molecule": str(x)} for x in selection]
    selection = cfg.get("selection", {}).get("molecules", "all")
    if selection != "all":
        wanted = {str(x) for x in selection}
        records = [r for r in records if str(r.get("compound_id")) in wanted]
    if molecule_filters:
        wanted = {str(x) for x in molecule_filters}
        records = [r for r in records if str(r.get("compound_id")) in wanted]
    return records


def _jobs(cfg: Mapping[str, Any], *, run_root: Path, molecules=None, legs=None, seeds=None, analysis_subdir: str) -> list[dict[str, Any]]:
    root = Path(cfg["campaign"]["root"]).expanduser().resolve()
    selection = cfg.get("selection", {})
    tr = cfg.get("training", {})
    selected_legs = list(legs or selection.get("legs", ["solv", "vac"]))
    selected_seeds = [int(x) for x in (seeds or tr.get("seeds", [123]))]
    state0_endpoint = str(selection.get("state0_endpoint", "ref_gaff_am1bcc"))
    state1_endpoint = str(selection.get("state1_endpoint", "tgt_openff_nagl"))
    out: list[dict[str, Any]] = []
    for rec in _selected_records(cfg, molecules):
        cid = str(rec.get("compound_id"))
        for leg in selected_legs:
            for seed in selected_seeds:
                run_dir = run_root / cid / str(leg) / f"seed_{int(seed)}"
                out.append(
                    {
                        "compound_id": cid,
                        "molecule": rec.get("molecule", cid),
                        "leg": str(leg),
                        "seed": int(seed),
                        "state0_dir": root / cid / str(leg) / state0_endpoint,
                        "state1_dir": root / cid / str(leg) / state1_endpoint,
                        "analysis_dir": run_dir / analysis_subdir,
                    }
                )
    return out


def _load_indices(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    return np.asarray(np.load(path), dtype=int).reshape(-1)


def _load_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _endpoint_paths(endpoint_dir: Path, traj_rel: str) -> tuple[Path, Path]:
    topology = endpoint_dir / "system" / "start.pdb"
    traj = endpoint_dir / traj_rel
    if not topology.is_file():
        raise FileNotFoundError(f"Topology not found: {topology}")
    if not traj.is_file():
        raise FileNotFoundError(f"Trajectory not found: {traj}")
    return topology, traj


def _atom_is_hydrogen(atom) -> bool:
    element = getattr(atom, "element", "")
    if str(element).upper() == "H":
        return True
    return str(getattr(atom, "name", "")).upper().startswith("H")


def _select_distance_pairs(atom_indices: Sequence[int], max_pairs: int) -> list[tuple[int, int]]:
    pairs = list(combinations([int(i) for i in atom_indices], 2))
    if len(pairs) <= int(max_pairs):
        return pairs
    keep = np.linspace(0, len(pairs) - 1, int(max_pairs), dtype=int)
    return [pairs[int(i)] for i in keep]


def _water_residues_and_oxygen_indices(universe, water_selection: str, oxygen_names: Sequence[str]) -> tuple[list[Any], np.ndarray]:
    water_atoms = universe.select_atoms(water_selection)
    if len(water_atoms) == 0:
        return [], np.asarray([], dtype=int)
    residues = list(water_atoms.residues)
    residues.sort(key=lambda r: int(np.min(r.atoms.indices)))
    oxygen_indices: list[int] = []
    kept_residues: list[Any] = []
    for residue in residues:
        oxygen = None
        for name in oxygen_names:
            ag = residue.atoms.select_atoms(f"name {name}")
            if len(ag) > 0:
                oxygen = int(ag.indices[0])
                break
        if oxygen is None:
            ag = residue.atoms.select_atoms("element O")
            if len(ag) > 0:
                oxygen = int(ag.indices[0])
        if oxygen is not None:
            kept_residues.append(residue)
            oxygen_indices.append(oxygen)
    return kept_residues, np.asarray(oxygen_indices, dtype=int)


def extract_trajectory_features(
    *,
    topology: Path,
    trajectory: Path,
    solute_selection: str,
    water_selection: str,
    oxygen_names: Sequence[str],
    shell_cutoffs: Sequence[float],
    shell_top_k: Sequence[int],
    stride: int,
    max_frames: int | None,
    max_distance_pairs: int,
) -> pd.DataFrame:
    """Extract low-cost solute and solvent-shell CVs from one endpoint trajectory."""
    import MDAnalysis as mda

    universe = mda.Universe(str(topology), str(trajectory))
    solute_indices = resolve_solute_atom_indices(universe, solute_selection=solute_selection)
    solute_set = set(solute_indices)
    heavy_solute = [int(a.index) for a in universe.atoms if int(a.index) in solute_set and not _atom_is_hydrogen(a)]
    if len(heavy_solute) < 2:
        heavy_solute = list(solute_indices)
    pairs = _select_distance_pairs(heavy_solute, max_pairs=max_distance_pairs)
    residues, oxygen_indices = _water_residues_and_oxygen_indices(universe, water_selection, oxygen_names)
    residue_ids = np.asarray([int(r.resid) for r in residues], dtype=int) if residues else np.asarray([], dtype=int)

    rows: list[dict[str, Any]] = []
    pair_vectors: list[np.ndarray] = []
    frame_counter = 0
    ref_pair_vector: np.ndarray | None = None
    stride = max(1, int(stride))
    for ts in universe.trajectory[::stride]:
        if max_frames is not None and frame_counter >= int(max_frames):
            break
        pos = np.asarray(ts.positions, dtype=float)
        solute_pos = pos[np.asarray(solute_indices, dtype=int)]
        heavy_pos = pos[np.asarray(heavy_solute, dtype=int)]
        center = solute_pos.mean(axis=0)
        centered = solute_pos - center
        rg = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1)))) if solute_pos.size else float("nan")

        pair_values: list[float] = []
        for i, j in pairs:
            pair_values.append(float(np.linalg.norm(pos[int(i)] - pos[int(j)])))
        pair_vector = np.asarray(pair_values, dtype=float)
        if ref_pair_vector is None and pair_vector.size:
            ref_pair_vector = pair_vector.copy()
        internal_rmsd = float(np.sqrt(np.mean((pair_vector - ref_pair_vector) ** 2))) if pair_vector.size and ref_pair_vector is not None else float("nan")
        pair_vectors.append(pair_vector)

        row: dict[str, Any] = {
            "frame_index": int(ts.frame),
            "time_ps": float(getattr(ts, "time", np.nan)),
            "solute_rg_angstrom": rg,
            "solute_internal_rmsd_to_first_angstrom": internal_rmsd,
            "solute_pair_distance_mean_angstrom": float(np.mean(pair_vector)) if pair_vector.size else float("nan"),
            "solute_pair_distance_std_angstrom": float(np.std(pair_vector)) if pair_vector.size else float("nan"),
            "solute_pair_distance_min_angstrom": float(np.min(pair_vector)) if pair_vector.size else float("nan"),
            "solute_pair_distance_max_angstrom": float(np.max(pair_vector)) if pair_vector.size else float("nan"),
            "solute_center_x_angstrom": float(center[0]),
            "solute_center_y_angstrom": float(center[1]),
            "solute_center_z_angstrom": float(center[2]),
        }

        if oxygen_indices.size:
            dvec = pos[oxygen_indices] - center
            dvec = minimum_image_displacements(dvec, ts.dimensions)
            dist = np.linalg.norm(dvec, axis=1)
            order = np.argsort(dist)
            sorted_dist = dist[order]
            row["shell_nearest_distance_angstrom"] = float(sorted_dist[0]) if sorted_dist.size else float("nan")
            row["nearest_water_slot"] = int(order[0]) if order.size else -1
            row["nearest_water_resid"] = int(residue_ids[order[0]]) if order.size else -1
            for cutoff in shell_cutoffs:
                label = str(float(cutoff)).replace(".", "p")
                row[f"shell_count_le_{label}_angstrom"] = int(np.sum(dist <= float(cutoff)))
            for k in shell_top_k:
                kk = min(int(k), sorted_dist.size)
                row[f"shell_mean_first_{int(k)}_distance_angstrom"] = float(np.mean(sorted_dist[:kk])) if kk else float("nan")
                row[f"shell_max_first_{int(k)}_distance_angstrom"] = float(np.max(sorted_dist[:kk])) if kk else float("nan")
        rows.append(row)
        frame_counter += 1

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Add PCA over sampled solute pair-distance vectors when enough dimensions exist.
    if pair_vectors and max(len(v) for v in pair_vectors) >= 2:
        width = max(len(v) for v in pair_vectors)
        x = np.full((len(pair_vectors), width), np.nan, dtype=float)
        for i, vec in enumerate(pair_vectors):
            x[i, : len(vec)] = vec
        finite_cols = np.isfinite(x).all(axis=0) & (np.nanstd(x, axis=0) > 1.0e-12)
        if finite_cols.sum() >= 2:
            try:
                from sklearn.decomposition import PCA
                from sklearn.preprocessing import StandardScaler

                xp = StandardScaler().fit_transform(x[:, finite_cols])
                coords = PCA(n_components=2, random_state=0).fit_transform(xp)
                df["solute_dist_pca1"] = coords[:, 0]
                df["solute_dist_pca2"] = coords[:, 1]
            except Exception:
                df["solute_dist_pca1"] = np.nan
                df["solute_dist_pca2"] = np.nan
    return df


def _add_split_columns(features: pd.DataFrame, masks: SplitMasks) -> pd.DataFrame:
    out = features.copy()
    out["is_train"] = masks.train
    out["is_validation"] = masks.validation
    out["split_label"] = np.where(masks.train, "train", np.where(masks.validation, "validation", "unassigned"))
    return out


def _work_table(arrays_path: Path, state_index: int) -> pd.DataFrame:
    if not arrays_path.is_file():
        return pd.DataFrame()
    with np.load(arrays_path) as data:
        if state_index == 0:
            idx_key, raw_key, tfep_key = "state0_state1_traj_idx", "raw_w01", "tfep_w01"
        else:
            idx_key, raw_key, tfep_key = "state1_state0_traj_idx", "raw_w10", "tfep_w10"
        if idx_key not in data.files or raw_key not in data.files or tfep_key not in data.files:
            return pd.DataFrame()
        df = pd.DataFrame(
            {
                "frame_index": np.asarray(data[idx_key], dtype=int),
                "raw_work": np.asarray(data[raw_key], dtype=float),
                "tfep_work": np.asarray(data[tfep_key], dtype=float),
            }
        )
    df["work_shift"] = df["tfep_work"] - df["raw_work"]
    df["abs_work_shift"] = np.abs(df["work_shift"])
    df["abs_raw_work"] = np.abs(df["raw_work"])
    df["abs_tfep_work"] = np.abs(df["tfep_work"])
    return df


def _outlier_rows(features: pd.DataFrame, work: pd.DataFrame, *, metadata: Mapping[str, Any]) -> list[dict[str, Any]]:
    if features.empty or work.empty:
        return []
    merged = work.merge(features, on="frame_index", how="inner")
    if merged.empty or "abs_work_shift" not in merged.columns:
        return []
    threshold = float(np.nanquantile(merged["abs_work_shift"], 0.95)) if len(merged) >= 20 else float(np.nanmax(merged["abs_work_shift"]))
    selected = merged[merged["abs_work_shift"] >= threshold].copy()
    selected = selected.sort_values("abs_work_shift", ascending=False).head(50)
    keep = [
        "frame_index",
        "raw_work",
        "tfep_work",
        "work_shift",
        "abs_work_shift",
        "cluster",
        "solute_rg_angstrom",
        "solute_internal_rmsd_to_first_angstrom",
        "shell_nearest_distance_angstrom",
        "nearest_water_resid",
    ]
    rows: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        out = dict(metadata)
        for key in keep:
            if key in row.index:
                value = row[key]
                if isinstance(value, (np.integer, np.floating)):
                    value = value.item()
                out[key] = value
        out["outlier_threshold_abs_work_shift"] = threshold
        rows.append(out)
    return rows


def _write_report(path: Path, *, job: Mapping[str, Any], risk_rows: Sequence[Mapping[str, Any]], warning: str | None = None) -> None:
    lines = [
        "# Sampling Diagnostics Report",
        "",
        f"- Molecule: `{job.get('compound_id')}`",
        f"- Leg: `{job.get('leg')}`",
        f"- Seed: `{job.get('seed')}`",
        "",
        "These diagnostics are analysis-only. They do not change TFEP/TBar work arrays, train/validation splits, or estimator outputs.",
    ]
    if warning:
        lines.extend(["", f"Warning: {warning}"])
    if risk_rows:
        lines.extend(["", "## Sampling Risk", ""])
        for row in risk_rows:
            lines.append(
                f"- `{row.get('state')}`: **{row.get('sampling_risk_label')}** "
                f"(score {float(row.get('sampling_risk_score', 0.0)):.2f}) — {row.get('risk_reasons')}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_one_job(
    job: Mapping[str, Any],
    *,
    cfg: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, list[dict[str, Any]]]:
    tr = cfg.get("training", {})
    traj_rel = str(tr.get("traj", "03_md/traj.xtc"))
    solute_selection = str(tr.get("mapped_atoms", "resname UNL"))
    water_selection = str(tr.get("conditioning_shell_water_selection", "water"))
    oxygen_names = [x.strip() for x in str(tr.get("conditioning_shell_oxygen_names", "O,OW")).replace(",", " ").split() if x.strip()]
    expected_train_count = int(tr.get("train_count", 0)) if tr.get("train_count") is not None else None
    shell_cutoffs = _parse_floats(args.shell_cutoffs)
    shell_top_k = _parse_ints(args.shell_top_k)

    analysis_dir = Path(job["analysis_dir"])
    diag_dir = analysis_dir / "sampling_diagnostics"
    plots_dir = diag_dir / "plots"
    diag_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)

    summary = _load_summary(analysis_dir / "summary.json")
    split = summary.get("split", {}) if summary else {}
    n_samples = {
        0: int(split.get("n_state0", 0) or 0),
        1: int(split.get("n_state1", 0) or 0),
    }
    train_indices = {
        0: _load_indices(analysis_dir / "state0_train_indices.npy"),
        1: _load_indices(analysis_dir / "state1_train_indices.npy"),
    }
    val_indices = {
        0: _load_indices(analysis_dir / "state0_val_indices.npy"),
        1: _load_indices(analysis_dir / "state1_val_indices.npy"),
    }

    endpoint_dirs = {0: Path(job["state0_dir"]), 1: Path(job["state1_dir"])}
    all_split: list[dict[str, Any]] = []
    all_coverage: list[dict[str, Any]] = []
    all_cluster: list[dict[str, Any]] = []
    all_timeseries: list[dict[str, Any]] = []
    all_corr: list[dict[str, Any]] = []
    all_outliers: list[dict[str, Any]] = []
    all_risk: list[dict[str, Any]] = []

    for state_index in (0, 1):
        state_name = f"state{state_index}"
        metadata = {
            "compound_id": job["compound_id"],
            "molecule": job.get("molecule", job["compound_id"]),
            "leg": job["leg"],
            "seed": int(job["seed"]),
            "state": state_name,
        }
        split_row = split_consistency_row(
            **metadata,
            n_samples=n_samples[state_index],
            train_indices=train_indices[state_index],
            validation_indices=val_indices[state_index],
            expected_train_count=expected_train_count,
        )
        all_split.append(split_row)

        topology, trajectory = _endpoint_paths(endpoint_dirs[state_index], traj_rel)
        features = extract_trajectory_features(
            topology=topology,
            trajectory=trajectory,
            solute_selection=solute_selection,
            water_selection=water_selection,
            oxygen_names=oxygen_names,
            shell_cutoffs=shell_cutoffs,
            shell_top_k=shell_top_k,
            stride=args.feature_stride,
            max_frames=args.max_frames,
            max_distance_pairs=args.max_distance_pairs,
        )
        if features.empty:
            raise RuntimeError(f"No features extracted for {job['compound_id']} {job['leg']} {state_name}")

        masks = build_split_masks(
            features["frame_index"].to_numpy(dtype=int),
            [] if train_indices[state_index] is None else train_indices[state_index],
            [] if val_indices[state_index] is None else val_indices[state_index],
        )
        features = _add_split_columns(features, masks)
        features, _ = add_pca_and_clusters(features)
        feature_cols = [
            c for c in numeric_feature_columns(features)
            if c not in {"solute_center_x_angstrom", "solute_center_y_angstrom", "solute_center_z_angstrom"}
        ]
        features.to_csv(diag_dir / f"features_{state_name}.csv", index=False)

        coverage = compute_coverage_metrics(features, masks, feature_columns=feature_cols, metadata=metadata)
        cluster = compute_cluster_coverage(features, masks, metadata=metadata)
        timeseries = compute_timeseries_metrics(features, feature_columns=feature_cols, metadata=metadata)
        shell_rows = shell_residence_metrics(features, metadata=metadata)
        work = _work_table(analysis_dir / "validation_work_arrays.npz", state_index)
        corr = compute_work_cv_correlations(features, work, metadata=metadata, feature_columns=feature_cols)
        outliers = _outlier_rows(features, work, metadata=metadata)
        risk = compute_sampling_risk(
            split_row=split_row,
            coverage_rows=coverage,
            cluster_rows=cluster,
            timeseries_rows=timeseries,
            work_corr_rows=corr,
            shell_rows=shell_rows,
        )
        risk_row = {**metadata, **risk}

        all_coverage.extend(coverage)
        all_cluster.extend(cluster)
        all_timeseries.extend(timeseries)
        all_timeseries.extend(shell_rows)
        all_corr.extend(corr)
        all_outliers.extend(outliers)
        all_risk.append(risk_row)

        if not args.skip_plots:
            plot_feature_histograms(features, feature_cols, plots_dir / f"{state_name}_cv_histograms.png", dpi=args.dpi)
            plot_feature_timeseries(features, feature_cols, plots_dir / f"{state_name}_cv_timeseries.png", dpi=args.dpi)
            plot_cluster_scatter(features, plots_dir / f"{state_name}_cluster_coverage.png", dpi=args.dpi)
            plot_shell_residence(features, plots_dir / f"{state_name}_shell_residence.png", dpi=args.dpi)
            plot_work_vs_features(features, work, feature_cols, plots_dir / f"{state_name}_work_vs_cv.png", dpi=args.dpi)

    write_csv_rows(all_split, diag_dir / "split_coverage.csv")
    write_csv_rows(all_coverage, diag_dir / "cv_coverage_summary.csv")
    write_csv_rows(all_cluster, diag_dir / "cluster_coverage.csv")
    write_csv_rows(all_timeseries, diag_dir / "timeseries_metrics.csv")
    write_csv_rows(all_corr, diag_dir / "work_cv_correlations.csv")
    write_csv_rows(all_outliers, diag_dir / "outlier_frames.csv")
    write_csv_rows(all_risk, diag_dir / "sampling_risk_table.csv")
    _write_report(diag_dir / "sampling_report.md", job=job, risk_rows=all_risk)

    return {
        "split": all_split,
        "coverage": all_coverage,
        "cluster": all_cluster,
        "timeseries": all_timeseries,
        "correlations": all_corr,
        "outliers": all_outliers,
        "risk": all_risk,
    }


def write_campaign_report(path: Path, risk_rows: Sequence[Mapping[str, Any]], failures: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Campaign Sampling Diagnostics",
        "",
        "These outputs diagnose whether saved TFEP/TBar train/validation splits cover slow solute and solvent-shell coordinates.",
        "They are diagnostic only and do not correct free energies or modify estimator outputs.",
        "",
        f"- Completed diagnostic rows: `{len(risk_rows)}`",
        f"- Failed jobs: `{len(failures)}`",
    ]
    if risk_rows:
        labels = pd.DataFrame(risk_rows)["sampling_risk_label"].value_counts().to_dict()
        lines.extend(["", "## Risk Label Counts", ""])
        for label, count in sorted(labels.items()):
            lines.append(f"- `{label}`: `{count}`")
        high = [r for r in risk_rows if str(r.get("sampling_risk_label")) in {"high", "critical"}]
        if high:
            lines.extend(["", "## High-Risk States", ""])
            for row in high[:20]:
                lines.append(
                    f"- `{row.get('compound_id')}/{row.get('leg')}/{row.get('state')}`: "
                    f"{row.get('sampling_risk_label')} — {row.get('risk_reasons')}"
                )
    if failures:
        lines.extend(["", "## Failures", ""])
        for row in failures:
            lines.append(f"- `{row.get('compound_id')}/{row.get('leg')}/seed_{row.get('seed')}`: {row.get('error')}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    cfg = load_config(args.config)
    root = Path(cfg["campaign"]["root"]).expanduser().resolve()
    campaign_name = str(cfg["campaign"]["name"])
    run_root = args.run_root.expanduser().resolve() if args.run_root else root / "tbar_runs" / campaign_name
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else run_root / "analysis" / "sampling"
    analysis_subdir = args.analysis_subdir or str(cfg.get("training", {}).get("analysis_subdir", "holdout_small_train"))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)

    jobs = _jobs(
        cfg,
        run_root=run_root,
        molecules=args.molecules,
        legs=args.legs,
        seeds=args.seeds,
        analysis_subdir=analysis_subdir,
    )
    all_split: list[dict[str, Any]] = []
    all_coverage: list[dict[str, Any]] = []
    all_cluster: list[dict[str, Any]] = []
    all_timeseries: list[dict[str, Any]] = []
    all_corr: list[dict[str, Any]] = []
    all_outliers: list[dict[str, Any]] = []
    all_risk: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    print(f"[sampling] jobs={len(jobs)} run_root={run_root}")
    for i, job in enumerate(jobs, start=1):
        label = f"{job['compound_id']}/{job['leg']}/seed_{job['seed']}"
        print(f"[sampling] ({i}/{len(jobs)}) {label}", flush=True)
        try:
            result = analyze_one_job(job, cfg=cfg, args=args)
            all_split.extend(result["split"])
            all_coverage.extend(result["coverage"])
            all_cluster.extend(result["cluster"])
            all_timeseries.extend(result["timeseries"])
            all_corr.extend(result["correlations"])
            all_outliers.extend(result["outliers"])
            all_risk.extend(result["risk"])
        except Exception as exc:
            if args.strict:
                raise
            failures.append({**{k: job[k] for k in ("compound_id", "leg", "seed")}, "error": repr(exc)})
            print(f"[sampling] WARNING {label}: {exc!r}", flush=True)

    write_csv_rows(all_risk, output_dir / "sampling_risk_table.csv")
    write_csv_rows(all_split, output_dir / "split_consistency.csv")
    write_csv_rows(all_coverage, output_dir / "cv_coverage_summary.csv")
    write_csv_rows(all_cluster, output_dir / "cluster_coverage_summary.csv")
    write_csv_rows(all_corr, output_dir / "work_cv_correlations.csv")
    write_csv_rows(all_outliers, output_dir / "sampling_outliers.csv")
    write_csv_rows(failures, output_dir / "failures.csv")

    metadata = {
        "config": str(args.config.expanduser().resolve()),
        "run_root": str(run_root),
        "output_dir": str(output_dir),
        "feature_stride": int(args.feature_stride),
        "max_frames": args.max_frames,
        "analysis_only": True,
        "statistical_meaning": "Sampling diagnostics only; no TFEP/TBar/SNF work arrays or split indices are modified.",
        "n_jobs": len(jobs),
        "n_failures": len(failures),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_campaign_report(output_dir / "sampling_report.md", all_risk, failures)
    if not args.skip_plots:
        plot_campaign_risk_heatmap(pd.DataFrame(all_risk), output_dir / "plots" / "sampling_risk_heatmap.png", dpi=args.dpi)
    print(f"[sampling] wrote {output_dir}")


if __name__ == "__main__":
    main()
