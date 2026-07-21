#!/usr/bin/env python3
"""OpenMM-only one-sided small-molecule TFEP training.

This module is intentionally separate from ``small_molecule_tmbar``.  It uses
``TFEPMapBase``-derived maps and trains only the source-to-target direction with
the standard Boltzmann KL objective.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import MDAnalysis as mda
import lightning as L
import numpy as np
import pint
import torch

import tfep
from tfep.analysis import reweighting as rwlib
from tfep.app.cartesianmaf import CartesianMAFMap
from tfep.app.mixedmaf import MixedMAFMap
from tfep.app import small_molecule_tmbar as tmbar_mod
from tfep.io.dataset import TrajectorySubset
from tfep.potentials.openmm import OpenMMPotential


def _as_indices_or_selection(x: Optional[str]) -> Optional[Union[str, list[int]]]:
    return tmbar_mod._as_indices_or_selection(x)


def _load_index_array(path: Optional[Union[str, Path]]) -> Optional[np.ndarray]:
    return tmbar_mod._load_index_array(path)


def _subset_dataset(dataset, subset_indices: Optional[Sequence[int]], label: str):
    if subset_indices is None:
        return dataset
    return TrajectorySubset(dataset, tmbar_mod._normalize_subset_indices(subset_indices, len(dataset), label))


def _stable_fep_deltaf(work: np.ndarray) -> float:
    work = np.asarray(work, dtype=np.float64)
    work = work[np.isfinite(work)]
    if work.size == 0:
        return float("nan")
    x = -work
    m = float(np.max(x))
    return float(-(m + np.log(np.mean(np.exp(x - m)))))


def _fep_sigma_delta_method(work: np.ndarray) -> float:
    work = np.asarray(work, dtype=np.float64)
    work = work[np.isfinite(work)]
    if work.size < 2:
        return float("nan")
    x = -work
    m = float(np.max(x))
    weights = np.exp(x - m)
    mean = float(np.mean(weights))
    if mean <= 0.0 or not math.isfinite(mean):
        return float("nan")
    return float(np.std(weights, ddof=1) / math.sqrt(work.size) / mean)


def _ess_ratio(work: np.ndarray) -> float:
    work = np.asarray(work, dtype=np.float64)
    work = work[np.isfinite(work)]
    if work.size == 0:
        return float("nan")
    x = -work
    x = x - np.max(x)
    weights = np.exp(x)
    denom = float(np.sum(weights * weights))
    if denom <= 0.0:
        return float("nan")
    ess = float(np.sum(weights) ** 2 / denom)
    return ess / float(work.size)


def _work_summary(work: np.ndarray) -> Dict[str, Any]:
    work = np.asarray(work, dtype=np.float64)
    finite = work[np.isfinite(work)]
    return {
        "deltaf": _stable_fep_deltaf(finite),
        "sigma": _fep_sigma_delta_method(finite),
        "ess": _ess_ratio(finite),
        "overlap": _ess_ratio(finite),
        "direct_overlap": float("nan"),
        "n": int(finite.size),
        "n_total": int(work.size),
        "n_nonfinite": int(work.size - finite.size),
        "work_mean": float(np.mean(finite)) if finite.size else float("nan"),
        "work_std": float(np.std(finite, ddof=1)) if finite.size > 1 else float("nan"),
        "work_min": float(np.min(finite)) if finite.size else float("nan"),
        "work_max": float(np.max(finite)) if finite.size else float("nan"),
        "w_forward_std": float(np.std(finite, ddof=1)) if finite.size > 1 else float("nan"),
        "w_reverse_std": float("nan"),
    }


def _weighted_work_summary(work: np.ndarray, log_weights: Optional[np.ndarray]) -> Dict[str, Any]:
    work = np.asarray(work, dtype=np.float64)
    return {
        "deltaf": rwlib.weighted_fep_deltaf(work, log_weights),
        "sigma": float("nan"),
        "ess": rwlib.log_weight_diagnostics(log_weights).get("ess", float("nan")),
        "overlap": rwlib.log_weight_diagnostics(log_weights).get("ess_ratio", float("nan")),
        "n": int(np.isfinite(work).sum()),
        "estimator": "weighted_fep",
        "weight_diagnostics": rwlib.log_weight_diagnostics(log_weights),
        "statistical_warning": (
            "Weighted FEP assumes the supplied log weights correctly unbias the "
            "enhanced-sampling source trajectory."
        ),
    }


class _OneSidedMixin(tmbar_mod.NoFixedBoxResidueWrapMixin):
    """Shared dataset, optimizer, and source-potential helpers."""

    def _init_one_sided(
        self,
        *,
        lr: float,
        weight_decay: float,
        source_potential: torch.nn.Module,
        train_indices: Optional[Sequence[int]],
        source_log_weights: Optional[Sequence[float]],
        wrap_box_eval: bool,
        wrap_residue_blocks: Optional[Sequence[Sequence[int]]],
    ) -> None:
        self._lr = float(lr)
        self._weight_decay = float(weight_decay)
        self._source_potential = source_potential
        self._train_indices = None if train_indices is None else np.asarray(train_indices, dtype=int)
        self._source_log_weights = None if source_log_weights is None else np.asarray(source_log_weights, dtype=float)
        self._init_box_eval_wrap(wrap_box_eval=wrap_box_eval, wrap_residue_blocks=wrap_residue_blocks)

    def make_dataset(self, subset_indices: Optional[Sequence[int]] = None):
        dataset = self._make_universe_dataset()
        if self._source_log_weights is not None:
            dataset.set_log_weights(self._source_log_weights)
        return _subset_dataset(dataset, subset_indices, label="state0")

    def create_dataset(self):
        return self.make_dataset(self._train_indices)

    def _make_universe_dataset(self):
        return tfep.io.TrajectoryDataset(universe=self.create_universe())

    def create_universe(self):
        if isinstance(self._coordinates_file_path, str):
            coords = [self._coordinates_file_path]
        else:
            coords = list(self._coordinates_file_path)
        return mda.Universe(self._topology_file_path, *coords)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self._lr, weight_decay=self._weight_decay)

    def eval_source_potential(self, positions: torch.Tensor, dimensions: Optional[torch.Tensor]):
        return _eval_potential(self, self._source_potential, positions, dimensions)

    def eval_target_potential(self, positions: torch.Tensor, dimensions: Optional[torch.Tensor]):
        return _eval_potential(self, self._potential_energy_func, positions, dimensions)


class SmallMolOneSidedTFEPMapCartesian(_OneSidedMixin, CartesianMAFMap):
    def __init__(
        self,
        source_potential: torch.nn.Module,
        target_potential: torch.nn.Module,
        *,
        topology_file_path: str,
        coordinates_file_path: str,
        temperature: pint.Quantity,
        batch_size: int,
        mapped_atoms=None,
        conditioning_atoms=None,
        origin_atom=None,
        axes_atoms=None,
        tfep_logger_dir_path: str,
        n_maf_layers: int = 6,
        maf_hidden_layers: int = 2,
        maf_weight_norm: bool = True,
        maf_initialize_identity: bool = True,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        train_indices: Optional[Sequence[int]] = None,
        source_log_weights: Optional[Sequence[float]] = None,
        wrap_box_eval: bool = False,
        wrap_residue_blocks: Optional[Sequence[Sequence[int]]] = None,
    ):
        super().__init__(
            potential_energy_func=target_potential,
            topology_file_path=topology_file_path,
            coordinates_file_path=coordinates_file_path,
            temperature=temperature,
            batch_size=batch_size,
            mapped_atoms=mapped_atoms,
            conditioning_atoms=conditioning_atoms,
            origin_atom=origin_atom,
            axes_atoms=axes_atoms,
            tfep_logger_dir_path=tfep_logger_dir_path,
            n_maf_layers=n_maf_layers,
            hidden_layers=int(maf_hidden_layers),
            weight_norm=bool(maf_weight_norm),
            initialize_identity=bool(maf_initialize_identity),
        )
        self._init_one_sided(
            lr=lr,
            weight_decay=weight_decay,
            source_potential=source_potential,
            train_indices=train_indices,
            source_log_weights=source_log_weights,
            wrap_box_eval=wrap_box_eval,
            wrap_residue_blocks=wrap_residue_blocks,
        )


class SmallMolOneSidedTFEPMapMixed(_OneSidedMixin, MixedMAFMap):
    def __init__(
        self,
        source_potential: torch.nn.Module,
        target_potential: torch.nn.Module,
        *,
        topology_file_path: str,
        coordinates_file_path: str,
        temperature: pint.Quantity,
        batch_size: int,
        mapped_atoms=None,
        conditioning_atoms=None,
        origin_atom=None,
        axes_atoms=None,
        tfep_logger_dir_path: str,
        n_maf_layers: int = 6,
        remove_translation: bool = False,
        remove_rotation: bool = False,
        distance_lower_limit_displacement_angstrom: float = 0.3,
        maf_hidden_layers: int = 2,
        maf_weight_norm: bool = True,
        maf_initialize_identity: bool = True,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        train_indices: Optional[Sequence[int]] = None,
        source_log_weights: Optional[Sequence[float]] = None,
        wrap_box_eval: bool = False,
        wrap_residue_blocks: Optional[Sequence[Sequence[int]]] = None,
    ):
        ureg = target_potential.positions_unit._REGISTRY
        super().__init__(
            potential_energy_func=target_potential,
            topology_file_path=topology_file_path,
            coordinates_file_path=coordinates_file_path,
            temperature=temperature,
            batch_size=batch_size,
            mapped_atoms=mapped_atoms,
            conditioning_atoms=conditioning_atoms,
            origin_atom=origin_atom,
            axes_atoms=axes_atoms,
            tfep_logger_dir_path=tfep_logger_dir_path,
            n_maf_layers=n_maf_layers,
            remove_translation=bool(remove_translation),
            remove_rotation=bool(remove_rotation),
            distance_lower_limit_displacement=float(distance_lower_limit_displacement_angstrom) * ureg.angstrom,
            hidden_layers=int(maf_hidden_layers),
            weight_norm=bool(maf_weight_norm),
            initialize_identity=bool(maf_initialize_identity),
        )
        self._init_one_sided(
            lr=lr,
            weight_decay=weight_decay,
            source_potential=source_potential,
            train_indices=train_indices,
            source_log_weights=source_log_weights,
            wrap_box_eval=wrap_box_eval,
            wrap_residue_blocks=wrap_residue_blocks,
        )

    def create_universe(self):
        universe = super().create_universe()
        tmbar_mod._ensure_bonds_for_mixed_coordinates(universe)
        return universe


def _eval_potential(model, potential: torch.nn.Module, positions: torch.Tensor, dimensions: Optional[torch.Tensor]):
    positions = model._wrap_positions_for_box_eval(positions, dimensions)
    if isinstance(potential, tmbar_mod.IsolatedOpenMMPotential) and not bool(positions.requires_grad):
        return potential.energy_no_grad(positions, dimensions)
    if dimensions is None:
        return potential(positions)
    try:
        return potential(positions, dimensions)
    except TypeError:
        return potential(positions)


@dataclass
class OneSidedTrainingArtifacts:
    args: argparse.Namespace
    outdir: Path
    trainer: L.Trainer
    model: torch.nn.Module
    state0_dir: Path
    state1_dir: Path
    traj0_path: Path
    traj1_path: Path
    topo_path: Path
    system0_xml_path: Path
    system1_xml_path: Path


def build_argparser() -> argparse.ArgumentParser:
    parser = tmbar_mod.build_argparser()
    parser.description = "OpenMM-only one-sided TFEP training and evaluation."
    return parser


def create_model_from_args(
    args: argparse.Namespace,
    *,
    outdir_override: Optional[Union[str, Path]] = None,
    train_indices_0: Optional[Sequence[int]] = None,
    train_indices_1: Optional[Sequence[int]] = None,
):
    if str(getattr(args, "objective", "kl")).lower() != "kl":
        raise ValueError("One-sided standard TFEP requires --objective kl.")
    if float(getattr(args, "bar_lambda", 0.0) or 0.0) != 0.0 or float(getattr(args, "lambda_bar", 0.0) or 0.0) != 0.0:
        raise ValueError("One-sided standard TFEP does not use BAR regularization; set --bar-lambda 0 and --lambda-bar 0.")
    if float(getattr(args, "logJ_penalty", 0.0) or 0.0) != 0.0:
        raise ValueError("One-sided standard TFEP baseline requested here uses --logJ-penalty 0.")
    if str(args.flow_space) == "shell-equivariant":
        raise ValueError("One-sided standard TFEP does not support shell-equivariant water mapping.")
    if any(int(getattr(args, name, 0) or 0) for name in ("conditioning_shell_k1", "conditioning_shell_k2")):
        raise ValueError("One-sided standard TFEP disables shell conditioning; set conditioning_shell_k1/k2 to 0.")
    if getattr(args, "state1_reweight_file", None):
        raise ValueError("One-sided TFEP reweights only the source/state0 trajectory; do not pass --state1-reweight-file.")
    source_log_weights, source_reweight_metadata = tmbar_mod._load_state_log_weights_from_args(args, 0)
    if source_reweight_metadata is not None:
        print(
            "[reweighting] one-sided source: "
            f"kind={source_reweight_metadata['kind']} column={source_reweight_metadata['column']} "
            f"ESS/N={source_reweight_metadata.get('ess_ratio', float('nan')):.4g}"
        )

    outdir = Path(outdir_override) if outdir_override is not None else Path(args.outdir)
    state0_dir = Path(args.state0_dir).expanduser().resolve()
    state1_dir = Path(args.state1_dir).expanduser().resolve()
    traj0_path, system0_xml_path, topo_path = tmbar_mod._resolve_state_inputs(
        state0_dir,
        args.traj0,
        args.system0_xml,
        args.topology,
    )
    traj1_path, system1_xml_path, _ = tmbar_mod._resolve_state_inputs(
        state1_dir,
        args.traj1,
        args.system1_xml,
        args.topology,
    )

    system0 = tmbar_mod._load_openmm_system(system0_xml_path)
    system1 = tmbar_mod._load_openmm_system(system1_xml_path)
    platform = tmbar_mod._configure_openmm_platform(
        args.openmm_platform,
        args.openmm_device,
        args.openmm_precision,
        int(args.openmm_cpu_threads),
    )
    wrap_blocks = tmbar_mod.compute_residue_wrap_blocks(topo_path)

    ureg = pint.UnitRegistry()
    positions_unit = ureg.angstrom
    energy_unit = ureg("kJ/mol")
    temperature = float(args.temperature) * ureg.kelvin

    potential_cls = OpenMMPotential
    if bool(args.no_fixed_box):
        potential_cls = tmbar_mod.IsolatedOpenMMPotential
    else:
        tmbar_mod.set_system_default_box_from_cryst1(system0, topo_path)
        tmbar_mod.set_system_default_box_from_cryst1(system1, topo_path)
        aA, bA, cA, alpha, beta, gamma = tmbar_mod._read_cryst1_dims(topo_path)
        print(f"[box] System default box from CRYST1 (A,deg): [{aA}, {bA}, {cA}, {alpha}, {beta}, {gamma}]")

    source_potential = potential_cls(
        system=system0,
        platform=platform,
        positions_unit=positions_unit,
        energy_unit=energy_unit,
        system_name="onesided_source",
        precompute_gradient=False,
    )
    target_potential = potential_cls(
        system=system1,
        platform=platform,
        positions_unit=positions_unit,
        energy_unit=energy_unit,
        system_name="onesided_target",
        precompute_gradient=True,
    )
    if not bool(args.no_fixed_box):
        source_potential = tmbar_mod.NoCellWrapper(source_potential)
        target_potential = tmbar_mod.NoCellWrapper(target_potential)
        print("[box] ignoring per-frame dimensions; using System default box")
    else:
        print("[box] no-fixed-box enabled; source/target potentials evaluated in isolated OpenMM workers")

    mapped_atoms = _as_indices_or_selection(args.mapped_atoms)
    conditioning_atoms = _as_indices_or_selection(args.conditioning_atoms)
    origin_atom = _as_indices_or_selection(args.origin_atom)
    axes_atoms = None
    if args.axes_atoms is not None:
        axes_atoms = [_as_indices_or_selection(args.axes_atoms[0]), _as_indices_or_selection(args.axes_atoms[1])]

    common = dict(
        source_potential=source_potential,
        target_potential=target_potential,
        topology_file_path=str(topo_path),
        coordinates_file_path=str(traj0_path),
        temperature=temperature,
        batch_size=int(args.batch_size),
        mapped_atoms=mapped_atoms,
        conditioning_atoms=conditioning_atoms,
        origin_atom=origin_atom,
        axes_atoms=axes_atoms,
        tfep_logger_dir_path=str(outdir / "tfep_logs"),
        n_maf_layers=int(args.maf_layers),
        maf_hidden_layers=int(args.maf_hidden_layers),
        maf_weight_norm=bool(args.maf_weight_norm and not args.maf_no_weight_norm),
        maf_initialize_identity=bool(args.maf_initialize_identity and not args.maf_no_initialize_identity),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        train_indices=train_indices_0,
        source_log_weights=source_log_weights,
        wrap_box_eval=bool(args.no_fixed_box),
        wrap_residue_blocks=wrap_blocks,
    )

    if args.flow_space == "cartesian":
        model = SmallMolOneSidedTFEPMapCartesian(**common)
    elif args.flow_space == "mixedmaf":
        model = SmallMolOneSidedTFEPMapMixed(
            **common,
            remove_translation=bool(args.remove_translation),
            remove_rotation=bool(args.remove_rotation),
            distance_lower_limit_displacement_angstrom=float(args.distance_lower_limit_displacement_angstrom),
        )
    else:
        raise ValueError(f"Unsupported one-sided flow_space: {args.flow_space}")

    print("[objective] one-sided standard TFEP KL objective (ref->tgt only)")
    return model, {
        "outdir": outdir,
        "state0_dir": state0_dir,
        "state1_dir": state1_dir,
        "traj0_path": traj0_path,
        "traj1_path": traj1_path,
        "topo_path": topo_path,
        "system0_xml_path": system0_xml_path,
        "system1_xml_path": system1_xml_path,
        "objective_effective": "kl",
        "objective_requested": "kl",
        "direction": "ref_to_tgt",
        "reweighting": {
            "enabled": source_reweight_metadata is not None,
            "source": source_reweight_metadata,
            "estimator_warning": (
                "Unweighted one-sided estimates from biased enhanced-sampling trajectories "
                "are diagnostics only; use reweighted estimates for equilibrium free energies."
            ) if source_reweight_metadata is not None else None,
        },
    }


def build_trainer(args: argparse.Namespace, outdir: Path) -> L.Trainer:
    return tmbar_mod.build_trainer(args, outdir)


def run_training(
    args: argparse.Namespace,
    *,
    outdir_override: Optional[Union[str, Path]] = None,
    train_indices_0: Optional[Sequence[int]] = None,
    train_indices_1: Optional[Sequence[int]] = None,
) -> OneSidedTrainingArtifacts:
    outdir = Path(outdir_override) if outdir_override is not None else Path(args.outdir)
    if args.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    L.seed_everything(int(args.seed), workers=True)
    model, ctx = create_model_from_args(
        args,
        outdir_override=outdir,
        train_indices_0=train_indices_0,
        train_indices_1=train_indices_1,
    )

    run_config = tmbar_mod._namespace_to_jsonable(args)
    run_config.update(
        {
            "effective_outdir": str(outdir.resolve()),
            "estimator": "onesided_tfep",
            "direction": "ref_to_tgt",
            "objective_requested": "kl",
            "objective_effective": "kl",
            "legacy_bar_regularizer_active": False,
            "legacy_bar_lambda": 0.0,
            "lambda_bar": 0.0,
            "logJ_penalty_weight": 0.0,
            "objective_mode_description": "One-sided Boltzmann KL TFEP; no BAR objective.",
            "reweighting": ctx.get("reweighting", {"enabled": False}),
        }
    )
    if train_indices_0 is not None:
        run_config["train_indices_0_count"] = int(len(train_indices_0))
    (outdir / "run_config.json").write_text(json.dumps(tmbar_mod._json_ready(run_config), indent=2), encoding="utf-8")

    trainer = build_trainer(args, outdir)
    trainer.fit(model)

    return OneSidedTrainingArtifacts(
        args=args,
        outdir=outdir.resolve(),
        trainer=trainer,
        model=model,
        state0_dir=ctx["state0_dir"],
        state1_dir=ctx["state1_dir"],
        traj0_path=ctx["traj0_path"],
        traj1_path=ctx["traj1_path"],
        topo_path=ctx["topo_path"],
        system0_xml_path=ctx["system0_xml_path"],
        system1_xml_path=ctx["system1_xml_path"],
    )


def evaluate_one_sided_map(
    model,
    *,
    subset_indices_0: Optional[Sequence[int]] = None,
    batch_size: Optional[int] = None,
):
    model.eval()
    dataset = model.make_dataset(subset_indices=subset_indices_0)
    loader = torch.utils.data.DataLoader(dataset, batch_size=int(batch_size or model.hparams.batch_size), shuffle=False)

    records: Dict[str, list[Any]] = {
        "trajectory_sample_index": [],
        "dataset_sample_index": [],
        "raw_work": [],
        "tfep_work": [],
        "u_source": [],
        "u_target_raw": [],
        "u_target_mapped": [],
        "log_det_J": [],
        "log_weights": [],
    }
    with torch.no_grad():
        for batch in loader:
            dims = batch.get("dimensions", None)
            positions = batch["positions"]
            result = model.forward(batch)
            u_source = model.eval_source_potential(positions, dims) / model._kT
            u_target_raw = model.eval_target_potential(positions, dims) / model._kT
            u_target_mapped = model.eval_target_potential(result["positions"], dims) / model._kT
            log_det = result["log_det_J"]
            raw_work = u_target_raw - u_source
            tfep_work = (u_target_mapped - log_det) - u_source
            for key, tensor in (
                ("trajectory_sample_index", batch["trajectory_sample_index"]),
                ("dataset_sample_index", batch["dataset_sample_index"]),
                ("raw_work", raw_work),
                ("tfep_work", tfep_work),
                ("u_source", u_source),
                ("u_target_raw", u_target_raw),
                ("u_target_mapped", u_target_mapped),
                ("log_det_J", log_det),
            ):
                records[key].extend(tensor.detach().cpu().numpy().reshape(-1).tolist())
            if "log_weights" in batch:
                records["log_weights"].extend(batch["log_weights"].detach().cpu().numpy().reshape(-1).tolist())

    for key in list(records):
        dtype = np.int64 if key.endswith("sample_index") else np.float64
        records[key] = np.asarray(records[key], dtype=dtype)
    out = {
        "direction": "state0_state1",
        "raw": _work_summary(records["raw_work"]),
        "tfep": _work_summary(records["tfep_work"]),
        "state0_state1": records,
    }
    log_weights = records["log_weights"] if len(records["log_weights"]) == len(records["raw_work"]) and len(records["log_weights"]) > 0 else None
    if log_weights is not None:
        out["raw_reweighted"] = _weighted_work_summary(records["raw_work"], log_weights)
        out["tfep_reweighted"] = _weighted_work_summary(records["tfep_work"], log_weights)
    return out


def close_model_openmm_workers(model) -> None:
    if model is None:
        return
    for pot in (getattr(model, "_source_potential", None), getattr(model, "_potential_energy_func", None)):
        inner = getattr(pot, "pot", pot)
        close_fn = getattr(inner, "close", None)
        if callable(close_fn):
            close_fn()


def main() -> None:
    args = build_argparser().parse_args()
    artifacts = run_training(args)
    close_model_openmm_workers(artifacts.model)


if __name__ == "__main__":
    main()
