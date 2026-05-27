#!/usr/bin/env python3
"""
metane_tfep_tmbar_bidir_openmmonly.py

OpenMM-only bidirectional TFEP/TMBAR training on two trajectories.

NEW: Solvation-shell conditioning (KNN shells)
---------------------------------------------
You can condition the solute mapping on the *first and second solvation shells*
defined as K1 and K2 nearest water molecules (by O distance to a solute center)
*per frame*, while keeping a fixed conditioning input size.

How it works:
- We define "shell slots" as the first (K1+K2) water residues in the topology order.
- For each frame, we compute distances from solute center to all water oxygens.
- We permute *whole water molecules* (O+Hs) so that:
    slot 0..K1-1   contain the K1 nearest waters
    slot K1..K1+K2-1 contain the next K2 nearest waters
- We then set TFEP's conditioning_atoms to the atoms in those slots (fixed indices),
  so the flow can condition on shell waters without mapping solvent.

This relies on waters being identical (TIP3P) so permuting whole molecules does not
change energy/forces.

Robust OpenMM periodic box handling
-----------------------------------
Default behavior: set System default box vectors from CRYST1 and ignore per-frame
dimensions during OpenMM evaluation. This avoids the OpenMMException:
  "periodic box size has decreased to less than twice the nonbonded cutoff"

Use --no-fixed-box to disable and pass per-frame dimensions (may crash).

"""

from __future__ import annotations

import argparse
import atexit
import csv
import ctypes
import functools
import json
import multiprocessing as mp
import os
import re
import shutil
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import pint
import MDAnalysis as mda
from MDAnalysis.lib.mdamath import triclinic_vectors

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.plugins.environments import LightningEnvironment

from tfep.analysis.short_relaxation import (
    RelaxationDiagnosticConfig,
    run_short_relaxation_diagnostic,
    write_relaxation_outputs,
)
from tfep.stochastic.cli import add_stochastic_tfep_args
from tfep.stochastic.molecular import (
    MolecularStochasticConfig,
    evaluate_stochastic_direction,
    metadata_from_config as stochastic_metadata_from_config,
    resolve_snf_flat_indices,
)
from tfep.stochastic.training import StochasticTrainingConfig

from tfep.io.dataset import SolvationShellPermutingTrajectoryDataset, TrajectorySubset
from tfep.nn.flows.shell_water import (
    JointSoluteMAFAndShellWaterInternalFlow,
    ShellEquivariantWaterFlatFlow,
)
from tfep.solvation import (
    build_shell_equivariant_index_spec as _build_shell_equivariant_index_spec,
    compute_shell_slot_conditioning_indices,
    parse_csv_words,
    residue_sort_key as _residue_sort_key,
)


# -----------------------------------------------------------------------------
# OpenMM runtime bootstrap
# -----------------------------------------------------------------------------

def _bootstrap_openmm_runtime_env() -> None:
    """Point OpenMM/OpenCL to the active Python environment before importing OpenMM."""
    prefix = Path(sys.executable).resolve().parents[1]

    plugin_candidates: List[Path] = []
    if os.environ.get("OPENMM_PLUGIN_DIR"):
        plugin_candidates.append(Path(os.environ["OPENMM_PLUGIN_DIR"]).expanduser())
    plugin_candidates.extend([
        prefix / "lib" / "plugins",
        prefix / "Library" / "lib" / "plugins",
    ])
    for candidate in plugin_candidates:
        if candidate.exists():
            os.environ.setdefault("OPENMM_PLUGIN_DIR", str(candidate.resolve()))
            break

    if not os.environ.get("OCL_ICD_VENDORS") and not os.environ.get("OPENCL_VENDOR_PATH"):
        for candidate in (prefix / "etc" / "OpenCL" / "vendors", Path("/etc/OpenCL/vendors")):
            if candidate.exists():
                os.environ["OCL_ICD_VENDORS"] = str(candidate.resolve())
                break


_bootstrap_openmm_runtime_env()

try:
    from openmm import XmlSerializer, Platform, Vec3  # type: ignore
    from openmm import unit as omm_unit  # type: ignore
except ImportError as exc:
    XmlSerializer = None  # type: ignore[assignment]
    Platform = None  # type: ignore[assignment]
    Vec3 = None  # type: ignore[assignment]
    omm_unit = None  # type: ignore[assignment]
    _OPENMM_IMPORT_ERROR: Optional[ImportError] = exc
else:
    _OPENMM_IMPORT_ERROR = None

import tfep
from tfep.app.base import TMBARMapBase
from tfep.app.cartesianmaf import CartesianMAFMap
from tfep.app.mixedmaf import MixedMAFMap
from tfep.potentials.base import MultiStatePotential
from tfep.potentials.openmm import OpenMMPotential
from tfep.regularizers.bar import BARLikeRegularizer
from tfep.utils.misc import atom_to_flattened, flattened_to_atom, energies_array_to_tensor, forces_array_to_tensor

try:
    import pymbar
except Exception:
    pymbar = None


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _require_openmm() -> None:
    if _OPENMM_IMPORT_ERROR is not None:
        raise RuntimeError(
            "The small-molecule TMBAR OpenMM workflow requires OpenMM. "
            "Install it with `pip install openmm`, `conda install -c conda-forge openmm`, "
            "or the package extra `tfep[openmm]`."
        ) from _OPENMM_IMPORT_ERROR


def _as_indices_or_selection(x: Optional[str]) -> Optional[Union[str, List[int]]]:
    """If x looks like '1,2,3' convert to list[int], else keep as MDAnalysis selection string."""
    if x is None:
        return None
    s = x.strip()
    if re.fullmatch(r"[0-9,\s]+", s):
        return [int(t) for t in re.split(r"[,\s]+", s) if t != ""]
    return s


def _normalize_objective(objective: Optional[str]) -> str:
    normalized = str(objective or "kl").strip().lower()
    if normalized not in ("kl", "bar", "hybrid"):
        raise ValueError(f"Unsupported objective '{objective}'. Expected one of: kl, bar, hybrid.")
    return normalized


def _legacy_bar_regularizer_for_objective(objective: str, bar_lambda: float):
    """Legacy BAR regularizer is active only under KL objective."""
    if objective == "kl" and float(bar_lambda) > 0.0:
        return BARLikeRegularizer(lambda_bar=float(bar_lambda))
    return None


def _objective_mode_description(objective: str, lambda_bar: float, legacy_bar_lambda: float, logJ_penalty_weight: float = 0.0) -> str:
    logJ_note = ""
    if float(logJ_penalty_weight) > 0.0:
        logJ_note = f" + logJ penalty (weight={float(logJ_penalty_weight):.6g})"
    if objective == "bar":
        return f"BAR-only objective (Bennett objective from bidirectional works; no KL term).{logJ_note}"
    if objective == "hybrid":
        return f"Hybrid objective: KL + lambda_bar * BAR with lambda_bar={float(lambda_bar):.6g}.{logJ_note}"
    if float(legacy_bar_lambda) > 0.0:
        return (
            "KL objective (BoltzmannKLDivLoss) with legacy BAR regularizer enabled "
            f"(bar_lambda={float(legacy_bar_lambda):.6g}).{logJ_note}"
        )
    return f"KL objective (BoltzmannKLDivLoss) without legacy BAR regularizer.{logJ_note}"


def _resolve_state_inputs(
    state_dir: Path,
    traj_rel: str,
    system_xml: Optional[str],
    topology: Optional[str],
) -> Tuple[Path, Path, Path]:
    traj_path = (state_dir / traj_rel).resolve()
    system_xml_path = (state_dir / "system" / "system.xml").resolve() if system_xml is None else Path(system_xml).expanduser().resolve()
    topology_path = (state_dir / "system" / "start.pdb").resolve() if topology is None else Path(topology).expanduser().resolve()

    if not traj_path.exists():
        raise FileNotFoundError(f"Trajectory not found: {traj_path}")
    if not system_xml_path.exists():
        raise FileNotFoundError(f"System XML not found: {system_xml_path}")
    if not topology_path.exists():
        raise FileNotFoundError(f"Topology not found: {topology_path}")
    return traj_path, system_xml_path, topology_path


def _load_openmm_system(system_xml_path: Path):
    _require_openmm()
    return XmlSerializer.deserialize(system_xml_path.read_text())


def _available_openmm_platforms() -> List[str]:
    _require_openmm()
    return [Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())]


def _probe_opencl_runtime() -> Dict[str, Any]:
    """Query the low-level OpenCL ICD loader for visible platforms."""
    prefix = Path(sys.executable).resolve().parents[1]
    candidates = [
        prefix / "lib" / "libOpenCL.so.1",
        Path("/lib/x86_64-linux-gnu/libOpenCL.so.1"),
        Path("/usr/local/cuda/targets/x86_64-linux/lib/libOpenCL.so.1"),
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            lib = ctypes.cdll.LoadLibrary(str(candidate))
            num_platforms = ctypes.c_uint()
            err = lib.clGetPlatformIDs(0, None, ctypes.byref(num_platforms))
            return {
                "library": str(candidate),
                "clGetPlatformIDs_error": int(err),
                "num_platforms": int(num_platforms.value),
                "ocl_icd_vendors": os.environ.get("OCL_ICD_VENDORS", ""),
            }
        except Exception as exc:
            return {
                "library": str(candidate),
                "error": repr(exc),
                "ocl_icd_vendors": os.environ.get("OCL_ICD_VENDORS", ""),
            }
    return {
        "library": None,
        "error": "libOpenCL.so.1 not found",
        "ocl_icd_vendors": os.environ.get("OCL_ICD_VENDORS", ""),
    }


def _serialize_openmm_platform(platform) -> Tuple[Optional[str], Dict[str, str]]:
    """Convert an OpenMM platform object into a worker-safe name+properties pair."""
    _require_openmm()
    if platform is None:
        return None, {}
    if isinstance(platform, str):
        return platform, {}
    platform_name = platform.getName()
    platform_properties = {
        property_name: platform.getPropertyDefaultValue(property_name)
        for property_name in platform.getPropertyNames()
    }
    return platform_name, platform_properties


def _isolated_openmm_worker_main(
    conn,
    system_xml: str,
    platform_name: Optional[str],
    platform_properties: Dict[str, str],
    system_name: Optional[str],
):
    """Evaluate a single OpenMM state inside a dedicated subprocess."""
    _require_openmm()
    from tfep.potentials.openmm import _run_single_point_calculation, global_context_cache

    worker_system = XmlSerializer.deserialize(system_xml)
    worker_system_name = system_name or "isolated_openmm_state"

    try:
        while True:
            message = conn.recv()
            command = message.get("command")

            if command == "close":
                conn.send({"ok": True})
                break

            if command != "evaluate":
                raise ValueError(f"Unsupported worker command: {command!r}")

            batch_positions = message["positions"]
            batch_box_vectors = message["box_vectors"]
            return_forces = bool(message["return_forces"])

            results = [
                _run_single_point_calculation(
                    worker_system,
                    platform_name,
                    platform_properties,
                    worker_system_name,
                    return_forces,
                    positions,
                    box_vectors,
                )
                for positions, box_vectors in zip(batch_positions, batch_box_vectors)
            ]

            if return_forces:
                energies, forces = zip(*results)
                conn.send({
                    "ok": True,
                    "energies": np.asarray(energies, dtype=np.float64),
                    "forces": np.asarray(forces, dtype=np.float64),
                })
            else:
                conn.send({
                    "ok": True,
                    "energies": np.asarray(results, dtype=np.float64),
                })
    except EOFError:
        pass
    except Exception as exc:
        try:
            conn.send({
                "ok": False,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            })
        except Exception:
            pass
    finally:
        try:
            global_context_cache.pop(worker_system_name, None)
        except Exception:
            pass
        conn.close()


class IsolatedOpenMMWorker:
    """Persistent subprocess dedicated to one OpenMM state."""

    def __init__(self, system, platform, system_name: Optional[str]):
        _require_openmm()
        self._system_xml = XmlSerializer.serialize(system)
        self._platform_name, self._platform_properties = _serialize_openmm_platform(platform)
        self._system_name = system_name
        self._ctx = mp.get_context("spawn")
        self._conn = None
        self._process = None
        self._closed = False
        atexit.register(self.close)

    def _start(self) -> None:
        if self._closed:
            raise RuntimeError("OpenMM worker has already been closed")
        if self._process is not None:
            return
        parent_conn, child_conn = self._ctx.Pipe()
        process = self._ctx.Process(
            target=_isolated_openmm_worker_main,
            args=(child_conn, self._system_xml, self._platform_name, self._platform_properties, self._system_name),
            daemon=True,
        )
        process.start()
        child_conn.close()
        self._conn = parent_conn
        self._process = process

    def evaluate(self, batch_positions_nm: np.ndarray, box_vectors_nm: Sequence[Optional[np.ndarray]], *, return_forces: bool):
        self._start()
        assert self._conn is not None
        self._conn.send({
            "command": "evaluate",
            "positions": np.asarray(batch_positions_nm, dtype=np.float64),
            "box_vectors": list(box_vectors_nm),
            "return_forces": bool(return_forces),
        })
        reply = self._conn.recv()
        if not reply.get("ok", False):
            raise RuntimeError(
                "Isolated OpenMM worker failed: "
                f"{reply.get('error', 'unknown error')}\n{reply.get('traceback', '')}".rstrip()
            )
        return reply["energies"], reply.get("forces")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._conn is not None:
                try:
                    self._conn.send({"command": "close"})
                    self._conn.recv()
                except Exception:
                    pass
                self._conn.close()
        finally:
            self._conn = None

        if self._process is not None:
            self._process.join(timeout=2.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=2.0)
            self._process = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class IsolatedOpenMMPotentialEnergyFunc(torch.autograd.Function):
    """PyTorch autograd bridge for the isolated OpenMM worker."""

    @staticmethod
    def forward(
        ctx,
        batch_positions: torch.Tensor,
        worker: IsolatedOpenMMWorker,
        batch_cell: Optional[torch.Tensor] = None,
        positions_unit: Optional[pint.Unit] = None,
        energy_unit: Optional[pint.Unit] = None,
        precompute_gradient: bool = False,
    ):
        batch_positions_arr_nm, box_vectors_nm = _prepare_openmm_worker_inputs(batch_positions, batch_cell, positions_unit)

        # Sending multiple mapped configurations in a single worker request can
        # still trigger low-level OpenMM crashes in this no-fixed-box path.
        # Streaming one sample at a time through the same persistent worker
        # keeps the stable execution path while preserving context reuse.
        energy_chunks: List[np.ndarray] = []
        force_chunks: List[np.ndarray] = []
        for positions_nm, box_vectors in zip(batch_positions_arr_nm, box_vectors_nm):
            energies_i, forces_i = worker.evaluate(
                np.expand_dims(positions_nm, axis=0),
                [box_vectors],
                return_forces=bool(precompute_gradient),
            )
            energy_chunks.append(np.asarray(energies_i, dtype=np.float64))
            if precompute_gradient:
                force_chunks.append(np.asarray(forces_i, dtype=np.float64))

        energies = np.concatenate(energy_chunks, axis=0) if energy_chunks else np.empty(0, dtype=np.float64)
        forces = np.concatenate(force_chunks, axis=0) if precompute_gradient else None

        energies_tensor = _convert_openmm_energies_to_tensor(energies, energy_unit, like=batch_positions)

        if precompute_gradient:
            ctx.forces = np.asarray(forces, dtype=np.float64)
            ctx.energy_unit = energy_unit
            ctx.positions_unit = positions_unit
        else:
            ctx.forces = None

        return energies_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = [None for _ in range(6)]

        if ctx.needs_input_grad[0]:
            if ctx.forces is None:
                raise ValueError("precompute_gradient must be set to True for backward pass.")

            if (ctx.energy_unit is None) and (ctx.positions_unit is None):
                forces = torch.from_numpy(atom_to_flattened(ctx.forces))
            else:
                ureg = ctx.energy_unit._REGISTRY
                default_positions_unit = OpenMMPotential.default_positions_unit(ureg)
                default_energy_unit = OpenMMPotential.default_energy_unit(ureg)
                force_quantity = ctx.forces * default_energy_unit / default_positions_unit
                forces = forces_array_to_tensor(force_quantity, ctx.positions_unit, ctx.energy_unit)
            forces = forces.to(grad_output)
            grad_input[0] = -forces * grad_output[:, None]

        return tuple(grad_input)


def isolated_openmm_potential_energy(
    batch_positions: torch.Tensor,
    worker: IsolatedOpenMMWorker,
    batch_cell: Optional[torch.Tensor] = None,
    positions_unit: Optional[pint.Unit] = None,
    energy_unit: Optional[pint.Unit] = None,
    precompute_gradient: bool = False,
):
    return IsolatedOpenMMPotentialEnergyFunc.apply(
        batch_positions,
        worker,
        batch_cell,
        positions_unit,
        energy_unit,
        precompute_gradient,
    )


def _prepare_openmm_worker_inputs(
    batch_positions: torch.Tensor,
    batch_cell: Optional[torch.Tensor],
    positions_unit: Optional[pint.Unit],
) -> Tuple[np.ndarray, List[Optional[np.ndarray]]]:
    batch_positions_arr_nm = flattened_to_atom(batch_positions.detach().cpu().numpy())
    if positions_unit is not None:
        default_positions_unit = OpenMMPotential.default_positions_unit(positions_unit._REGISTRY)
        batch_positions_arr_nm = (batch_positions_arr_nm * positions_unit).to(default_positions_unit).magnitude

    if batch_cell is None:
        box_vectors_nm = [None for _ in range(batch_positions.shape[0])]
    else:
        batch_cell_arr_nm = batch_cell.detach().cpu().numpy().copy()
        if positions_unit is not None:
            default_positions_unit = OpenMMPotential.default_positions_unit(positions_unit._REGISTRY)
            batch_cell_arr_nm[:, :3] = (batch_cell_arr_nm[:, :3] * positions_unit).to(default_positions_unit).magnitude
        box_vectors_nm = [np.asarray(triclinic_vectors(x), dtype=np.float64) for x in batch_cell_arr_nm]

    return np.asarray(batch_positions_arr_nm, dtype=np.float64), box_vectors_nm


def _convert_openmm_energies_to_tensor(
    energies: np.ndarray,
    energy_unit: Optional[pint.Unit],
    *,
    like: torch.Tensor,
) -> torch.Tensor:
    if energy_unit is None:
        energies_tensor = torch.tensor(energies)
    else:
        energies_quantity = energies * OpenMMPotential.default_energy_unit(energy_unit._REGISTRY)
        energies_tensor = energies_array_to_tensor(energies_quantity, energy_unit)
    return energies_tensor.to(like)


def isolated_openmm_potential_energy_no_grad(
    batch_positions: torch.Tensor,
    worker: IsolatedOpenMMWorker,
    batch_cell: Optional[torch.Tensor] = None,
    positions_unit: Optional[pint.Unit] = None,
    energy_unit: Optional[pint.Unit] = None,
) -> torch.Tensor:
    batch_positions_arr_nm, box_vectors_nm = _prepare_openmm_worker_inputs(batch_positions, batch_cell, positions_unit)
    energy_chunks: List[np.ndarray] = []
    for positions_nm, box_vectors in zip(batch_positions_arr_nm, box_vectors_nm):
        energies_i, _ = worker.evaluate(
            np.expand_dims(positions_nm, axis=0),
            [box_vectors],
            return_forces=False,
        )
        energy_chunks.append(np.asarray(energies_i, dtype=np.float64))
    energies = np.concatenate(energy_chunks, axis=0) if energy_chunks else np.empty(0, dtype=np.float64)
    return _convert_openmm_energies_to_tensor(energies, energy_unit, like=batch_positions)


class IsolatedOpenMMPotential(torch.nn.Module):
    """OpenMM potential evaluated in a dedicated subprocess."""

    def __init__(
        self,
        system,
        platform,
        *,
        positions_unit: Optional[pint.Unit] = None,
        energy_unit: Optional[pint.Unit] = None,
        system_name: Optional[str] = None,
        precompute_gradient: bool = False,
    ):
        super().__init__()
        self.system = system
        self.platform = platform
        self.system_name = system_name
        self.precompute_gradient = precompute_gradient
        self.positions_unit = positions_unit
        self.energy_unit = energy_unit
        self._worker = IsolatedOpenMMWorker(system=system, platform=platform, system_name=system_name)

    def forward(self, batch_positions: torch.Tensor, batch_cell: Optional[torch.Tensor] = None) -> torch.Tensor:
        return isolated_openmm_potential_energy(
            batch_positions,
            self._worker,
            batch_cell=batch_cell,
            positions_unit=self.positions_unit,
            energy_unit=self.energy_unit,
            precompute_gradient=self.precompute_gradient,
        )

    def energy_no_grad(self, batch_positions: torch.Tensor, batch_cell: Optional[torch.Tensor] = None) -> torch.Tensor:
        return isolated_openmm_potential_energy_no_grad(
            batch_positions,
            self._worker,
            batch_cell=batch_cell,
            positions_unit=self.positions_unit,
            energy_unit=self.energy_unit,
        )

    def close(self) -> None:
        self._worker.close()


def _configure_openmm_platform(name: str, device: str, precision: str, cpu_threads: int):
    _require_openmm()
    available = _available_openmm_platforms()
    default_plugin_dir = Platform.getDefaultPluginsDirectory()
    configured_plugin_dir = os.environ.get("OPENMM_PLUGIN_DIR", "")
    if name not in available and configured_plugin_dir:
        try:
            if Path(configured_plugin_dir).resolve() != Path(default_plugin_dir).resolve():
                Platform.loadPluginsFromDirectory(configured_plugin_dir)
                available = _available_openmm_platforms()
        except Exception:
            pass

    try:
        platform = Platform.getPlatformByName(name)
    except Exception as exc:
        diagnostics = [
            f"requested={name!r}",
            f"available={available}",
            f"default_plugin_dir={default_plugin_dir}",
            f"OPENMM_PLUGIN_DIR={configured_plugin_dir!r}",
            f"plugin_load_failures={tuple(Platform.getPluginLoadFailures())}",
        ]
        if str(name).upper() == "OPENCL":
            diagnostics.append(f"opencl_runtime={_probe_opencl_runtime()}")
        raise RuntimeError("Failed to configure OpenMM platform: " + " | ".join(diagnostics)) from exc

    prop_names = set(platform.getPropertyNames())

    if name.upper() == "CUDA":
        if "DeviceIndex" in prop_names:
            platform.setPropertyDefaultValue("DeviceIndex", str(device))
        if "CudaPrecision" in prop_names:
            platform.setPropertyDefaultValue("CudaPrecision", str(precision))
    elif name.upper() == "OpenCL":
        if "OpenCLDeviceIndex" in prop_names:
            platform.setPropertyDefaultValue("OpenCLDeviceIndex", str(device))
        if "OpenCLPrecision" in prop_names:
            platform.setPropertyDefaultValue("OpenCLPrecision", str(precision))
    elif name.upper() == "CPU":
        if "Threads" in prop_names:
            platform.setPropertyDefaultValue("Threads", str(int(cpu_threads)))
    return platform


def _read_cryst1_dims(pdb_path: Path) -> Tuple[float, float, float, float, float, float]:
    """Return (a,b,c,alpha,beta,gamma) from CRYST1 in Å and degrees."""
    cryst = None
    for line in pdb_path.read_text().splitlines():
        if line.startswith("CRYST1"):
            cryst = line
            break
    if cryst is None:
        raise RuntimeError(f"No CRYST1 record found in {pdb_path}. Cannot set default box.")

    a = float(cryst[6:15])
    b = float(cryst[15:24])
    c = float(cryst[24:33])
    alpha = float(cryst[33:40])
    beta  = float(cryst[40:47])
    gamma = float(cryst[47:54])
    return a, b, c, alpha, beta, gamma


def set_system_default_box_from_cryst1(system, pdb_path: Path) -> None:
    """
    Set System default periodic box vectors from CRYST1.

    For orthorhombic boxes (alpha=beta=gamma=90) we set diagonal vectors.
    """
    _require_openmm()
    aA, bA, cA, alpha, beta, gamma = _read_cryst1_dims(pdb_path)
    ax = aA * 0.1
    by = bA * 0.1
    cz = cA * 0.1

    if abs(alpha - 90.0) < 1e-6 and abs(beta - 90.0) < 1e-6 and abs(gamma - 90.0) < 1e-6:
        a = Vec3(ax, 0.0, 0.0) * omm_unit.nanometer
        b = Vec3(0.0, by, 0.0) * omm_unit.nanometer
        c = Vec3(0.0, 0.0, cz) * omm_unit.nanometer
        system.setDefaultPeriodicBoxVectors(a, b, c)
        return

    # Generic triclinic fallback
    from MDAnalysis.lib.mdamath import triclinic_vectors
    dims = np.array([aA, bA, cA, alpha, beta, gamma], dtype=float)
    M_A = np.array(triclinic_vectors(dims), dtype=float)  # 3x3 in Å
    v0 = Vec3(*(M_A[0] * 0.1)) * omm_unit.nanometer
    v1 = Vec3(*(M_A[1] * 0.1)) * omm_unit.nanometer
    v2 = Vec3(*(M_A[2] * 0.1)) * omm_unit.nanometer
    system.setDefaultPeriodicBoxVectors(v0, v1, v2)


def _select_atom_indices_from_spec(universe: mda.Universe, spec) -> np.ndarray:
    if spec is None:
        return universe.atoms.indices.astype(int)
    if isinstance(spec, str):
        atoms = universe.select_atoms(spec)
        if len(atoms) == 0:
            raise ValueError(f"Atom selection returned 0 atoms: {spec!r}")
        return atoms.indices.astype(int)
    arr = np.asarray(spec, dtype=int).reshape(-1)
    if arr.size == 0:
        raise ValueError("Atom index selection is empty")
    return arr


def compute_residue_wrap_blocks(topology_pdb: Path) -> List[np.ndarray]:
    """
    Determine residue blocks that should be wrapped as rigid units before OpenMM
    evaluation when per-frame box vectors are enabled.

    For the local bromomethane workflow each residue is a separate molecule
    (solute or water), so whole-residue wrapping preserves intramolecular
    geometry while keeping all molecules inside the primary cell.
    """
    u = mda.Universe(str(topology_pdb))
    blocks: List[np.ndarray] = []
    for res in sorted(u.residues, key=_residue_sort_key):
        blocks.append(res.atoms.indices.astype(int))
    return blocks


class NoCellWrapper(torch.nn.Module):
    """
    Wrapper that IGNORES per-frame dimensions and relies on System default box vectors.

    This prevents tfep.potentials.openmm from calling context.setPeriodicBoxVectors(),
    which is where the "box < 2*cutoff" exception was being triggered.

    Proxies units used by TFEP:
      - positions_unit
      - energy_unit
    and delegates other attributes.
    """
    def __init__(self, pot: torch.nn.Module):
        super().__init__()
        self.pot = pot
        self.positions_unit = getattr(pot, "positions_unit", None)
        self.energy_unit = getattr(pot, "energy_unit", None)

    def forward(self, positions: torch.Tensor, dimensions: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.pot(positions)  # IMPORTANT: do NOT pass dimensions

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.pot, name)


class WrappedCellBoxWrapper(torch.nn.Module):
    """
    Wrap mapped residues into the current orthorhombic box before OpenMM energy
    evaluation. This avoids low-level OpenMM crashes on mapped coordinates that
    drift outside the primary cell when per-frame box vectors are enabled.
    """

    def __init__(self, pot: torch.nn.Module, wrap_blocks: Sequence[Sequence[int]]):
        super().__init__()
        self.pot = pot
        self.positions_unit = getattr(pot, "positions_unit", None)
        self.energy_unit = getattr(pot, "energy_unit", None)
        self._wrap_blocks = [torch.as_tensor(block, dtype=torch.long) for block in wrap_blocks if len(block) > 0]
        self._warned_non_orthorhombic = False

    def _wrap_positions(self, positions: torch.Tensor, dimensions: Optional[torch.Tensor]) -> torch.Tensor:
        if dimensions is None or len(self._wrap_blocks) == 0:
            return positions

        if positions.ndim != 2:
            return positions

        if dimensions.ndim == 1:
            dims = dimensions.unsqueeze(0)
        else:
            dims = dimensions

        if dims.shape[-1] < 3:
            return positions

        lengths = dims[:, :3].to(dtype=positions.dtype, device=positions.device)
        if not torch.all(torch.isfinite(lengths)) or torch.any(lengths <= 0):
            return positions

        if dims.shape[-1] >= 6:
            angles = dims[:, 3:6].to(dtype=positions.dtype, device=positions.device)
            if not torch.allclose(angles, torch.full_like(angles, 90.0), atol=1e-4, rtol=0.0):
                if not self._warned_non_orthorhombic:
                    print("[box] non-orthorhombic per-frame box detected; mapped-residue wrapping disabled")
                    self._warned_non_orthorhombic = True
                return positions

        wrapped = positions.reshape(positions.shape[0], -1, 3).clone()
        lengths = lengths.unsqueeze(1)
        for block in self._wrap_blocks:
            block = block.to(device=positions.device)
            ref = wrapped.index_select(1, block).mean(dim=1)
            shift = torch.floor(ref / lengths[:, 0, :]) * lengths[:, 0, :]
            wrapped[:, block, :] = wrapped.index_select(1, block) - shift.unsqueeze(1)
        return wrapped.reshape_as(positions)

    def forward(self, positions: torch.Tensor, dimensions: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.pot(self._wrap_positions(positions, dimensions), dimensions)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.pot, name)


class NoFixedBoxResidueWrapMixin:
    """Wrap whole residues into the current orthorhombic box before evaluation."""

    def _init_box_eval_wrap(self, wrap_box_eval: bool, wrap_residue_blocks: Optional[Sequence[Sequence[int]]]) -> None:
        self._wrap_box_eval = bool(wrap_box_eval)
        self._wrap_residue_blocks = [torch.as_tensor(block, dtype=torch.long) for block in (wrap_residue_blocks or []) if len(block) > 0]
        self._warned_non_orthorhombic_box_eval = False

    def _wrap_positions_for_box_eval(self, positions: torch.Tensor, dimensions: Optional[torch.Tensor]) -> torch.Tensor:
        if not getattr(self, "_wrap_box_eval", False):
            return positions
        if dimensions is None or positions.ndim != 2 or len(getattr(self, "_wrap_residue_blocks", [])) == 0:
            return positions

        dims = dimensions.unsqueeze(0) if dimensions.ndim == 1 else dimensions
        if dims.shape[-1] < 3:
            return positions

        lengths = dims[:, :3].to(dtype=positions.dtype, device=positions.device)
        if not torch.all(torch.isfinite(lengths)) or torch.any(lengths <= 0):
            return positions

        if dims.shape[-1] >= 6:
            angles = dims[:, 3:6].to(dtype=positions.dtype, device=positions.device)
            if not torch.allclose(angles, torch.full_like(angles, 90.0), atol=1e-4, rtol=0.0):
                if not self._warned_non_orthorhombic_box_eval:
                    print("[box] non-orthorhombic per-frame box detected; residue wrapping disabled")
                    self._warned_non_orthorhombic_box_eval = True
                return positions

        wrapped = positions.reshape(positions.shape[0], -1, 3).clone()
        for block in self._wrap_residue_blocks:
            block = block.to(device=positions.device)
            ref = wrapped.index_select(1, block).mean(dim=1)
            shift = torch.floor(ref / lengths) * lengths
            wrapped[:, block, :] = wrapped.index_select(1, block) - shift.unsqueeze(1)
        return wrapped.reshape_as(positions)

    def _eval_potential(self, state: int, positions: torch.Tensor, dimensions: Optional[torch.Tensor]):
        positions = self._wrap_positions_for_box_eval(positions, dimensions)
        pot = self._potential_energy_func

        # Bypass MultiStatePotential.energy() here: in this local no-fixed-box path
        # it can segfault even when the per-state OpenMM potential evaluates the
        # same wrapped coordinates correctly.
        if hasattr(pot, "potentials"):
            pot_state = pot.potentials[int(state)]
        elif isinstance(pot, (list, tuple)):
            pot_state = pot[int(state)]
        elif isinstance(pot, dict):
            pot_state = pot[int(state)]
        else:
            return TMBARMapBase._eval_potential(self, state, positions, dimensions)

        if dimensions is None:
            return pot_state(positions)
        try:
            return pot_state(positions, dimensions)
        except TypeError:
            return pot_state(positions)


# -----------------------------------------------------------------------------
# Solvation-shell conditioning dataset
# -----------------------------------------------------------------------------

def _parse_csv_names(s: str) -> List[str]:
    return parse_csv_words(s)


def _load_index_array(path: Optional[Union[str, Path]]) -> Optional[np.ndarray]:
    if path is None:
        return None

    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Index file not found: {p}")

    if p.suffix == ".npy":
        arr = np.load(p)
    elif p.suffix == ".npz":
        data = np.load(p)
        if len(data.files) != 1:
            raise ValueError(f"Expected exactly one array in {p}, found {list(data.files)}")
        arr = data[data.files[0]]
    elif p.suffix == ".json":
        payload = json.loads(p.read_text())
        if isinstance(payload, dict):
            if "indices" not in payload:
                raise ValueError(f"JSON index file {p} must contain an 'indices' field")
            payload = payload["indices"]
        arr = np.asarray(payload)
    else:
        arr = np.loadtxt(p, dtype=int)

    arr = np.asarray(arr, dtype=int).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"Index file {p} is empty")
    return arr


def _normalize_subset_indices(indices: Optional[Sequence[int]], length: int, label: str) -> Optional[np.ndarray]:
    if indices is None:
        return None

    idx = np.asarray(indices, dtype=int).reshape(-1)
    if idx.size == 0:
        raise ValueError(f"{label}: empty subset indices")
    if np.any(idx < 0) or np.any(idx >= int(length)):
        bad = idx[(idx < 0) | (idx >= int(length))][:10]
        raise ValueError(f"{label}: found out-of-range indices {bad.tolist()} for dataset length {length}")
    if np.unique(idx).size != idx.size:
        raise ValueError(f"{label}: duplicate indices are not allowed")
    return idx


def _subset_dataset(dataset, subset_indices: Optional[Sequence[int]], label: str):
    if subset_indices is None:
        return dataset
    idx = _normalize_subset_indices(subset_indices, len(dataset), label)
    return TrajectorySubset(dataset, idx)


# -----------------------------------------------------------------------------
# Bonds helper for MixedMAF
# -----------------------------------------------------------------------------

def _add_water_bonds_if_missing(universe: mda.Universe, water_sel: str = "water") -> List[Tuple[int, int]]:
    """
    Add O-H bonds inside water residues if topology has no bonds.
    Returns list of (i,j) pairs in global atom indices.
    """
    bonds: List[Tuple[int, int]] = []

    wat_atoms = universe.select_atoms(water_sel)
    if len(wat_atoms) == 0:
        return bonds

    for res in wat_atoms.residues:
        # common TIP3P naming patterns
        O = None
        Hs: List[int] = []

        # try by element
        O_ag = res.atoms.select_atoms("element O")
        if len(O_ag) > 0:
            O = int(O_ag.indices[0])

        H_ag = res.atoms.select_atoms("element H")
        if len(H_ag) >= 2:
            Hs = [int(i) for i in H_ag.indices[:2]]

        # fallback by names if element missing
        if O is None:
            for nm in ("O", "OW"):
                ag = res.atoms.select_atoms(f"name {nm}")
                if len(ag) > 0:
                    O = int(ag.indices[0])
                    break
        if len(Hs) < 2:
            for nm in ("H1", "H2", "HW1", "HW2"):
                ag = res.atoms.select_atoms(f"name {nm}")
                if len(ag) > 0:
                    Hs.append(int(ag.indices[0]))
            Hs = Hs[:2]

        if O is not None and len(Hs) == 2:
            bonds.append((O, Hs[0]))
            bonds.append((O, Hs[1]))

    return bonds


def _ensure_bonds_for_mixed_coordinates(universe: mda.Universe) -> None:
    """
    MixedMAF needs connectivity (bonds) to build fragments/Z-matrices.

    If the topology lacks bonds, we:
      1) add water O-H bonds deterministically
      2) guess bonds for non-water atoms (solute etc.), mapping local->global indices
    """
    try:
        n_bonds = len(universe.bonds)
    except Exception:
        n_bonds = 0
    if n_bonds > 0:
        return

    universe.trajectory[0]
    dims = getattr(universe.trajectory.ts, "dimensions", None)

    bonds: List[Tuple[int, int]] = []
    bonds.extend(_add_water_bonds_if_missing(universe, water_sel="water"))

    # guess bonds for non-water atoms
    from MDAnalysis.topology.guessers import guess_bonds
    try:
        nonwat = universe.select_atoms("not water")
    except Exception:
        nonwat = universe.atoms

    if len(nonwat) > 0:
        gb = guess_bonds(nonwat, nonwat.positions, box=dims)
        # guess_bonds returns pairs in local indices for the passed AtomGroup
        nonwat_global = nonwat.indices.astype(int)
        for i, j in gb:
            bonds.append((int(nonwat_global[int(i)]), int(nonwat_global[int(j)])))

    if len(bonds) == 0:
        raise RuntimeError("MixedMAF requires bonds, but none could be built/guessed.")

    # Remove duplicates
    bonds = list({(min(i, j), max(i, j)) for (i, j) in bonds})
    universe.add_TopologyAttr("bonds", bonds)


def _build_conditioned_or_plain_dataset(
    universe: mda.Universe,
    *,
    shell_enabled: bool,
    shell_center: Optional[Union[str, Sequence[int]]],
    mapped_atoms,
    shell_water_selection: str,
    shell_oxygen_names: Sequence[str],
    shell_k1: int,
    shell_k2: int,
):
    if shell_enabled:
        center = shell_center if shell_center is not None else mapped_atoms
        return SolvationShellPermutingTrajectoryDataset(
            universe=universe,
            center_atoms=center,
            water_selection=shell_water_selection,
            oxygen_names=shell_oxygen_names,
            k1=shell_k1,
            k2=shell_k2,
        )
    return tfep.io.TrajectoryDataset(universe=universe)


# -----------------------------------------------------------------------------
# Bidirectional map classes
# -----------------------------------------------------------------------------

class SmallMolBidirectionalTMBARMapCartesian(NoFixedBoxResidueWrapMixin, TMBARMapBase, CartesianMAFMap):
    def __init__(
        self,
        potential_0: torch.nn.Module,
        potential_1: torch.nn.Module,
        topology_file_path: str,
        coordinates_file_path: str,
        coordinates_file_path_2: str,
        temperature: pint.Quantity,
        *,
        batch_size: int,
        mapped_atoms=None,
        conditioning_atoms=None,
        origin_atom=None,
        axes_atoms=None,
        tfep_logger_dir_path: str,
        dataloader_kwargs: Optional[dict] = None,
        n_maf_layers: int = 6,
        maf_hidden_layers: int = 2,
        maf_weight_norm: bool = True,
        maf_initialize_identity: bool = True,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        bar_lambda: float = 0.0,
        # shell conditioning
        shell_k1: int = 0,
        shell_k2: int = 0,
        shell_center: Optional[Union[str, Sequence[int]]] = None,
        shell_water_selection: str = "water",
        shell_oxygen_names: Sequence[str] = ("O", "OW"),
        train_indices_0: Optional[Sequence[int]] = None,
        train_indices_1: Optional[Sequence[int]] = None,
        wrap_box_eval: bool = False,
        wrap_residue_blocks: Optional[Sequence[Sequence[int]]] = None,
        **kwargs,
    ):
        self._lr = float(lr)
        self._weight_decay = float(weight_decay)
        objective = _normalize_objective(kwargs.get("objective", "kl"))
        bar_reg = _legacy_bar_regularizer_for_objective(objective, float(bar_lambda))

        self._shell_k1 = int(shell_k1)
        self._shell_k2 = int(shell_k2)
        self._shell_enabled = (self._shell_k1 + self._shell_k2) > 0
        self._shell_center = shell_center
        self._shell_water_selection = str(shell_water_selection)
        self._shell_oxygen_names = list(shell_oxygen_names)
        self._train_indices_0 = None if train_indices_0 is None else np.asarray(train_indices_0, dtype=int)
        self._train_indices_1 = None if train_indices_1 is None else np.asarray(train_indices_1, dtype=int)
        self._init_box_eval_wrap(wrap_box_eval=wrap_box_eval, wrap_residue_blocks=wrap_residue_blocks)

        super().__init__(
            potential_energy_func=MultiStatePotential(potential_0, potential_1),
            topology_file_path=topology_file_path,
            coordinates_file_path=coordinates_file_path,
            coordinates_file_path_2=coordinates_file_path_2,
            temperature=temperature,
            batch_size=batch_size,
            mapped_atoms=mapped_atoms,
            conditioning_atoms=conditioning_atoms,
            origin_atom=origin_atom,
            axes_atoms=axes_atoms,
            tfep_logger_dir_path=tfep_logger_dir_path,
            dataloader_kwargs=dataloader_kwargs,
            n_states=2,
            state_names=["state0", "state1"],
            bar_regularizer=bar_reg,
            **kwargs,
        )

        self.save_hyperparameters("n_maf_layers")
        self.kwargs = dict(
            hidden_layers=int(maf_hidden_layers),
            weight_norm=bool(maf_weight_norm),
            initialize_identity=bool(maf_initialize_identity),
        )

    def make_dataset(self, state: int, subset_indices: Optional[Sequence[int]] = None):
        if int(state) == 0:
            universe = self.create_universe()
            label = "state0"
        elif int(state) == 1:
            universe = self.create_universe_2()
            label = "state1"
        else:
            raise ValueError(f"Unsupported state index: {state}")

        dataset = _build_conditioned_or_plain_dataset(
            universe=universe,
            shell_enabled=self._shell_enabled,
            shell_center=self._shell_center,
            mapped_atoms=self.hparams.mapped_atoms,
            shell_water_selection=self._shell_water_selection,
            shell_oxygen_names=self._shell_oxygen_names,
            shell_k1=self._shell_k1,
            shell_k2=self._shell_k2,
        )
        return _subset_dataset(dataset, subset_indices, label=label)

    def create_dataset(self):
        return self.make_dataset(0, subset_indices=self._train_indices_0)

    def create_dataset_2(self):
        return self.make_dataset(1, subset_indices=self._train_indices_1)

    def configure_flow(self):
        return CartesianMAFMap.configure_flow(self)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self._lr, weight_decay=self._weight_decay)




class SmallMolBidirectionalTMBARMapShellEquivariant(SmallMolBidirectionalTMBARMapCartesian):
    """
    Diagnostic joint solute + shell-water flow.

    This reuses the Cartesian bidirectional TMBAR machinery but replaces the
    standard Cartesian flow with:

        solute exact autoregressive affine MAF
        followed by shell-water internal exact equivariant flow.

    Important:
    - mapped_atoms must already be set to solute + all water atoms.
    - The solute block is transformed and optimized again.
    - The water block rotates/deforms current-shell water H atoms around fixed O atoms.
    - The total log_det is the exact sum of solute MAF logJ and internal water logJ.
    """

    def __init__(
        self,
        *args,
        shell_equiv_spec: Dict[str, Any],
        shell_equiv_cutoff_angstrom: float = 4.5,
        shell_equiv_tau_angstrom: float = 0.3,
        shell_equiv_hidden_dim: int = 64,
        shell_equiv_max_displacement_angstrom: float = 0.05,
        shell_equiv_initial_log_scale: float = -1.0,
        shell_equiv_top_k: int = 12,
        shell_equiv_max_rotation_radians: float = 0.35,
        shell_equiv_max_internal_log_scale: float = 0.06,
        shell_equiv_solute_maf_layers: int = 2,
        shell_equiv_solute_maf_hidden_dim: int = 128,
        shell_equiv_solute_maf_max_log_scale: float = 0.20,
        shell_equiv_solute_maf_max_shift_angstrom: float = 0.20,
        **kwargs,
    ):
        self._shell_equiv_spec = shell_equiv_spec
        self._shell_equiv_cutoff_angstrom = float(shell_equiv_cutoff_angstrom)
        self._shell_equiv_tau_angstrom = float(shell_equiv_tau_angstrom)
        self._shell_equiv_hidden_dim = int(shell_equiv_hidden_dim)
        self._shell_equiv_max_displacement_angstrom = float(shell_equiv_max_displacement_angstrom)
        self._shell_equiv_initial_log_scale = float(shell_equiv_initial_log_scale)
        self._shell_equiv_top_k = int(shell_equiv_top_k)
        self._shell_equiv_max_rotation_radians = float(shell_equiv_max_rotation_radians)
        self._shell_equiv_max_internal_log_scale = float(shell_equiv_max_internal_log_scale)
        self._shell_equiv_solute_maf_layers = int(shell_equiv_solute_maf_layers)
        self._shell_equiv_solute_maf_hidden_dim = int(shell_equiv_solute_maf_hidden_dim)
        self._shell_equiv_solute_maf_max_log_scale = float(shell_equiv_solute_maf_max_log_scale)
        self._shell_equiv_solute_maf_max_shift_angstrom = float(shell_equiv_solute_maf_max_shift_angstrom)
        super().__init__(*args, **kwargs)

    def configure_flow(self):
        spec = self._shell_equiv_spec

        water_flow = ShellEquivariantWaterFlatFlow(
            n_mapped_atoms=int(spec["n_mapped_atoms"]),
            solute_local_indices=spec["solute_local_indices"],
            water_oxygen_local_indices=spec["water_oxygen_local_indices"],
            water_h1_local_indices=spec["water_h1_local_indices"],
            water_h2_local_indices=spec["water_h2_local_indices"],
            water_atom_local_flat_indices=spec.get("water_atom_local_flat_indices"),
            water_atom_owner_indices=spec.get("water_atom_owner_indices"),
            cutoff_angstrom=float(self._shell_equiv_cutoff_angstrom),
            tau_angstrom=float(self._shell_equiv_tau_angstrom),
            hidden_dim=int(self._shell_equiv_hidden_dim),
            max_displacement_angstrom=float(self._shell_equiv_max_displacement_angstrom),
            initial_log_scale=float(self._shell_equiv_initial_log_scale),
            top_k=int(self._shell_equiv_top_k),
            max_rotation_radians=float(self._shell_equiv_max_rotation_radians),
            max_internal_log_scale=float(self._shell_equiv_max_internal_log_scale),
        )

        return JointSoluteMAFAndShellWaterInternalFlow(
            n_mapped_atoms=int(spec["n_mapped_atoms"]),
            solute_local_indices=spec["solute_local_indices"],
            water_flow=water_flow,
            solute_maf_layers=int(self._shell_equiv_solute_maf_layers),
            solute_maf_hidden_dim=int(self._shell_equiv_solute_maf_hidden_dim),
            solute_maf_max_log_scale=float(self._shell_equiv_solute_maf_max_log_scale),
            solute_maf_max_shift_angstrom=float(self._shell_equiv_solute_maf_max_shift_angstrom),
        )


class SmallMolBidirectionalTMBARMapMixed(NoFixedBoxResidueWrapMixin, TMBARMapBase, MixedMAFMap):
    def __init__(
        self,
        potential_0: torch.nn.Module,
        potential_1: torch.nn.Module,
        topology_file_path: str,
        coordinates_file_path: str,
        coordinates_file_path_2: str,
        temperature: pint.Quantity,
        *,
        batch_size: int,
        mapped_atoms=None,
        conditioning_atoms=None,
        origin_atom=None,
        axes_atoms=None,
        tfep_logger_dir_path: str,
        dataloader_kwargs: Optional[dict] = None,
        n_maf_layers: int = 6,
        remove_translation: bool = False,
        remove_rotation: bool = False,
        distance_lower_limit_displacement_angstrom: float = 0.3,
        maf_hidden_layers: int = 2,
        maf_weight_norm: bool = True,
        maf_initialize_identity: bool = True,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        bar_lambda: float = 0.0,
        # shell conditioning
        shell_k1: int = 0,
        shell_k2: int = 0,
        shell_center: Optional[Union[str, Sequence[int]]] = None,
        shell_water_selection: str = "water",
        shell_oxygen_names: Sequence[str] = ("O", "OW"),
        train_indices_0: Optional[Sequence[int]] = None,
        train_indices_1: Optional[Sequence[int]] = None,
        wrap_box_eval: bool = False,
        wrap_residue_blocks: Optional[Sequence[Sequence[int]]] = None,
        **kwargs,
    ):
        self._lr = float(lr)
        self._weight_decay = float(weight_decay)
        objective = _normalize_objective(kwargs.get("objective", "kl"))
        bar_reg = _legacy_bar_regularizer_for_objective(objective, float(bar_lambda))

        self._shell_k1 = int(shell_k1)
        self._shell_k2 = int(shell_k2)
        self._shell_enabled = (self._shell_k1 + self._shell_k2) > 0
        self._shell_center = shell_center
        self._shell_water_selection = str(shell_water_selection)
        self._shell_oxygen_names = list(shell_oxygen_names)
        self._train_indices_0 = None if train_indices_0 is None else np.asarray(train_indices_0, dtype=int)
        self._train_indices_1 = None if train_indices_1 is None else np.asarray(train_indices_1, dtype=int)
        self._init_box_eval_wrap(wrap_box_eval=wrap_box_eval, wrap_residue_blocks=wrap_residue_blocks)

        super().__init__(
            potential_energy_func=MultiStatePotential(potential_0, potential_1),
            topology_file_path=topology_file_path,
            coordinates_file_path=coordinates_file_path,
            coordinates_file_path_2=coordinates_file_path_2,
            temperature=temperature,
            batch_size=batch_size,
            mapped_atoms=mapped_atoms,
            conditioning_atoms=conditioning_atoms,
            origin_atom=origin_atom,
            axes_atoms=axes_atoms,
            tfep_logger_dir_path=tfep_logger_dir_path,
            dataloader_kwargs=dataloader_kwargs,
            n_states=2,
            state_names=["state0", "state1"],
            bar_regularizer=bar_reg,
            **kwargs,
        )

        positions_unit = self._potential_energy_func.positions_unit
        ureg = positions_unit._REGISTRY
        dist = float(distance_lower_limit_displacement_angstrom) * ureg.angstrom
        self.hparams["distance_lower_limit_displacement"] = dist.to(positions_unit).magnitude
        self.save_hyperparameters("n_maf_layers", "remove_translation", "remove_rotation")

        self._kwargs = dict(
            hidden_layers=int(maf_hidden_layers),
            weight_norm=bool(maf_weight_norm),
            initialize_identity=bool(maf_initialize_identity),
        )

    def create_universe(self):
        u = super().create_universe()
        _ensure_bonds_for_mixed_coordinates(u)
        return u

    def create_universe_2(self):
        u = super().create_universe_2()
        _ensure_bonds_for_mixed_coordinates(u)
        return u

    def make_dataset(self, state: int, subset_indices: Optional[Sequence[int]] = None):
        if int(state) == 0:
            universe = self.create_universe()
            label = "state0"
        elif int(state) == 1:
            universe = self.create_universe_2()
            label = "state1"
        else:
            raise ValueError(f"Unsupported state index: {state}")

        dataset = _build_conditioned_or_plain_dataset(
            universe=universe,
            shell_enabled=self._shell_enabled,
            shell_center=self._shell_center,
            mapped_atoms=self.hparams.mapped_atoms,
            shell_water_selection=self._shell_water_selection,
            shell_oxygen_names=self._shell_oxygen_names,
            shell_k1=self._shell_k1,
            shell_k2=self._shell_k2,
        )
        return _subset_dataset(dataset, subset_indices, label=label)

    def create_dataset(self):
        return self.make_dataset(0, subset_indices=self._train_indices_0)

    def create_dataset_2(self):
        return self.make_dataset(1, subset_indices=self._train_indices_1)

    def configure_flow(self):
        return MixedMAFMap.configure_flow(self)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self._lr, weight_decay=self._weight_decay)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--state0-dir", required=True)
    p.add_argument("--state1-dir", required=True)
    p.add_argument("--traj0", required=True)
    p.add_argument("--traj1", required=True)
    p.add_argument("--system0-xml", default=None)
    p.add_argument("--system1-xml", default=None)
    p.add_argument("--topology", default=None)

    p.add_argument("--temperature", type=float, required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--outdir", type=str, default="tfep_tmbar_openmm")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--state0-train-indices-file", type=str, default=None,
                   help="Optional file containing the subset of state0 frames to use for training")
    p.add_argument("--state1-train-indices-file", type=str, default=None,
                   help="Optional file containing the subset of state1 frames to use for training")

    p.add_argument(
    "--flow-space",
    choices=["cartesian", "mixedmaf", "shell-equivariant"],
    default="cartesian",)

    p.add_argument("--maf-layers", type=int, default=6)
    p.add_argument("--maf-hidden-layers", type=int, default=2)
    p.add_argument("--maf-weight-norm", action="store_true")
    p.add_argument("--maf-no-weight-norm", action="store_true")
    p.add_argument("--maf-initialize-identity", action="store_true")
    p.add_argument("--maf-no-initialize-identity", action="store_true")

    p.add_argument("--remove-translation", action="store_true")
    p.add_argument("--remove-rotation", action="store_true")
    p.add_argument("--distance-lower-limit-displacement-angstrom", type=float, default=0.3)

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--objective", choices=["kl", "bar", "hybrid"], default="kl")
    p.add_argument("--lambda-bar", type=float, default=1.0)
    p.add_argument("--bar-lambda", type=float, default=0.0,
                   help="Legacy BAR regularizer weight (used only when --objective=kl)")
    p.add_argument("--bar-detach-df", action="store_true")
    p.add_argument("--bar-no-detach-df", action="store_true")
    p.add_argument("--bar-warm-start", action="store_true")
    p.add_argument("--bar-no-warm-start", action="store_true")
    p.add_argument("--bar-max-iter", type=int, default=25)
    p.add_argument("--bar-tol", type=float, default=1e-10)
    p.add_argument("--logJ-penalty", type=float, default=0.0,
                   help="Weight for log|J|^2 Jacobian regularizer. Prevents Jacobian hacking "
                        "under BAR/hybrid objectives by penalising large log-det-Jacobian values. "
                        "Start with 0.01-0.1. Default 0.0 (disabled).")

    p.add_argument("--mapped-atoms", type=str, default=None)
    p.add_argument(
    "--shell-equivariant-solute-selection",
    type=str,
    default="AUTO",
    help=(
        "Solute selection for shell-equivariant water flow. "
        "AUTO means all atoms whose residue name is not water or ion."
    ),
    )
    p.add_argument(
        "--shell-equivariant-water-resnames",
        type=str,
        default="HOH,SOL,WAT,TIP3,TP3M",
        help="Water residue names for shell-equivariant water flow.",
    )
    p.add_argument(
        "--shell-equivariant-water-oxygen-names",
        type=str,
        default="O,OW,OH2",
        help="Water oxygen atom names for shell-equivariant water flow.",
    )
    p.add_argument(
        "--shell-equivariant-ion-resnames",
        type=str,
        default="NA,CL,K,MG,CA,Na,Cl",
        help="Ion residue names excluded from AUTO solute detection.",
    )
    p.add_argument(
        "--shell-equivariant-cutoff-angstrom",
        type=float,
        default=4.5,
        help="Smooth first-shell cutoff for shell-equivariant water flow.",
    )
    p.add_argument(
        "--shell-equivariant-tau-angstrom",
        type=float,
        default=0.3,
        help="Smoothness of shell gate for shell-equivariant water flow.",
    )
    p.add_argument(
        "--shell-equivariant-hidden-dim",
        type=int,
        default=64,
        help="Hidden dimension of the shared water scalar network.",
    )
    p.add_argument(
        "--shell-equivariant-max-displacement-angstrom",
        type=float,
        default=0.05,
        help="Maximum diagnostic water displacement scale.",
    )
    p.add_argument(
        "--shell-equivariant-initial-log-scale",
        type=float,
        default=-1.0,
        help=(
            "Initial log scale multiplying the bounded shell-equivariant displacement. "
            "-1 gives a less frozen diagnostic map than the old -4 value."
        ),
    )
    p.add_argument(
        "--shell-equivariant-top-k",
        type=int,
        default=12,
        help=(
            "Hard-select the K nearest waters per frame for internal dipole/geometry mapping. "
            "Set <=0 to use a smooth sigmoid shell gate for all waters."
        ),
    )
    p.add_argument(
        "--shell-equivariant-max-rotation-radians",
        type=float,
        default=0.35,
        help="Maximum per-water internal H rotation angle in radians before learned/global scaling.",
    )
    p.add_argument(
        "--shell-equivariant-max-internal-log-scale",
        type=float,
        default=0.06,
        help=(
            "Maximum absolute internal log-scale for each O-H vector before learned/global scaling. "
            "The analytic log-Jacobian is 3*s_H1 + 3*s_H2 per water."
        ),
    )
    p.add_argument(
        "--shell-equivariant-solute-maf-layers",
        type=int,
        default=2,
        help="Number of exact masked autoregressive affine layers for the solute block in shell-equivariant mode.",
    )
    p.add_argument(
        "--shell-equivariant-solute-maf-hidden-dim",
        type=int,
        default=128,
        help="Hidden dimension for the solute autoregressive affine MAF block.",
    )
    p.add_argument(
        "--shell-equivariant-solute-maf-max-log-scale",
        type=float,
        default=0.20,
        help="Maximum absolute per-coordinate solute MAF log-scale.",
    )
    p.add_argument(
        "--shell-equivariant-solute-maf-max-shift-angstrom",
        type=float,
        default=0.20,
        help="Maximum absolute per-coordinate solute MAF shift in Angstrom.",
    )


    p.add_argument("--conditioning-atoms", type=str, default=None)
    p.add_argument("--origin-atom", type=str, default=None)
    p.add_argument("--axes-atoms", nargs=2, default=None)

    # --- Shell conditioning (KNN shells) ---
    p.add_argument("--conditioning-shell-k1", type=int, default=0, help="Number of nearest waters in shell 1")
    p.add_argument("--conditioning-shell-k2", type=int, default=0, help="Number of next-nearest waters in shell 2")
    p.add_argument("--conditioning-shell-center", type=str, default=None,
                   help="Selection or indices defining solute center for shell distances (default: mapped-atoms)")
    p.add_argument("--conditioning-shell-water-selection", type=str, default="water",
                   help="MDAnalysis selection for water molecules to consider (default: 'water')")
    p.add_argument("--conditioning-shell-oxygen-names", type=str, default="O,OW",
                   help="Comma-separated oxygen atom names used to compute water distances")
    p.add_argument("--conditioning-shell-condition-on", choices=["molecule", "oxygen"], default="molecule",
                   help="Whether conditioning_atoms include full water molecules or only oxygens in the shell slots")

    p.add_argument("--openmm-platform", choices=["CUDA", "OpenCL", "CPU"], default="CUDA")
    p.add_argument("--openmm-device", type=str, default="0")
    p.add_argument("--openmm-precision", choices=["single", "mixed", "double"], default="mixed")
    p.add_argument("--openmm-cpu-threads", type=int, default=1)

    p.add_argument(
        "--relax-diagnostic",
        choices=["none", "minimize", "langevin", "restrained-langevin"],
        default="none",
        help=(
            "Diagnostic-only target-side relaxation applied after the deterministic "
            "map during held-out analysis. Relaxed coordinates are never used as "
            "TFEP/BAR/TMBAR work values."
        ),
    )
    p.add_argument("--relax-steps", type=int, default=100)
    p.add_argument("--relax-timestep-fs", type=float, default=0.5)
    p.add_argument(
        "--relax-temperature-k",
        type=float,
        default=None,
        help="Temperature for short Langevin diagnostics. Defaults to --temperature.",
    )
    p.add_argument("--relax-friction-ps", type=float, default=10.0)
    p.add_argument(
        "--relax-seed",
        type=int,
        default=None,
        help="Seed for stochastic relaxation diagnostics. Defaults to --seed.",
    )
    p.add_argument(
        "--relax-restraint-k",
        type=float,
        default=1000.0,
        help="Harmonic restraint force constant in kJ/mol/nm^2 for restrained-langevin.",
    )
    p.add_argument("--relax-restraint-selection", type=str, default="mapped")
    p.add_argument("--relax-output-dir", type=str, default="relaxation_diagnostics")
    p.add_argument(
        "--relax-every-n-frames",
        type=int,
        default=1,
        help="Relax every Nth mapped validation frame in each direction.",
    )
    p.add_argument(
        "--relax-max-frames",
        type=int,
        default=100,
        help="Maximum relaxed validation frames per direction.",
    )
    p.add_argument("--relax-save-trajectories", action="store_true")
    p.add_argument("--relax-backend", choices=["openmm", "gromacs", "auto"], default="auto")
    p.add_argument(
        "--relax-strict",
        action="store_true",
        help="Abort on the first failed relaxation frame instead of recording the failure.",
    )

    # Default is fixed System box + ignore per-frame dims.
    p.add_argument(
        "--no-fixed-box",
        action="store_true",
        help="Do NOT set System default box from CRYST1 / do NOT ignore per-frame dimensions (may crash).",
    )

    p.add_argument("--torch-accelerator", choices=["cpu", "gpu"], default="cpu")
    p.add_argument("--torch-devices", type=int, default=1)

    add_stochastic_tfep_args(p)

    return p


@dataclass
class TrainingArtifacts:
    args: argparse.Namespace
    outdir: Path
    trainer: L.Trainer
    model: L.LightningModule
    state0_dir: Path
    state1_dir: Path
    traj0_path: Path
    traj1_path: Path
    topo_path: Path
    system0_xml_path: Path
    system1_xml_path: Path


def _namespace_to_jsonable(args: argparse.Namespace) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            out[key] = str(value)
        else:
            out[key] = value
    return out


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _safe_sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def _eval_potential_direct_for_model(model, state: int, positions: torch.Tensor, dimensions: Optional[torch.Tensor]):
    """Direct per-state potential evaluation used to avoid MultiStatePotential crashes."""
    if hasattr(model, "_wrap_positions_for_box_eval"):
        positions = model._wrap_positions_for_box_eval(positions, dimensions)

    pot = model._potential_energy_func
    if hasattr(pot, "potentials"):
        pot_state = pot.potentials[int(state)]
    elif isinstance(pot, (list, tuple)):
        pot_state = pot[int(state)]
    elif isinstance(pot, dict):
        pot_state = pot[int(state)]
    else:
        return TMBARMapBase._eval_potential(model, state, positions, dimensions)

    if isinstance(pot_state, IsolatedOpenMMPotential) and not bool(positions.requires_grad):
        return pot_state.energy_no_grad(positions, dimensions)

    if dimensions is None:
        return pot_state(positions)
    try:
        return pot_state(positions, dimensions)
    except TypeError:
        return pot_state(positions)


def bar_deltaf(w_forward: np.ndarray, w_reverse: np.ndarray) -> Tuple[float, float]:
    w_forward = np.asarray(w_forward, dtype=np.float64)
    w_reverse = np.asarray(w_reverse, dtype=np.float64)
    w_forward = w_forward[np.isfinite(w_forward)]
    w_reverse = w_reverse[np.isfinite(w_reverse)]
    if len(w_forward) < 10 or len(w_reverse) < 10:
        return float("nan"), float("nan")

    if pymbar is not None:
        try:
            if hasattr(pymbar, "other_estimators") and hasattr(pymbar.other_estimators, "bar"):
                out = pymbar.other_estimators.bar(w_forward, w_reverse, compute_uncertainty=True)
                return float(out["Delta_f"]), float(out.get("dDelta_f", np.nan))
        except Exception:
            pass

        try:
            if hasattr(pymbar, "BAR"):
                out = pymbar.BAR(w_forward, w_reverse, return_dict=True)
                if isinstance(out, dict):
                    return float(out["Delta_f"]), float(out.get("dDelta_f", np.nan))
                df, ddf = out
                return float(df), float(ddf)
        except Exception:
            pass

    def f(df):
        return np.mean(_safe_sigmoid(df - w_forward)) - np.mean(_safe_sigmoid(-(w_reverse + df)))

    mid0 = 0.5 * (float(np.median(w_forward)) - float(np.median(w_reverse)))
    lo, hi = mid0 - 50.0, mid0 + 50.0
    flo, fhi = f(lo), f(hi)
    n_expand = 0
    while flo * fhi > 0 and n_expand < 20:
        lo -= 50.0
        hi += 50.0
        flo, fhi = f(lo), f(hi)
        n_expand += 1
    if flo * fhi > 0:
        return float("nan"), float("nan")

    for _ in range(200):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if abs(fm) < 1e-12 or abs(hi - lo) < 1e-10:
            return float(mid), float("nan")
        if flo * fm <= 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm

    return float(0.5 * (lo + hi)), float("nan")


def overlap_integral(x: np.ndarray, y: np.ndarray, bins: int = 200) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return float("nan")

    lo = min(float(np.min(x)), float(np.min(y)))
    hi = max(float(np.max(x)), float(np.max(y)))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return float("nan")

    edges = np.linspace(lo, hi, int(bins) + 1)
    hx, _ = np.histogram(x, bins=edges, density=True)
    hy, _ = np.histogram(y, bins=edges, density=True)
    return float(np.sum(np.minimum(hx, hy) * np.diff(edges)))


def bar_consistent_overlap_integral(w_forward: np.ndarray, w_reverse: np.ndarray, bins: int = 200) -> float:
    """BAR/Crooks overlap on the common axis: forward work vs sign-flipped reverse work."""
    return overlap_integral(np.asarray(w_forward, dtype=np.float64), -np.asarray(w_reverse, dtype=np.float64), bins=bins)


def _move_batch_to_device(batch, device):
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: _move_batch_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, list):
        return [_move_batch_to_device(v, device) for v in batch]
    if isinstance(batch, tuple):
        return tuple(_move_batch_to_device(v, device) for v in batch)
    return batch


def _to_numpy(x) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def summarize_work_pair(w_forward: np.ndarray, w_reverse: np.ndarray) -> Dict[str, float]:
    df, ddf = bar_deltaf(w_forward, w_reverse)
    bar_overlap = bar_consistent_overlap_integral(w_forward, w_reverse, bins=300)
    direct_overlap = overlap_integral(w_forward, w_reverse, bins=300)
    return {
        "deltaf": float(df),
        "sigma": float(ddf),
        "overlap": float(bar_overlap),
        "bar_overlap": float(bar_overlap),
        "direct_overlap": float(direct_overlap),
        "n_forward": int(np.isfinite(w_forward).sum()),
        "n_reverse": int(np.isfinite(w_reverse).sum()),
        "w_forward_mean": float(np.nanmean(w_forward)),
        "w_forward_std": float(np.nanstd(w_forward)),
        "w_reverse_mean": float(np.nanmean(w_reverse)),
        "w_reverse_std": float(np.nanstd(w_reverse)),
    }


def _collect_direction_works(model, loader, *, state_from: int, state_to: int, direction: str, include_tfep: bool):
    records = {
        "dataset_sample_index": [],
        "trajectory_sample_index": [],
        "raw_work": [],
        "u_from": [],
        "u_to_raw": [],
    }
    if include_tfep:
        records.update({
            "tfep_work": [],
            "u_to_mapped": [],
            "log_det_J": [],
        })

    device = model.device
    for batch in loader:
        batch = _move_batch_to_device(batch, device)
        dims = batch.get("dimensions", None)
        u_from = model._eval_potential(state_from, batch["positions"], dims) / model._kT
        u_to_raw = model._eval_potential(state_to, batch["positions"], dims) / model._kT

        records["dataset_sample_index"].append(_to_numpy(batch["dataset_sample_index"]))
        records["trajectory_sample_index"].append(_to_numpy(batch["trajectory_sample_index"]))
        records["u_from"].append(_to_numpy(u_from))
        records["u_to_raw"].append(_to_numpy(u_to_raw))
        records["raw_work"].append(_to_numpy(u_to_raw - u_from))

        if include_tfep:
            if direction == "forward":
                result = model.forward(batch)
            elif direction == "inverse":
                result = model.inverse(batch)
            else:
                raise ValueError(f"Unsupported direction: {direction}")

            u_to_mapped = model._eval_potential(state_to, result["positions"], dims) / model._kT
            tfep_work = (u_to_mapped - result["log_det_J"]) - u_from
            records["u_to_mapped"].append(_to_numpy(u_to_mapped))
            records["log_det_J"].append(_to_numpy(result["log_det_J"]))
            records["tfep_work"].append(_to_numpy(tfep_work))

    return {
        key: np.concatenate([np.asarray(chunk) for chunk in chunks]) if len(chunks) > 0 else np.empty(0, dtype=float)
        for key, chunks in records.items()
    }


@torch.no_grad()
def evaluate_bidirectional_map(
    model,
    *,
    subset_indices_0: Optional[Sequence[int]] = None,
    subset_indices_1: Optional[Sequence[int]] = None,
    batch_size: Optional[int] = None,
    include_tfep: bool = True,
) -> Dict[str, Any]:
    if include_tfep and (not hasattr(model, "_flow") or model._flow is None):
        model.setup("fit")

    model.eval()
    eval_batch_size = int(batch_size) if batch_size is not None else int(model.hparams.batch_size)
    dataset0 = model.make_dataset(0, subset_indices=subset_indices_0)
    dataset1 = model.make_dataset(1, subset_indices=subset_indices_1)
    loader0 = torch.utils.data.DataLoader(dataset0, batch_size=eval_batch_size, shuffle=False)
    loader1 = torch.utils.data.DataLoader(dataset1, batch_size=eval_batch_size, shuffle=False)

    state0_state1 = _collect_direction_works(
        model,
        loader0,
        state_from=0,
        state_to=1,
        direction="forward",
        include_tfep=include_tfep,
    )
    state1_state0 = _collect_direction_works(
        model,
        loader1,
        state_from=1,
        state_to=0,
        direction="inverse",
        include_tfep=include_tfep,
    )

    out: Dict[str, Any] = {
        "state0_state1": state0_state1,
        "state1_state0": state1_state0,
        "raw": summarize_work_pair(state0_state1["raw_work"], state1_state0["raw_work"]),
    }
    if include_tfep:
        out["tfep"] = summarize_work_pair(state0_state1["tfep_work"], state1_state0["tfep_work"])
    return out


def _relaxation_config_from_args(args: argparse.Namespace, analysis_dir: Path) -> RelaxationDiagnosticConfig:
    output_dir = Path(str(getattr(args, "relax_output_dir", "relaxation_diagnostics"))).expanduser()
    if not output_dir.is_absolute():
        output_dir = Path(analysis_dir) / output_dir
    return RelaxationDiagnosticConfig(
        mode=str(getattr(args, "relax_diagnostic", "none")),
        backend=str(getattr(args, "relax_backend", "auto")),
        steps=int(getattr(args, "relax_steps", 100)),
        timestep_fs=float(getattr(args, "relax_timestep_fs", 0.5)),
        temperature_k=float(
            getattr(args, "relax_temperature_k", None)
            if getattr(args, "relax_temperature_k", None) is not None
            else getattr(args, "temperature", 298.15)
        ),
        friction_ps=float(getattr(args, "relax_friction_ps", 10.0)),
        seed=int(
            getattr(args, "relax_seed", None)
            if getattr(args, "relax_seed", None) is not None
            else getattr(args, "seed", 123)
        ),
        restraint_k=float(getattr(args, "relax_restraint_k", 1000.0)),
        restraint_selection=str(getattr(args, "relax_restraint_selection", "mapped")),
        output_dir=str(output_dir),
        every_n_frames=max(1, int(getattr(args, "relax_every_n_frames", 1))),
        max_frames=max(0, int(getattr(args, "relax_max_frames", 100))),
        save_trajectories=bool(getattr(args, "relax_save_trajectories", False)),
        strict=bool(getattr(args, "relax_strict", False)),
        platform_name="CPU",
        cpu_threads=int(getattr(args, "openmm_cpu_threads", 1)),
    )


def _target_topology_for_state(artifacts: TrainingArtifacts, state: int) -> Path:
    state_dir = artifacts.state0_dir if int(state) == 0 else artifacts.state1_dir
    candidate = Path(state_dir) / "system" / "start.pdb"
    if candidate.is_file():
        return candidate.resolve()
    return Path(artifacts.topo_path).resolve()


def _target_system_for_state(artifacts: TrainingArtifacts, state: int) -> Path:
    return Path(artifacts.system0_xml_path if int(state) == 0 else artifacts.system1_xml_path).resolve()


def _mapped_atom_indices_for_model(model) -> Optional[np.ndarray]:
    try:
        mapped = model.get_mapped_indices(idx_type="atom", remove_fixed=False)
        return _to_numpy(mapped).astype(int).reshape(-1)
    except Exception:
        return None


@torch.no_grad()
def _collect_mapped_relaxation_frames(
    model,
    loader,
    *,
    direction_label: str,
    source_state: int,
    target_state: int,
    direction: str,
    eval_direction: Dict[str, np.ndarray],
    config: RelaxationDiagnosticConfig,
) -> List[Dict[str, Any]]:
    frames: List[Dict[str, Any]] = []
    if config.max_frames <= 0:
        return frames

    device = model.device
    candidate_i = 0
    for batch in loader:
        batch = _move_batch_to_device(batch, device)
        if direction == "forward":
            result = model.forward(batch)
        elif direction == "inverse":
            result = model.inverse(batch)
        else:
            raise ValueError(f"Unsupported direction: {direction}")

        mapped_positions = _to_numpy(result["positions"])
        log_det_j = _to_numpy(result["log_det_J"])
        dataset_idx = _to_numpy(batch["dataset_sample_index"])
        traj_idx = _to_numpy(batch["trajectory_sample_index"])
        dims = _to_numpy(batch["dimensions"]) if "dimensions" in batch else None
        tfep_work = np.asarray(eval_direction.get("tfep_work", []), dtype=float)

        for row_i in range(mapped_positions.shape[0]):
            if candidate_i % config.every_n_frames == 0:
                if len(frames) >= config.max_frames:
                    return frames
                frame = {
                    "frame_index": int(candidate_i),
                    "direction": direction_label,
                    "source_state": int(source_state),
                    "target_state": int(target_state),
                    "positions_angstrom": np.asarray(mapped_positions[row_i], dtype=float).reshape(-1, 3),
                    "dataset_sample_index": int(np.asarray(dataset_idx).reshape(-1)[row_i]),
                    "trajectory_sample_index": int(np.asarray(traj_idx).reshape(-1)[row_i]),
                    "log_det_J": float(np.asarray(log_det_j).reshape(-1)[row_i]),
                    "deterministic_tfep_work": (
                        float(tfep_work[candidate_i])
                        if candidate_i < len(tfep_work)
                        else float("nan")
                    ),
                }
                if dims is not None:
                    frame["dimensions_angstrom"] = np.asarray(dims[row_i], dtype=float)
                frames.append(frame)
            candidate_i += 1

    return frames


def run_relaxation_diagnostics_for_model(
    artifacts: TrainingArtifacts,
    args: argparse.Namespace,
    *,
    analysis_dir: Path,
    subset_indices_0: Optional[Sequence[int]],
    subset_indices_1: Optional[Sequence[int]],
    batch_size: int,
    eval_result: Dict[str, Any],
    water_shell_mapping_path: Optional[Union[str, Path]] = None,
) -> Optional[Dict[str, Any]]:
    """Run diagnostic relaxation on held-out mapped frames.

    This function intentionally runs after deterministic validation work arrays
    have already been computed. It maps a small validation subset again only to
    obtain coordinates for diagnostic relaxation. The relaxed coordinates and
    energies are never fed back into BAR/TMBAR.
    """

    config = _relaxation_config_from_args(args, analysis_dir)
    if not config.enabled:
        return None

    if not hasattr(artifacts.model, "_flow") or artifacts.model._flow is None:
        artifacts.model.setup("fit")
    artifacts.model.eval()

    dataset0 = artifacts.model.make_dataset(0, subset_indices=subset_indices_0)
    dataset1 = artifacts.model.make_dataset(1, subset_indices=subset_indices_1)
    loader0 = torch.utils.data.DataLoader(dataset0, batch_size=int(batch_size), shuffle=False)
    loader1 = torch.utils.data.DataLoader(dataset1, batch_size=int(batch_size), shuffle=False)

    frames0 = _collect_mapped_relaxation_frames(
        artifacts.model,
        loader0,
        direction_label="state0_to_state1",
        source_state=0,
        target_state=1,
        direction="forward",
        eval_direction=eval_result["state0_state1"],
        config=config,
    )
    frames1 = _collect_mapped_relaxation_frames(
        artifacts.model,
        loader1,
        direction_label="state1_to_state0",
        source_state=1,
        target_state=0,
        direction="inverse",
        eval_direction=eval_result["state1_state0"],
        config=config,
    )

    target_system_xml_paths = {
        "state0_to_state1": _target_system_for_state(artifacts, 1),
        "state1_to_state0": _target_system_for_state(artifacts, 0),
    }
    target_topology_pdb_paths = {
        "state0_to_state1": _target_topology_for_state(artifacts, 1),
        "state1_to_state0": _target_topology_for_state(artifacts, 0),
    }
    output_dir = Path(config.output_dir).expanduser().resolve()
    mapped_indices = _mapped_atom_indices_for_model(artifacts.model)

    result = run_short_relaxation_diagnostic(
        mapped_frames={
            "state0_to_state1": frames0,
            "state1_to_state0": frames1,
        },
        target_system_xml_paths=target_system_xml_paths,
        target_topology_pdb_paths=target_topology_pdb_paths,
        config=config,
        mapped_atom_indices=mapped_indices,
        solute_selection=str(getattr(args, "relax_restraint_selection", "mapped")),
        water_shell_mapping_path=water_shell_mapping_path,
    )
    write_relaxation_outputs(
        result,
        output_dir=output_dir,
        config=config,
        metadata={
            "analysis_dir": str(Path(analysis_dir).resolve()),
            "frames_state0_to_state1_requested": len(frames0),
            "frames_state1_to_state0_requested": len(frames1),
            "target_system_xml_paths": {k: str(v) for k, v in target_system_xml_paths.items()},
            "target_topology_pdb_paths": {k: str(v) for k, v in target_topology_pdb_paths.items()},
            "mapped_atom_indices_count": int(0 if mapped_indices is None else len(mapped_indices)),
            "water_shell_mapping_path": str(water_shell_mapping_path) if water_shell_mapping_path else None,
        },
    )

    records = result.get("records", [])
    success_count = sum(1 for record in records if bool(getattr(record, "success", False)))
    failure_count = len(records) - success_count
    return {
        "enabled": True,
        "mode": config.mode,
        "backend": config.backend,
        "output_dir": str(output_dir),
        "records": int(len(records)),
        "success_count": int(success_count),
        "failure_count": int(failure_count),
        "statistical_note": (
            "Relaxed coordinates are diagnostic only and are not used in deterministic "
            "TFEP/BAR/TMBAR work arrays."
        ),
    }


def _stochastic_config_from_args(args: argparse.Namespace, analysis_dir: Path) -> MolecularStochasticConfig:
    output_dir = Path(str(getattr(args, "snf_output_dir", "stochastic_tfep_outputs"))).expanduser()
    if not output_dir.is_absolute():
        output_dir = Path(analysis_dir) / output_dir
    return MolecularStochasticConfig(
        estimator=str(getattr(args, "estimator", "deterministic-tfep")),
        kernel=str(getattr(args, "snf_kernel", "gaussian-rw")),
        noise_sigma=getattr(args, "snf_noise_sigma", None),
        step_size=getattr(args, "snf_step_size", None),
        diffusion=float(getattr(args, "snf_diffusion", 1.0)),
        num_blocks=int(getattr(args, "snf_num_blocks", 1)),
        steps_per_block=int(getattr(args, "snf_steps_per_block", 1)),
        gradient_policy=str(getattr(args, "snf_gradient_policy", "stop-gradient")),
        seed=int(getattr(args, "snf_seed", getattr(args, "seed", 123))),
        max_frames=getattr(args, "snf_max_frames", None),
        every_n_frames=max(1, int(getattr(args, "snf_every_n_frames", 1))),
        selected_atoms=getattr(args, "snf_selected_atoms", None),
        apply_to=str(getattr(args, "snf_apply_to", "selected")),
        output_dir=str(output_dir),
        strict=bool(getattr(args, "snf_strict", False)),
        allow_molecular_ula=bool(getattr(args, "snf_allow_molecular_ula", False)),
        allow_constrained_cartesian_ula=bool(getattr(args, "snf_allow_constrained_cartesian_ula", False)),
    )


def _stochastic_training_config_from_args(args: argparse.Namespace) -> Optional[StochasticTrainingConfig]:
    if not bool(getattr(args, "snf_train", False)):
        return None
    return StochasticTrainingConfig(
        enabled=True,
        kernel=str(getattr(args, "snf_kernel", "gaussian-rw")),
        noise_sigma=getattr(args, "snf_noise_sigma", None),
        step_size=getattr(args, "snf_step_size", None),
        diffusion=float(getattr(args, "snf_diffusion", 1.0)),
        num_blocks=int(getattr(args, "snf_num_blocks", 1)),
        steps_per_block=int(getattr(args, "snf_steps_per_block", 1)),
        train_mc_samples=int(getattr(args, "snf_train_mc_samples", 1)),
        gradient_policy=str(getattr(args, "snf_gradient_policy", "stop-gradient")),
        apply_to=str(getattr(args, "snf_apply_to", "selected")),
        allow_constrained_cartesian_ula=bool(getattr(args, "snf_allow_constrained_cartesian_ula", False)),
        allow_molecular_ula=bool(getattr(args, "snf_allow_molecular_ula", False)),
        strict=bool(getattr(args, "snf_strict", False)),
    )


def _write_snf_path_csv(path: Path, records: Dict[str, np.ndarray]) -> None:
    keys = [
        "dataset_sample_index",
        "trajectory_sample_index",
        "snf_work",
        "u_from",
        "u_to_snf",
        "u_to_mapped",
        "log_det_J",
        "sum_logq_forward",
        "sum_logq_reverse",
        "path_log_weight",
        "mapped_to_snf_rmsd_angstrom",
        "mapped_to_snf_max_disp_angstrom",
    ]
    n_rows = int(len(records.get("snf_work", [])))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row_i in range(n_rows):
            row = {}
            for key in keys:
                values = np.asarray(records.get(key, []))
                row[key] = values[row_i].item() if row_i < len(values) else ""
            writer.writerow(row)


def run_stochastic_path_evaluation_for_model(
    artifacts: TrainingArtifacts,
    args: argparse.Namespace,
    *,
    analysis_dir: Path,
    subset_indices_0: Optional[Sequence[int]],
    subset_indices_1: Optional[Sequence[int]],
    batch_size: int,
    water_shell_mapping_path: Optional[Union[str, Path]] = None,
) -> Optional[Dict[str, Any]]:
    """Run experimental path-weighted stochastic TFEP on mapped validation frames.

    The deterministic map still supplies the exact ``log_det_J`` term. The
    stochastic kernel is applied after the deterministic map and the resulting
    path work includes ``+logq_forward-logq_reverse``. These arrays are saved
    separately from deterministic TFEP/BAR arrays.
    """

    config = _stochastic_config_from_args(args, analysis_dir)
    if not config.enabled:
        return None

    if not hasattr(artifacts.model, "_flow") or artifacts.model._flow is None:
        artifacts.model.setup("fit")
    artifacts.model.eval()

    mapped_indices = _mapped_atom_indices_for_model(artifacts.model)
    selected_flat_indices = resolve_snf_flat_indices(
        model=artifacts.model,
        topology_path=artifacts.topo_path,
        config=config,
        mapped_atom_indices=mapped_indices,
        water_shell_mapping_path=water_shell_mapping_path,
    )

    dataset0 = artifacts.model.make_dataset(0, subset_indices=subset_indices_0)
    dataset1 = artifacts.model.make_dataset(1, subset_indices=subset_indices_1)
    loader0 = torch.utils.data.DataLoader(dataset0, batch_size=int(batch_size), shuffle=False)
    loader1 = torch.utils.data.DataLoader(dataset1, batch_size=int(batch_size), shuffle=False)

    forward = None
    reverse = None
    direction = str(getattr(args, "snf_direction", "bidirectional"))
    if direction in ("forward", "bidirectional"):
        forward = evaluate_stochastic_direction(
            model=artifacts.model,
            loader=loader0,
            state_from=0,
            state_to=1,
            direction="forward",
            config=config,
            selected_flat_indices=selected_flat_indices,
            move_batch_to_device=_move_batch_to_device,
            to_numpy=_to_numpy,
            seed_offset=31000,
        )
    if direction in ("reverse", "bidirectional"):
        reverse = evaluate_stochastic_direction(
            model=artifacts.model,
            loader=loader1,
            state_from=1,
            state_to=0,
            direction="inverse",
            config=config,
            selected_flat_indices=selected_flat_indices,
            move_batch_to_device=_move_batch_to_device,
            to_numpy=_to_numpy,
            seed_offset=41000,
        )

    output_dir = Path(config.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = stochastic_metadata_from_config(config, selected_flat_indices)
    metadata.update(
        {
            "analysis_dir": str(Path(analysis_dir).resolve()),
            "topology_path": str(Path(artifacts.topo_path).resolve()),
            "state0_dir": str(Path(artifacts.state0_dir).resolve()),
            "state1_dir": str(Path(artifacts.state1_dir).resolve()),
            "mapped_atom_indices_count": int(0 if mapped_indices is None else len(mapped_indices)),
            "water_shell_mapping_path": str(water_shell_mapping_path) if water_shell_mapping_path else None,
            "direction": direction,
        }
    )

    summary: Dict[str, Any] = {"enabled": True, "metadata": metadata, "output_dir": str(output_dir)}
    if forward is not None and reverse is not None:
        summary["summary"] = summarize_work_pair(forward["snf_work"], reverse["snf_work"])
    elif forward is not None:
        summary["summary"] = {"n_forward": int(len(forward["snf_work"])), "n_reverse": 0}
    elif reverse is not None:
        summary["summary"] = {"n_forward": 0, "n_reverse": int(len(reverse["snf_work"]))}
    else:
        summary["summary"] = {"n_forward": 0, "n_reverse": 0}

    readme = (
        "# Experimental stochastic path-weighted TFEP\n\n"
        "These work values include the deterministic map log-Jacobian and the "
        "stochastic transition-density terms. They are not deterministic TFEP "
        "work arrays and must not be mixed into deterministic BAR/TMBAR outputs.\n"
    )
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    (output_dir / "metadata.json").write_text(json.dumps(_json_ready(summary), indent=2), encoding="utf-8")

    npz_payload: Dict[str, np.ndarray] = {}
    if forward is not None:
        _write_snf_path_csv(output_dir / "paths_forward.csv", forward)
        for key, value in forward.items():
            npz_payload[f"forward_{key}"] = np.asarray(value)
    if reverse is not None:
        _write_snf_path_csv(output_dir / "paths_reverse.csv", reverse)
        for key, value in reverse.items():
            npz_payload[f"reverse_{key}"] = np.asarray(value)
    if npz_payload:
        np.savez_compressed(output_dir / "stochastic_work_arrays.npz", **npz_payload)

    return {
        "enabled": True,
        "output_dir": str(output_dir),
        "metadata": metadata,
        "summary": summary.get("summary", {}),
        "state0_state1": forward,
        "state1_state0": reverse,
    }


def create_model_from_args(
    args: argparse.Namespace,
    *,
    outdir_override: Optional[Union[str, Path]] = None,
    train_indices_0: Optional[Sequence[int]] = None,
    train_indices_1: Optional[Sequence[int]] = None,
):
    outdir = Path(outdir_override) if outdir_override is not None else Path(args.outdir)
    _require_openmm()

    maf_weight_norm = True
    if args.maf_no_weight_norm:
        maf_weight_norm = False
    if args.maf_weight_norm:
        maf_weight_norm = True

    maf_initialize_identity = True
    if args.maf_no_initialize_identity:
        maf_initialize_identity = False
    if args.maf_initialize_identity:
        maf_initialize_identity = True

    bar_detach_df = True
    if args.bar_no_detach_df:
        bar_detach_df = False
    if args.bar_detach_df:
        bar_detach_df = True

    bar_warm_start = True
    if args.bar_no_warm_start:
        bar_warm_start = False
    if args.bar_warm_start:
        bar_warm_start = True

    objective = _normalize_objective(args.snf_train_objective if getattr(args, "snf_train_objective", None) else args.objective)
    lambda_bar = float(args.lambda_bar)
    legacy_bar_lambda = float(args.bar_lambda)
    legacy_bar_active = bool(objective == "kl" and legacy_bar_lambda > 0.0)
    logJ_penalty_weight = float(getattr(args, 'logJ_penalty', 0.0))
    stochastic_training_config = _stochastic_training_config_from_args(args)

    state0_dir = Path(args.state0_dir).resolve()
    state1_dir = Path(args.state1_dir).resolve()

    traj0_path, system0_xml_path, topo_path = _resolve_state_inputs(
        state0_dir, args.traj0, args.system0_xml, args.topology
    )
    traj1_path, system1_xml_path, _ = _resolve_state_inputs(
        state1_dir, args.traj1, args.system1_xml, args.topology
    )

    k1 = int(args.conditioning_shell_k1)
    k2 = int(args.conditioning_shell_k2)
    k_total = k1 + k2
    shell_enabled = k_total > 0

    oxygen_names = _parse_csv_names(args.conditioning_shell_oxygen_names)
    shell_center = args.conditioning_shell_center
    if shell_center is None:
        shell_center = args.mapped_atoms

    if train_indices_0 is None:
        train_indices_0 = _load_index_array(args.state0_train_indices_file)
    if train_indices_1 is None:
        train_indices_1 = _load_index_array(args.state1_train_indices_file)

    shell_equiv_spec = None

    if str(args.flow_space) == "shell-equivariant":
        shell_equiv_solute_selection = getattr(
            args,
            "shell_equivariant_solute_selection",
            "AUTO",
        )

        # If the user did not explicitly pass a shell-equivariant solute selection,
        # reuse --mapped-atoms when available, otherwise AUTO.
        if (
            shell_equiv_solute_selection == "AUTO"
            and getattr(args, "mapped_atoms", None) is not None
        ):
            shell_equiv_solute_selection = str(args.mapped_atoms)

        shell_equiv_spec = _build_shell_equivariant_index_spec(
            topology_path=topo_path,
            solute_selection=shell_equiv_solute_selection,
            water_resnames_text=getattr(
                args,
                "shell_equivariant_water_resnames",
                "HOH,SOL,WAT,TIP3,TP3M",
            ),
            water_oxygen_names_text=getattr(
                args,
                "shell_equivariant_water_oxygen_names",
                "O,OW,OH2",
            ),
            ion_resnames_text=getattr(
                args,
                "shell_equivariant_ion_resnames",
                "NA,CL,K,MG,CA,Na,Cl",
            ),
        )

        # Important:
        # The existing TFEP machinery scatters transformed mapped coordinates
        # back into the full system. Therefore the mapped subset must include
        # solute + all water atoms for this diagnostic flow.
        args.mapped_atoms = (
            "index "
            + " ".join(str(int(i)) for i in shell_equiv_spec["mapped_global_indices"])
        )

        print("[shell-equivariant] enabled")
        print(f"[shell-equivariant] n_total_atoms   = {shell_equiv_spec['n_atoms_total']}")
        print(f"[shell-equivariant] n_mapped_atoms  = {shell_equiv_spec['n_mapped_atoms']}")
        print(f"[shell-equivariant] n_solute_atoms  = {shell_equiv_spec['n_solute_atoms']}")
        print(f"[shell-equivariant] n_waters        = {shell_equiv_spec['n_waters']}")
        print(f"[shell-equivariant] internal top_k  = {getattr(args, 'shell_equivariant_top_k', 12)}")
        print(f"[shell-equivariant] exact logJ      = solute MAF + internal O-H coupling analytic log determinants")
        print(f"[shell-equivariant] mapped_atoms set to solute + all water atoms")
        print(f"[shell-equivariant] full map        = solute exact MAF -> internal shell-water exact flow")
        print(f"[shell-equivariant] solute MAF      = layers={int(args.shell_equivariant_solute_maf_layers)}, hidden={int(args.shell_equivariant_solute_maf_hidden_dim)}")

    mapped_atoms = _as_indices_or_selection(args.mapped_atoms)
    origin_atom = _as_indices_or_selection(args.origin_atom)
    axes_atoms = None
    if args.axes_atoms is not None:
        axes_atoms = [_as_indices_or_selection(args.axes_atoms[0]),
                      _as_indices_or_selection(args.axes_atoms[1])]

    if shell_enabled:
        conditioning_atoms = compute_shell_slot_conditioning_indices(
            topo_path,
            water_selection=args.conditioning_shell_water_selection,
            oxygen_names=oxygen_names,
            k_total=k_total,
            condition_on=args.conditioning_shell_condition_on,
        )
        print(f"[shell-conditioning] enabled: k1={k1}, k2={k2}, total={k_total}")
        print(f"[shell-conditioning] water_selection={args.conditioning_shell_water_selection!r}, oxygen_names={oxygen_names}")
        print(f"[shell-conditioning] center={shell_center!r}")
        print(f"[shell-conditioning] conditioning_atoms count = {len(conditioning_atoms)} (slots = first {k_total} residues)")
    else:
        conditioning_atoms = _as_indices_or_selection(args.conditioning_atoms)

    system0 = _load_openmm_system(system0_xml_path)
    system1 = _load_openmm_system(system1_xml_path)
    platform = _configure_openmm_platform(
        args.openmm_platform, args.openmm_device, args.openmm_precision, int(args.openmm_cpu_threads)
    )
    wrap_blocks = compute_residue_wrap_blocks(topo_path)

    ureg = pint.UnitRegistry()
    positions_unit = ureg.angstrom
    energy_unit = ureg("kJ/mol")
    temperature = float(args.temperature) * ureg.kelvin

    if not args.no_fixed_box:
        set_system_default_box_from_cryst1(system0, topo_path)
        set_system_default_box_from_cryst1(system1, topo_path)
        aA, bA, cA, alpha, beta, gamma = _read_cryst1_dims(topo_path)
        print(f"[box] System default box from CRYST1 (Å,deg): [{aA}, {bA}, {cA}, {alpha}, {beta}, {gamma}]")

    potential_cls = OpenMMPotential
    if args.no_fixed_box:
        potential_cls = IsolatedOpenMMPotential

    pot0 = potential_cls(
        system=system0,
        platform=platform,
        positions_unit=positions_unit,
        energy_unit=energy_unit,
        system_name="state0",
        precompute_gradient=True,
    )
    pot1 = potential_cls(
        system=system1,
        platform=platform,
        positions_unit=positions_unit,
        energy_unit=energy_unit,
        system_name="state1",
        precompute_gradient=True,
    )

    if not args.no_fixed_box:
        pot0 = NoCellWrapper(pot0)
        pot1 = NoCellWrapper(pot1)
        print("[box] ignoring per-frame dimensions; using System default box")
    else:
        if len(wrap_blocks) > 0:
            print(f"[box] no-fixed-box enabled; whole-residue wrapping will be applied for {len(wrap_blocks)} residue block(s) before OpenMM evaluation")
        print("[box] no-fixed-box enabled; each state will be evaluated in its own OpenMM worker process")

    if objective in ("bar", "hybrid") and legacy_bar_lambda > 0.0:
        print("[objective] NOTE: --bar-lambda is a legacy KL regularizer and is ignored for objective='bar'/'hybrid'.")
    print(f"[objective] {_objective_mode_description(objective, lambda_bar=lambda_bar, legacy_bar_lambda=legacy_bar_lambda, logJ_penalty_weight=logJ_penalty_weight)}")
    print(
        "[objective] settings: "
        f"objective={objective}, lambda_bar={lambda_bar}, "
        f"legacy_bar_lambda={legacy_bar_lambda}, legacy_bar_active={legacy_bar_active}, "
        f"bar_detach_df={bar_detach_df}, bar_warm_start={bar_warm_start}, "
        f"logJ_penalty_weight={logJ_penalty_weight}"
    )
    if stochastic_training_config is not None:
        print(
            "[snf-train] enabled: "
            f"kernel={stochastic_training_config.kernel}, steps={stochastic_training_config.n_stochastic_steps}, "
            f"mc_samples={stochastic_training_config.train_mc_samples}, "
            f"apply_to={stochastic_training_config.apply_to}, "
            f"gradient_policy={stochastic_training_config.gradient_policy}, "
            f"allow_constrained_cartesian_ula={stochastic_training_config.allow_constrained_cartesian_ula}"
        )

    if args.flow_space == "cartesian":
        model = SmallMolBidirectionalTMBARMapCartesian(
            pot0, pot1,
            topology_file_path=str(topo_path),
            coordinates_file_path=str(traj0_path),
            coordinates_file_path_2=str(traj1_path),
            temperature=temperature,
            batch_size=int(args.batch_size),
            mapped_atoms=mapped_atoms,
            conditioning_atoms=conditioning_atoms,
            origin_atom=origin_atom,
            axes_atoms=axes_atoms,
            tfep_logger_dir_path=str(outdir / "tfep_logs"),
            n_maf_layers=int(args.maf_layers),
            maf_hidden_layers=int(args.maf_hidden_layers),
            maf_weight_norm=maf_weight_norm,
            maf_initialize_identity=maf_initialize_identity,
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            objective=objective,
            lambda_bar=lambda_bar,
            bar_detach_df=bool(bar_detach_df),
            bar_warm_start=bool(bar_warm_start),
            bar_max_iter=int(args.bar_max_iter),
            bar_tol=float(args.bar_tol),
            bar_lambda=legacy_bar_lambda,
            logJ_penalty_weight=logJ_penalty_weight,
            stochastic_training_config=stochastic_training_config,
            shell_k1=k1,
            shell_k2=k2,
            shell_center=_as_indices_or_selection(shell_center) if shell_enabled else None,
            shell_water_selection=args.conditioning_shell_water_selection,
            shell_oxygen_names=oxygen_names,
            train_indices_0=train_indices_0,
            train_indices_1=train_indices_1,
            wrap_box_eval=bool(args.no_fixed_box),
            wrap_residue_blocks=wrap_blocks,
        )
    elif args.flow_space == "mixedmaf":
        model = SmallMolBidirectionalTMBARMapMixed(
            pot0, pot1,
            topology_file_path=str(topo_path),
            coordinates_file_path=str(traj0_path),
            coordinates_file_path_2=str(traj1_path),
            temperature=temperature,
            batch_size=int(args.batch_size),
            mapped_atoms=mapped_atoms,
            conditioning_atoms=conditioning_atoms,
            origin_atom=origin_atom,
            axes_atoms=axes_atoms,
            tfep_logger_dir_path=str(outdir / "tfep_logs"),
            n_maf_layers=int(args.maf_layers),
            remove_translation=bool(args.remove_translation),
            remove_rotation=bool(args.remove_rotation),
            distance_lower_limit_displacement_angstrom=float(args.distance_lower_limit_displacement_angstrom),
            maf_hidden_layers=int(args.maf_hidden_layers),
            maf_weight_norm=maf_weight_norm,
            maf_initialize_identity=maf_initialize_identity,
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            objective=objective,
            lambda_bar=lambda_bar,
            bar_detach_df=bool(bar_detach_df),
            bar_warm_start=bool(bar_warm_start),
            bar_max_iter=int(args.bar_max_iter),
            bar_tol=float(args.bar_tol),
            bar_lambda=legacy_bar_lambda,
            logJ_penalty_weight=logJ_penalty_weight,
            stochastic_training_config=stochastic_training_config,
            shell_k1=k1,
            shell_k2=k2,
            shell_center=_as_indices_or_selection(shell_center) if shell_enabled else None,
            shell_water_selection=args.conditioning_shell_water_selection,
            shell_oxygen_names=oxygen_names,
            train_indices_0=train_indices_0,
            train_indices_1=train_indices_1,
            wrap_box_eval=bool(args.no_fixed_box),
            wrap_residue_blocks=wrap_blocks,
        )
    elif args.flow_space == "shell-equivariant":
        if shell_equiv_spec is None:
            raise RuntimeError("Internal error: shell_equiv_spec was not built.")

        model = SmallMolBidirectionalTMBARMapShellEquivariant(
            pot0, pot1,
            topology_file_path=str(topo_path),
            coordinates_file_path=str(traj0_path),
            coordinates_file_path_2=str(traj1_path),
            temperature=temperature,
            batch_size=int(args.batch_size),
            mapped_atoms=mapped_atoms,
            conditioning_atoms=conditioning_atoms,
            origin_atom=origin_atom,
            axes_atoms=axes_atoms,
            tfep_logger_dir_path=str(outdir / "tfep_logs"),
            n_maf_layers=1,
            maf_hidden_layers=1,
            maf_weight_norm=False,
            maf_initialize_identity=True,
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            objective=objective,
            lambda_bar=lambda_bar,
            bar_detach_df=bool(bar_detach_df),
            bar_warm_start=bool(bar_warm_start),
            bar_max_iter=int(args.bar_max_iter),
            bar_tol=float(args.bar_tol),
            bar_lambda=legacy_bar_lambda,
            logJ_penalty_weight=logJ_penalty_weight,
            stochastic_training_config=stochastic_training_config,
            shell_k1=k1,
            shell_k2=k2,
            shell_center=_as_indices_or_selection(shell_center) if shell_enabled else None,
            shell_water_selection=args.conditioning_shell_water_selection,
            shell_oxygen_names=oxygen_names,
            train_indices_0=train_indices_0,
            train_indices_1=train_indices_1,
            wrap_box_eval=bool(args.no_fixed_box),
            wrap_residue_blocks=wrap_blocks,
            shell_equiv_spec=shell_equiv_spec,
            shell_equiv_cutoff_angstrom=float(args.shell_equivariant_cutoff_angstrom),
            shell_equiv_tau_angstrom=float(args.shell_equivariant_tau_angstrom),
            shell_equiv_hidden_dim=int(args.shell_equivariant_hidden_dim),
            shell_equiv_max_displacement_angstrom=float(args.shell_equivariant_max_displacement_angstrom),
            shell_equiv_initial_log_scale=float(args.shell_equivariant_initial_log_scale),
            shell_equiv_top_k=int(args.shell_equivariant_top_k),
            shell_equiv_max_rotation_radians=float(args.shell_equivariant_max_rotation_radians),
            shell_equiv_max_internal_log_scale=float(args.shell_equivariant_max_internal_log_scale),
            shell_equiv_solute_maf_layers=int(args.shell_equivariant_solute_maf_layers),
            shell_equiv_solute_maf_hidden_dim=int(args.shell_equivariant_solute_maf_hidden_dim),
            shell_equiv_solute_maf_max_log_scale=float(args.shell_equivariant_solute_maf_max_log_scale),
            shell_equiv_solute_maf_max_shift_angstrom=float(args.shell_equivariant_solute_maf_max_shift_angstrom),
        )
    # Install a direct evaluator on the instance so inherited training/evaluation
    # paths avoid the MultiStatePotential.energy() code path that segfaults here.
    model._eval_potential = functools.partial(_eval_potential_direct_for_model, model)
    objective_effective = str(getattr(model, "_objective", objective))
    print(f"[objective] model-effective objective='{objective_effective}'")

    return model, {
        "outdir": outdir,
        "state0_dir": state0_dir,
        "state1_dir": state1_dir,
        "traj0_path": traj0_path,
        "traj1_path": traj1_path,
        "topo_path": topo_path,
        "system0_xml_path": system0_xml_path,
        "system1_xml_path": system1_xml_path,
        "objective_effective": objective_effective,
        "objective_requested": objective,
        "legacy_bar_regularizer_active": legacy_bar_active,
        "legacy_bar_lambda": legacy_bar_lambda,
        "lambda_bar": lambda_bar,
        "logJ_penalty_weight": logJ_penalty_weight,
        "objective_mode_description": _objective_mode_description(
            objective,
            lambda_bar=lambda_bar,
            legacy_bar_lambda=legacy_bar_lambda,
            logJ_penalty_weight=logJ_penalty_weight,
        ),
        "stochastic_training": None if stochastic_training_config is None else {
            "enabled": bool(stochastic_training_config.enabled),
            "kernel": stochastic_training_config.kernel,
            "step_size": stochastic_training_config.step_size,
            "diffusion": stochastic_training_config.diffusion,
            "num_blocks": stochastic_training_config.num_blocks,
            "steps_per_block": stochastic_training_config.steps_per_block,
            "train_mc_samples": stochastic_training_config.train_mc_samples,
            "gradient_policy": stochastic_training_config.gradient_policy,
            "apply_to": stochastic_training_config.apply_to,
            "allow_constrained_cartesian_ula": stochastic_training_config.allow_constrained_cartesian_ula,
            "statistical_warning": (
                "Experimental Cartesian ULA on constrained/PBC molecular endpoints; "
                "not a rigorously derived constrained-manifold stochastic flow."
                if stochastic_training_config.allow_constrained_cartesian_ula else ""
            ),
        },
    }


def build_trainer(args: argparse.Namespace, outdir: Path) -> L.Trainer:
    ckpt_dir = outdir / "checkpoints"
    ckpt_cb = ModelCheckpoint(dirpath=str(ckpt_dir), save_top_k=1, monitor="loss", mode="min", filename="best")
    logger = CSVLogger(save_dir=str(outdir / "logs"), name="tfep")
    return L.Trainer(
        max_epochs=int(args.epochs),
        accelerator=args.torch_accelerator,
        devices=int(args.torch_devices),
        callbacks=[ckpt_cb],
        logger=logger,
        plugins=[LightningEnvironment()],
        enable_progress_bar=True,
        log_every_n_steps=1,
    )


def close_model_openmm_workers(model) -> None:
    """Close any isolated OpenMM workers attached to the model."""
    if model is None:
        return
    potential_func = getattr(model, "_potential_energy_func", None)
    potentials = getattr(potential_func, "potentials", None)
    if potentials is None:
        potentials = [potential_func]
    for pot in potentials:
        inner = getattr(pot, "pot", pot)
        close_fn = getattr(inner, "close", None)
        if callable(close_fn):
            close_fn()


def run_training(
    args: argparse.Namespace,
    *,
    outdir_override: Optional[Union[str, Path]] = None,
    train_indices_0: Optional[Sequence[int]] = None,
    train_indices_1: Optional[Sequence[int]] = None,
) -> TrainingArtifacts:
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

    run_config = _namespace_to_jsonable(args)
    run_config["effective_outdir"] = str(outdir.resolve())
    run_config["objective_requested"] = str(ctx.get("objective_requested", str(args.objective)))
    run_config["objective_effective"] = str(ctx.get("objective_effective", run_config["objective_requested"]))
    run_config["legacy_bar_regularizer_active"] = bool(ctx.get("legacy_bar_regularizer_active", False))
    run_config["legacy_bar_lambda"] = float(ctx.get("legacy_bar_lambda", float(args.bar_lambda)))
    run_config["lambda_bar"] = float(ctx.get("lambda_bar", float(args.lambda_bar)))
    run_config["logJ_penalty_weight"] = float(ctx.get("logJ_penalty_weight", 0.0))
    run_config["objective_mode_description"] = str(ctx.get("objective_mode_description", ""))
    if train_indices_0 is not None:
        run_config["train_indices_0_count"] = int(len(train_indices_0))
    elif args.state0_train_indices_file is not None:
        run_config["train_indices_0_file"] = str(Path(args.state0_train_indices_file).expanduser().resolve())
    if train_indices_1 is not None:
        run_config["train_indices_1_count"] = int(len(train_indices_1))
    elif args.state1_train_indices_file is not None:
        run_config["train_indices_1_file"] = str(Path(args.state1_train_indices_file).expanduser().resolve())
    (outdir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    trainer = build_trainer(args, outdir)
    trainer.fit(model)

    return TrainingArtifacts(
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


def main():
    args = build_argparser().parse_args()
    artifacts = run_training(args)
    try:
        if str(getattr(args, "estimator", "deterministic-tfep")) == "stochastic-path-tfep":
            run_stochastic_path_evaluation_for_model(
                artifacts,
                args,
                analysis_dir=artifacts.outdir,
                subset_indices_0=None,
                subset_indices_1=None,
                batch_size=int(args.batch_size),
            )
    finally:
        close_model_openmm_workers(artifacts.model)


if __name__ == "__main__":
    main()
