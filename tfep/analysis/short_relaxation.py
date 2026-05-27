"""Diagnostic short target-side relaxation for deterministic TFEP maps.

This module intentionally does *not* implement a corrected TFEP estimator.  It
only answers a diagnostic question: after a deterministic TFEP map has produced
``y_B = M(x_A)``, would a short local target-side relaxation remove strain or
clashes in ``y_B``?

The production deterministic mapped work remains

``u_B(M(x_A)) - u_A(x_A) - log|det J_M(x_A)|``.

Relaxed coordinates and relaxed energies written by this module must therefore
be interpreted as post-processing diagnostics, not as BAR/TMBAR work values.
"""
from __future__ import annotations

import csv
import json
import math
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

_R_KJ_MOL_K = 0.00831446261815324


@dataclass
class RelaxationDiagnosticConfig:
    """Configuration for diagnostic-only target-side relaxation.

    All coordinates consumed and written by this module are in Angstrom.  OpenMM
    energies are recorded both in kJ/mol and in reduced units using
    ``temperature_k``.
    """

    mode: str = "none"
    backend: str = "auto"
    steps: int = 100
    timestep_fs: float = 0.5
    temperature_k: float = 298.15
    friction_ps: float = 10.0
    seed: int = 123
    restraint_k: float = 1000.0
    restraint_selection: str = "mapped"
    output_dir: str = "relaxation_diagnostics"
    every_n_frames: int = 1
    max_frames: int = 100
    save_trajectories: bool = False
    strict: bool = False
    platform_name: str = "CPU"
    cpu_threads: int = 1
    clash_cutoff_angstrom: float = 1.2
    min_distance_search_cutoff_angstrom: float = 5.0

    @property
    def enabled(self) -> bool:
        return str(self.mode).lower() not in {"", "none"}

    @property
    def kt_kj_mol(self) -> float:
        return _R_KJ_MOL_K * float(self.temperature_k)


@dataclass
class RelaxationFrameRecord:
    """One diagnostic record for one mapped frame."""

    frame_index: int
    direction: str
    source_state: int
    target_state: int
    dataset_sample_index: int
    trajectory_sample_index: int
    success: bool
    exception: str = ""
    target_energy_before_kj_mol: float = math.nan
    target_energy_after_kj_mol: float = math.nan
    target_reduced_energy_before: float = math.nan
    target_reduced_energy_after: float = math.nan
    delta_u_relax: float = math.nan
    deterministic_tfep_work: float = math.nan
    log_det_J: float = math.nan
    rmsd_all_angstrom: float = math.nan
    rmsd_solute_angstrom: float = math.nan
    rmsd_shell_angstrom: float = math.nan
    max_displacement_angstrom: float = math.nan
    n_atoms_disp_gt_0p5A: int = 0
    n_atoms_disp_gt_1p0A: int = 0
    n_atoms_disp_gt_2p0A: int = 0
    min_pair_distance_before_angstrom: float = math.nan
    min_pair_distance_after_angstrom: float = math.nan
    clash_count_before: int = 0
    clash_count_after: int = 0
    water_oxygen_displacement_mean_angstrom: float = math.nan
    water_oxygen_displacement_max_angstrom: float = math.nan


def _as_path(path: str | Path | None) -> Optional[Path]:
    if path is None:
        return None
    return Path(path).expanduser().resolve()


def _load_openmm_objects(system_xml_path: Path, topology_pdb_path: Path):
    try:
        import openmm
        from openmm.app import PDBFile
    except Exception as exc:  # pragma: no cover - depends on optional OpenMM.
        raise ImportError("OpenMM is required for short relaxation diagnostics") from exc

    with Path(system_xml_path).open("r", encoding="utf-8") as handle:
        system = openmm.XmlSerializer.deserialize(handle.read())
    pdb = PDBFile(str(topology_pdb_path))
    return openmm, system, pdb.topology


def _platform(openmm, config: RelaxationDiagnosticConfig):
    name = str(config.platform_name or "CPU")
    try:
        platform = openmm.Platform.getPlatformByName(name)
    except Exception:
        platform = openmm.Platform.getPlatformByName("Reference")
        return platform, {}

    properties: dict[str, str] = {}
    if name == "CPU":
        properties["Threads"] = str(int(config.cpu_threads))
    return platform, properties


def _copy_system(openmm, system):
    xml = openmm.XmlSerializer.serialize(system)
    return openmm.XmlSerializer.deserialize(xml)


def _dimensions_to_vectors_nm(openmm, dimensions_angstrom: Optional[np.ndarray]):
    if dimensions_angstrom is None:
        return None
    dims = np.asarray(dimensions_angstrom, dtype=float).reshape(-1)
    if dims.size < 6 or not np.all(np.isfinite(dims[:6])):
        return None
    try:
        from MDAnalysis.lib.mdamath import triclinic_vectors
    except Exception:
        return None
    vectors_angstrom = np.asarray(triclinic_vectors(dims[:6]), dtype=float)
    vectors_nm = vectors_angstrom / 10.0
    return tuple(openmm.Vec3(*row) for row in vectors_nm)


def _positions_to_openmm(openmm, positions_angstrom: np.ndarray):
    from openmm import unit

    positions_nm = np.asarray(positions_angstrom, dtype=float).reshape(-1, 3) / 10.0
    return [openmm.Vec3(*xyz) for xyz in positions_nm] * unit.nanometer


def _positions_from_state_angstrom(state) -> np.ndarray:
    from openmm import unit

    positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    return np.asarray(positions, dtype=float) * 10.0


def _energy_kj_mol(state) -> float:
    from openmm import unit

    return float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))


def _topology_bond_pairs(topology) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for atom1, atom2 in topology.bonds():
        i, j = int(atom1.index), int(atom2.index)
        if i > j:
            i, j = j, i
        pairs.add((i, j))
    return pairs


def _selection_to_indices(
    topology_pdb_path: Path,
    selection: str,
    *,
    mapped_indices: Optional[np.ndarray] = None,
) -> np.ndarray:
    text = str(selection or "").strip()
    if text == "" or text.lower() == "none":
        return np.empty(0, dtype=int)
    if text.lower() == "mapped":
        if mapped_indices is None:
            return np.empty(0, dtype=int)
        return np.asarray(mapped_indices, dtype=int).reshape(-1)

    import MDAnalysis as mda

    universe = mda.Universe(str(topology_pdb_path))
    selected = universe.select_atoms(text)
    return np.asarray(selected.indices, dtype=int)


def _indices_from_water_shell_mapping(mapping_path: str | Path | None) -> np.ndarray:
    path = _as_path(mapping_path)
    if path is None or not path.is_file():
        return np.empty(0, dtype=int)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    indices: list[int] = []
    for record in data.get("selected_water_records", []):
        indices.extend(int(i) for i in record.get("atom_indices", []))
    return np.unique(np.asarray(indices, dtype=int)) if indices else np.empty(0, dtype=int)


def _water_oxygen_indices(topology_pdb_path: Path) -> np.ndarray:
    import MDAnalysis as mda

    universe = mda.Universe(str(topology_pdb_path))
    residues = {"HOH", "SOL", "WAT", "TIP3", "TP3M"}
    names = {"O", "OW", "OH2"}
    indices = [
        int(atom.index)
        for atom in universe.atoms
        if str(atom.resname).strip() in residues and str(atom.name).strip() in names
    ]
    return np.asarray(indices, dtype=int)


class OpenMMRelaxationRunner:
    """OpenMM runner for diagnostic target-side relaxation.

    The runner evaluates and relaxes mapped coordinates under one endpoint
    potential.  It does not know about TFEP work or BAR and therefore cannot
    contaminate deterministic estimator arrays.
    """

    def __init__(
        self,
        *,
        system_xml_path: str | Path,
        topology_pdb_path: str | Path,
        config: RelaxationDiagnosticConfig,
        restraint_indices: Optional[Sequence[int]] = None,
    ):
        self.system_xml_path = Path(system_xml_path).expanduser().resolve()
        self.topology_pdb_path = Path(topology_pdb_path).expanduser().resolve()
        self.config = config
        self.openmm, base_system, self.topology = _load_openmm_objects(
            self.system_xml_path,
            self.topology_pdb_path,
        )
        self.system = _copy_system(self.openmm, base_system)
        self.restraint_indices = np.asarray(restraint_indices if restraint_indices is not None else [], dtype=int)
        self._restraint_force = None

        mode = str(config.mode).lower()
        if mode in {"restrained-langevin"} and self.restraint_indices.size > 0:
            self._add_restraints()

        self.integrator = self._make_integrator(mode)
        platform, properties = _platform(self.openmm, config)
        self.context = self.openmm.Context(self.system, self.integrator, platform, properties)

    def _make_integrator(self, mode: str):
        from openmm import unit

        if mode in {"langevin", "restrained-langevin"}:
            return self.openmm.LangevinMiddleIntegrator(
                float(self.config.temperature_k) * unit.kelvin,
                float(self.config.friction_ps) / unit.picosecond,
                float(self.config.timestep_fs) * unit.femtosecond,
            )
        return self.openmm.VerletIntegrator(0.001 * unit.picosecond)

    def _add_restraints(self) -> None:
        force = self.openmm.CustomExternalForce(
            "0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)"
        )
        force.addGlobalParameter("k", float(self.config.restraint_k))
        force.addPerParticleParameter("x0")
        force.addPerParticleParameter("y0")
        force.addPerParticleParameter("z0")
        for atom_idx in self.restraint_indices:
            force.addParticle(int(atom_idx), [0.0, 0.0, 0.0])
        self.system.addForce(force)
        self._restraint_force = force

    def _set_restraint_references(self, positions_angstrom: np.ndarray) -> None:
        if self._restraint_force is None:
            return
        positions_nm = np.asarray(positions_angstrom, dtype=float).reshape(-1, 3) / 10.0
        for local_i, atom_idx in enumerate(self.restraint_indices):
            self._restraint_force.setParticleParameters(
                int(local_i),
                int(atom_idx),
                [float(x) for x in positions_nm[int(atom_idx)]],
            )
        self._restraint_force.updateParametersInContext(self.context)

    def relax(
        self,
        positions_angstrom: np.ndarray,
        *,
        dimensions_angstrom: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> dict[str, Any]:
        """Relax one mapped frame and return diagnostic coordinates/energies."""

        mode = str(self.config.mode).lower()
        positions_angstrom = np.asarray(positions_angstrom, dtype=float).reshape(-1, 3)

        box_vectors = _dimensions_to_vectors_nm(self.openmm, dimensions_angstrom)
        if box_vectors is not None:
            self.context.setPeriodicBoxVectors(*box_vectors)

        self.context.setPositions(_positions_to_openmm(self.openmm, positions_angstrom))
        self._set_restraint_references(positions_angstrom)
        before_state = self.context.getState(getEnergy=True)
        energy_before = _energy_kj_mol(before_state)

        if mode == "minimize":
            self.openmm.LocalEnergyMinimizer.minimize(
                self.context,
                tolerance=10.0,
                maxIterations=max(1, int(self.config.steps)),
            )
        elif mode in {"langevin", "restrained-langevin"}:
            from openmm import unit

            if seed is not None and hasattr(self.integrator, "setRandomNumberSeed"):
                self.integrator.setRandomNumberSeed(int(seed))
            self.context.setVelocitiesToTemperature(
                float(self.config.temperature_k) * unit.kelvin,
                int(seed or self.config.seed),
            )
            self.integrator.step(max(1, int(self.config.steps)))
        elif mode == "noop":
            pass
        else:
            raise ValueError(f"Unsupported relaxation mode for OpenMM runner: {mode}")

        after_state = self.context.getState(getEnergy=True, getPositions=True)
        energy_after = _energy_kj_mol(after_state)
        relaxed = _positions_from_state_angstrom(after_state)
        return {
            "energy_before_kj_mol": energy_before,
            "energy_after_kj_mol": energy_after,
            "relaxed_positions_angstrom": relaxed,
        }


def _rmsd(displacement: np.ndarray, indices: Optional[np.ndarray] = None) -> float:
    if indices is None:
        disp = displacement
    else:
        idx = np.asarray(indices, dtype=int).reshape(-1)
        if idx.size == 0:
            return math.nan
        disp = displacement[idx]
    if disp.size == 0:
        return math.nan
    return float(np.sqrt(np.mean(np.sum(disp * disp, axis=1))))


def _pair_stats(
    coords_angstrom: np.ndarray,
    *,
    dimensions_angstrom: Optional[np.ndarray],
    bonded_pairs: set[tuple[int, int]],
    clash_cutoff_angstrom: float,
    min_search_cutoff_angstrom: float,
) -> tuple[float, int]:
    try:
        from MDAnalysis.lib.distances import self_capped_distance

        coords = np.asarray(coords_angstrom, dtype=np.float32).reshape(-1, 3)
        box = None
        if dimensions_angstrom is not None:
            dims = np.asarray(dimensions_angstrom, dtype=np.float32).reshape(-1)
            if dims.size >= 6 and np.all(np.isfinite(dims[:6])):
                box = dims[:6]
        pairs, distances = self_capped_distance(
            coords,
            max_cutoff=float(min_search_cutoff_angstrom),
            box=box,
            return_distances=True,
        )
        if len(pairs) == 0:
            return math.nan, 0
        pairs = np.asarray(pairs, dtype=int)
        distances = np.asarray(distances, dtype=float)
        keep = np.ones(len(pairs), dtype=bool)
        if bonded_pairs:
            for k, (i, j) in enumerate(pairs):
                a, b = (int(i), int(j)) if i < j else (int(j), int(i))
                if (a, b) in bonded_pairs:
                    keep[k] = False
        distances = distances[keep]
        if distances.size == 0:
            return math.nan, 0
        return float(np.min(distances)), int(np.count_nonzero(distances < float(clash_cutoff_angstrom)))
    except Exception:
        return math.nan, 0


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def _make_record(
    *,
    frame_index: int,
    direction: str,
    source_state: int,
    target_state: int,
    dataset_sample_index: int,
    trajectory_sample_index: int,
    mapped_positions: np.ndarray,
    relaxed_positions: np.ndarray,
    energy_before_kj_mol: float,
    energy_after_kj_mol: float,
    config: RelaxationDiagnosticConfig,
    dimensions_angstrom: Optional[np.ndarray],
    bonded_pairs: set[tuple[int, int]],
    solute_indices: Optional[np.ndarray],
    shell_indices: Optional[np.ndarray],
    water_oxygen_indices: Optional[np.ndarray],
    deterministic_tfep_work: float = math.nan,
    log_det_J: float = math.nan,
) -> RelaxationFrameRecord:
    displacement = np.asarray(relaxed_positions, dtype=float) - np.asarray(mapped_positions, dtype=float)
    disp_norm = np.linalg.norm(displacement, axis=1)
    min_before, clashes_before = _pair_stats(
        mapped_positions,
        dimensions_angstrom=dimensions_angstrom,
        bonded_pairs=bonded_pairs,
        clash_cutoff_angstrom=config.clash_cutoff_angstrom,
        min_search_cutoff_angstrom=config.min_distance_search_cutoff_angstrom,
    )
    min_after, clashes_after = _pair_stats(
        relaxed_positions,
        dimensions_angstrom=dimensions_angstrom,
        bonded_pairs=bonded_pairs,
        clash_cutoff_angstrom=config.clash_cutoff_angstrom,
        min_search_cutoff_angstrom=config.min_distance_search_cutoff_angstrom,
    )

    water_mean = math.nan
    water_max = math.nan
    if water_oxygen_indices is not None and len(water_oxygen_indices) > 0:
        valid = np.asarray(water_oxygen_indices, dtype=int)
        valid = valid[(valid >= 0) & (valid < len(disp_norm))]
        if valid.size > 0:
            water_disp = disp_norm[valid]
            water_mean = float(np.mean(water_disp))
            water_max = float(np.max(water_disp))

    before_reduced = float(energy_before_kj_mol) / config.kt_kj_mol
    after_reduced = float(energy_after_kj_mol) / config.kt_kj_mol
    success = True
    exception = ""
    if str(config.mode).lower() == "minimize" and float(energy_after_kj_mol) > float(energy_before_kj_mol) + 1e-8:
        success = False
        exception = (
            "Energy increased during minimization; relaxed coordinates were "
            "written for inspection but should be treated as a failed diagnostic frame."
        )

    return RelaxationFrameRecord(
        frame_index=int(frame_index),
        direction=str(direction),
        source_state=int(source_state),
        target_state=int(target_state),
        dataset_sample_index=int(dataset_sample_index),
        trajectory_sample_index=int(trajectory_sample_index),
        success=success,
        exception=exception,
        target_energy_before_kj_mol=float(energy_before_kj_mol),
        target_energy_after_kj_mol=float(energy_after_kj_mol),
        target_reduced_energy_before=before_reduced,
        target_reduced_energy_after=after_reduced,
        delta_u_relax=after_reduced - before_reduced,
        deterministic_tfep_work=_safe_float(deterministic_tfep_work),
        log_det_J=_safe_float(log_det_J),
        rmsd_all_angstrom=_rmsd(displacement),
        rmsd_solute_angstrom=_rmsd(displacement, solute_indices),
        rmsd_shell_angstrom=_rmsd(displacement, shell_indices),
        max_displacement_angstrom=float(np.max(disp_norm)) if disp_norm.size else math.nan,
        n_atoms_disp_gt_0p5A=int(np.count_nonzero(disp_norm > 0.5)),
        n_atoms_disp_gt_1p0A=int(np.count_nonzero(disp_norm > 1.0)),
        n_atoms_disp_gt_2p0A=int(np.count_nonzero(disp_norm > 2.0)),
        min_pair_distance_before_angstrom=min_before,
        min_pair_distance_after_angstrom=min_after,
        clash_count_before=clashes_before,
        clash_count_after=clashes_after,
        water_oxygen_displacement_mean_angstrom=water_mean,
        water_oxygen_displacement_max_angstrom=water_max,
    )


def failure_record(
    *,
    frame_index: int,
    direction: str,
    source_state: int,
    target_state: int,
    dataset_sample_index: int,
    trajectory_sample_index: int,
    exception: BaseException,
) -> RelaxationFrameRecord:
    return RelaxationFrameRecord(
        frame_index=int(frame_index),
        direction=str(direction),
        source_state=int(source_state),
        target_state=int(target_state),
        dataset_sample_index=int(dataset_sample_index),
        trajectory_sample_index=int(trajectory_sample_index),
        success=False,
        exception=f"{type(exception).__name__}: {exception}",
    )


def run_short_relaxation_diagnostic(
    *,
    mapped_frames: Mapping[str, list[dict[str, Any]]],
    target_system_xml_paths: Mapping[str, str | Path],
    target_topology_pdb_paths: Mapping[str, str | Path],
    config: RelaxationDiagnosticConfig,
    mapped_atom_indices: Optional[Sequence[int]] = None,
    solute_selection: Optional[str] = None,
    water_shell_mapping_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run diagnostic relaxation on mapped frames from one or more directions.

    Parameters
    ----------
    mapped_frames
        Mapping from direction label to frame dictionaries.  Each frame dict must
        include ``positions_angstrom`` and may include dimensions, work, logJ,
        sample indices, and source/target state labels.
    target_system_xml_paths, target_topology_pdb_paths
        Per-direction endpoint files for the target state.
    config
        Diagnostic configuration.

    Returns
    -------
    dict
        In-memory records and coordinate arrays ready for writing.
    """

    mode = str(config.mode).lower()
    backend = str(config.backend).lower()
    if mode == "none":
        return {"records": [], "mapped_coordinates": {}, "relaxed_coordinates": {}}
    if backend == "gromacs":
        raise NotImplementedError(
            "GROMACS short relaxation is not implemented. The repository has "
            "GROMACS rerun energy support, but no safe reusable minimization/MD "
            "relaxation utility."
        )
    if backend not in {"auto", "openmm"}:
        raise ValueError(f"Unsupported relaxation backend: {config.backend}")

    mapped_indices = None if mapped_atom_indices is None else np.asarray(mapped_atom_indices, dtype=int)
    shell_indices = _indices_from_water_shell_mapping(water_shell_mapping_path)
    records: list[RelaxationFrameRecord] = []
    mapped_out: dict[str, list[np.ndarray]] = {}
    relaxed_out: dict[str, list[np.ndarray]] = {}

    for direction, frames in mapped_frames.items():
        topology_path = Path(target_topology_pdb_paths[direction]).expanduser().resolve()
        restraint_indices = _selection_to_indices(
            topology_path,
            config.restraint_selection,
            mapped_indices=mapped_indices,
        )
        solute_indices = _selection_to_indices(
            topology_path,
            solute_selection or "mapped",
            mapped_indices=mapped_indices,
        )
        water_oxygen = _water_oxygen_indices(topology_path)
        runner = OpenMMRelaxationRunner(
            system_xml_path=target_system_xml_paths[direction],
            topology_pdb_path=topology_path,
            config=config,
            restraint_indices=restraint_indices,
        )
        bonded_pairs = _topology_bond_pairs(runner.topology)
        mapped_out[direction] = []
        relaxed_out[direction] = []

        for local_i, frame in enumerate(frames):
            mapped_positions = np.asarray(frame["positions_angstrom"], dtype=float).reshape(-1, 3)
            try:
                result = runner.relax(
                    mapped_positions,
                    dimensions_angstrom=frame.get("dimensions_angstrom"),
                    seed=int(config.seed) + int(frame.get("frame_index", local_i)),
                )
                relaxed_positions = np.asarray(result["relaxed_positions_angstrom"], dtype=float).reshape(-1, 3)
                record = _make_record(
                    frame_index=int(frame.get("frame_index", local_i)),
                    direction=direction,
                    source_state=int(frame.get("source_state", -1)),
                    target_state=int(frame.get("target_state", -1)),
                    dataset_sample_index=int(frame.get("dataset_sample_index", -1)),
                    trajectory_sample_index=int(frame.get("trajectory_sample_index", -1)),
                    mapped_positions=mapped_positions,
                    relaxed_positions=relaxed_positions,
                    energy_before_kj_mol=float(result["energy_before_kj_mol"]),
                    energy_after_kj_mol=float(result["energy_after_kj_mol"]),
                    config=config,
                    dimensions_angstrom=frame.get("dimensions_angstrom"),
                    bonded_pairs=bonded_pairs,
                    solute_indices=solute_indices,
                    shell_indices=shell_indices,
                    water_oxygen_indices=water_oxygen,
                    deterministic_tfep_work=frame.get("deterministic_tfep_work", math.nan),
                    log_det_J=frame.get("log_det_J", math.nan),
                )
                mapped_out[direction].append(mapped_positions)
                relaxed_out[direction].append(relaxed_positions)
            except Exception as exc:
                if config.strict:
                    raise
                record = failure_record(
                    frame_index=int(frame.get("frame_index", local_i)),
                    direction=direction,
                    source_state=int(frame.get("source_state", -1)),
                    target_state=int(frame.get("target_state", -1)),
                    dataset_sample_index=int(frame.get("dataset_sample_index", -1)),
                    trajectory_sample_index=int(frame.get("trajectory_sample_index", -1)),
                    exception=exc,
                )
                record.exception += "\n" + traceback.format_exc(limit=3)
            records.append(record)

    return {
        "records": records,
        "mapped_coordinates": {
            key: np.asarray(value, dtype=np.float32) if value else np.empty((0, 0, 3), dtype=np.float32)
            for key, value in mapped_out.items()
        },
        "relaxed_coordinates": {
            key: np.asarray(value, dtype=np.float32) if value else np.empty((0, 0, 3), dtype=np.float32)
            for key, value in relaxed_out.items()
        },
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_relaxation_outputs(
    result: Mapping[str, Any],
    *,
    output_dir: str | Path,
    config: RelaxationDiagnosticConfig,
    metadata: Optional[Mapping[str, Any]] = None,
) -> None:
    """Write diagnostic records, arrays, plots, and explanatory README."""

    outdir = Path(output_dir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    records = [asdict(record) if hasattr(record, "__dataclass_fields__") else dict(record) for record in result.get("records", [])]

    _write_csv(records, outdir / "relaxation_summary.csv")
    (outdir / "relaxation_summary.json").write_text(json.dumps(_json_ready(records), indent=2), encoding="utf-8")

    failures = [row for row in records if not bool(row.get("success", False))]
    _write_csv(failures, outdir / "failure_log.csv")

    displacement_rows = [
        {
            key: row.get(key)
            for key in (
                "frame_index",
                "direction",
                "rmsd_all_angstrom",
                "rmsd_solute_angstrom",
                "rmsd_shell_angstrom",
                "max_displacement_angstrom",
                "n_atoms_disp_gt_0p5A",
                "n_atoms_disp_gt_1p0A",
                "n_atoms_disp_gt_2p0A",
                "water_oxygen_displacement_mean_angstrom",
                "water_oxygen_displacement_max_angstrom",
            )
        }
        for row in records
    ]
    clash_rows = [
        {
            key: row.get(key)
            for key in (
                "frame_index",
                "direction",
                "min_pair_distance_before_angstrom",
                "min_pair_distance_after_angstrom",
                "clash_count_before",
                "clash_count_after",
            )
        }
        for row in records
    ]
    _write_csv(displacement_rows, outdir / "displacement_stats.csv")
    _write_csv(clash_rows, outdir / "clash_stats.csv")

    energy_pairs = np.asarray(
        [
            [
                _safe_float(row.get("target_energy_before_kj_mol")),
                _safe_float(row.get("target_energy_after_kj_mol")),
                _safe_float(row.get("target_reduced_energy_before")),
                _safe_float(row.get("target_reduced_energy_after")),
                _safe_float(row.get("delta_u_relax")),
            ]
            for row in records
        ],
        dtype=np.float64,
    )
    np.save(outdir / "energy_before_after.npy", energy_pairs)

    for direction, array in result.get("mapped_coordinates", {}).items():
        safe = str(direction).replace("/", "_")
        np.save(outdir / f"mapped_coordinates_{safe}.npy", np.asarray(array, dtype=np.float32))
    for direction, array in result.get("relaxed_coordinates", {}).items():
        safe = str(direction).replace("/", "_")
        np.save(outdir / f"relaxed_coordinates_{safe}.npy", np.asarray(array, dtype=np.float32))

    meta = {
        "statistical_meaning": (
            "Diagnostic only. Relaxed coordinates/energies are not used in the "
            "deterministic TFEP/BAR/TMBAR estimator."
        ),
        "config": asdict(config),
        "metadata": dict(metadata or {}),
    }
    (outdir / "metadata.json").write_text(json.dumps(_json_ready(meta), indent=2), encoding="utf-8")
    _write_readme(outdir / "README.md")
    plot_relaxation_diagnostics(records, outdir / "plots")


def _write_readme(path: Path) -> None:
    text = """# Short Relaxation Diagnostics

These files are diagnostic post-processing outputs only.

The deterministic TFEP/BAR/TMBAR estimator still uses the unrelaxed mapped
coordinates and the original deterministic log-Jacobian:

```text
w_A_to_B = u_B(M(x_A)) - u_A(x_A) - log|det J_M(x_A)|
```

Do not interpret `target_energy_after_*` or `delta_u_relax` as corrected TFEP
work.  A statistically valid stochastic relaxation estimator would need the
forward and reverse path-probability terms.

Interpretation guide:

- Large negative `delta_u_relax` means mapped target structures are strained.
- Large shell-water RMSD suggests missing solvent relaxation.
- Large solute RMSD suggests the solute map is insufficient or too rigid.
- A few huge energy drops often indicate outlier clashes worth visual inspection.
- Broad systematic improvement after relaxation motivates either a better
  deterministic map or a future path-weighted stochastic TFEP estimator.
"""
    path.write_text(text, encoding="utf-8")


def _finite(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    vals = np.asarray([_safe_float(row.get(key)) for row in rows if bool(row.get("success", False))], dtype=float)
    return vals[np.isfinite(vals)]


def _scatter(rows: list[dict[str, Any]], xkey: str, ykey: str) -> tuple[np.ndarray, np.ndarray]:
    x = []
    y = []
    for row in rows:
        if not bool(row.get("success", False)):
            continue
        xv = _safe_float(row.get(xkey))
        yv = _safe_float(row.get(ykey))
        if np.isfinite(xv) and np.isfinite(yv):
            x.append(xv)
            y.append(yv)
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def _maybe_savefig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def plot_relaxation_diagnostics(records: list[dict[str, Any]], output_dir: str | Path) -> None:
    """Create simple diagnostic plots if matplotlib is available."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    outdir = Path(output_dir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    def hist(key: str, path: str, xlabel: str) -> None:
        vals = _finite(records, key)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        if vals.size:
            ax.hist(vals, bins=min(50, max(10, int(np.sqrt(vals.size)))))
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")
        _maybe_savefig(fig, outdir / path)
        plt.close(fig)

    hist("delta_u_relax", "energy_drop_hist.png", "Delta u_relax = u_after - u_before")
    hist("rmsd_all_angstrom", "rmsd_hist.png", "mapped-relaxed RMSD (Angstrom)")

    plots = [
        (
            "target_reduced_energy_before",
            "target_reduced_energy_after",
            "mapped_vs_relaxed_energy.png",
            "target reduced energy before",
            "target reduced energy after",
        ),
        (
            "min_pair_distance_before_angstrom",
            "min_pair_distance_after_angstrom",
            "min_distance_before_after.png",
            "min pair distance before (Angstrom)",
            "min pair distance after (Angstrom)",
        ),
        (
            "deterministic_tfep_work",
            "delta_u_relax",
            "work_vs_relax_energy_drop.png",
            "deterministic unrelaxed TFEP work",
            "Delta u_relax",
        ),
        (
            "rmsd_solute_angstrom",
            "rmsd_shell_angstrom",
            "solute_rmsd_vs_shell_rmsd.png",
            "solute RMSD (Angstrom)",
            "shell RMSD (Angstrom)",
        ),
        (
            "clash_count_before",
            "clash_count_after",
            "clash_count_before_after.png",
            "clash count before",
            "clash count after",
        ),
    ]
    for xkey, ykey, filename, xlabel, ylabel in plots:
        x, y = _scatter(records, xkey, ykey)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        if x.size:
            ax.scatter(x, y, s=12, alpha=0.75)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        _maybe_savefig(fig, outdir / filename)
        plt.close(fig)
