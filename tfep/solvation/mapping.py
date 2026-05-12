#!/usr/bin/env python

"""Reusable atom/water-shell mapping utilities for solvated TFEP workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np


def parse_csv_words(text: Optional[str]) -> List[str]:
    """Split comma/space-separated words while preserving order."""
    if text is None:
        return []
    return [x.strip() for x in str(text).replace(",", " ").split() if x.strip()]


def residue_sort_key(residue) -> int:
    """Stable residue ordering key from the smallest atom index."""
    return int(np.min(residue.atoms.indices))


def _atom_resname(atom) -> str:
    return str(atom.residue.resname).strip()


def _atom_name(atom) -> str:
    return str(atom.name).strip()


def _orthorhombic_lengths_from_dimensions(dimensions) -> Optional[np.ndarray]:
    """Return orthorhombic box lengths in Angstrom, else ``None``."""
    if dimensions is None:
        return None
    dims = np.asarray(dimensions, dtype=float).reshape(-1)
    if dims.size < 3 or not np.all(np.isfinite(dims[:3])):
        return None
    if dims.size >= 6:
        angles = dims[3:6]
        if not np.allclose(angles, 90.0, atol=1e-4):
            return None
    return dims[:3]


def minimum_image_displacements(delta: np.ndarray, dimensions) -> np.ndarray:
    """Apply an orthorhombic minimum-image convention to displacement vectors."""
    lengths = _orthorhombic_lengths_from_dimensions(dimensions)
    if lengths is None:
        return delta
    return delta - lengths * np.round(delta / lengths)


def resolve_solute_atom_indices(
    universe,
    *,
    solute_selection: str = "AUTO",
    water_resnames: Sequence[str] = ("HOH", "SOL", "WAT"),
    ion_resnames: Sequence[str] = ("NA", "CL"),
) -> List[int]:
    """Resolve solute atom indices from explicit selection or AUTO mode."""
    selection = str(solute_selection).strip()

    if selection.upper() != "AUTO":
        ag = universe.select_atoms(selection)
        if len(ag) == 0:
            raise ValueError(f"solute selection returned zero atoms: {selection}")
        return [int(i) for i in ag.indices]

    excluded = set(str(x).strip() for x in water_resnames) | set(str(x).strip() for x in ion_resnames)
    indices = [int(atom.index) for atom in universe.atoms if _atom_resname(atom) not in excluded]
    if not indices:
        raise ValueError(
            "AUTO solute detection found zero atoms. Pass solute_selection explicitly, "
            "for example 'resname LIG'."
        )
    return indices


def _resolve_water_residues_and_oxygens(
    universe,
    *,
    water_selection: str,
    oxygen_names: Sequence[str],
) -> Tuple[List[Any], List[int]]:
    water_atoms = universe.select_atoms(str(water_selection))
    if len(water_atoms) == 0:
        raise ValueError(f"Water selection returned 0 atoms: {water_selection!r}")

    residues = list(water_atoms.residues)
    residues.sort(key=residue_sort_key)

    oxygen_idx: List[int] = []
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
        if oxygen is None:
            raise RuntimeError(
                f"Could not find oxygen atom in water residue {residue} using names {list(oxygen_names)}"
            )
        oxygen_idx.append(oxygen)

    return residues, oxygen_idx


def rank_shell_waters_by_occupancy(
    universe,
    *,
    solute_atom_indices: Sequence[int],
    water_selection: str = "water",
    oxygen_names: Sequence[str] = ("O", "OW"),
    cutoff_angstrom: float = 4.5,
    min_occupancy: float = 0.0,
    stride: int = 1,
    frame_indices: Optional[Sequence[int]] = None,
) -> List[Dict[str, Any]]:
    """Rank water residues by shell occupancy around a solute-center definition.

    The shell is defined by oxygen distance to the solute center in each inspected
    frame. Returned records are sorted by descending occupancy then increasing
    minimum distance.
    """
    solute_idx = np.asarray(list(solute_atom_indices), dtype=int).reshape(-1)
    if solute_idx.size == 0:
        raise ValueError("solute_atom_indices is empty")

    residues, oxygen_idx = _resolve_water_residues_and_oxygens(
        universe,
        water_selection=water_selection,
        oxygen_names=oxygen_names,
    )
    oxygen_idx_arr = np.asarray(oxygen_idx, dtype=int)

    if frame_indices is None:
        inspected_frames = list(range(0, len(universe.trajectory), max(1, int(stride))))
    else:
        inspected_frames = [int(i) for i in frame_indices]

    if len(inspected_frames) == 0:
        raise ValueError("No frames selected for occupancy ranking")

    n_waters = len(residues)
    counts = np.zeros(n_waters, dtype=np.int64)
    min_distance = np.full(n_waters, np.inf, dtype=float)

    for frame in inspected_frames:
        universe.trajectory[int(frame)]
        pos = np.asarray(universe.trajectory.ts.positions, dtype=float)
        center = pos[solute_idx].mean(axis=0)

        dvec = pos[oxygen_idx_arr] - center
        dvec = minimum_image_displacements(dvec, universe.trajectory.ts.dimensions)
        dist = np.linalg.norm(dvec, axis=1)

        in_shell = dist <= float(cutoff_angstrom)
        counts[in_shell] += 1
        min_distance = np.minimum(min_distance, dist)

    n_frames = float(len(inspected_frames))
    records: List[Dict[str, Any]] = []
    for i, residue in enumerate(residues):
        occ = float(counts[i]) / n_frames
        if occ < float(min_occupancy):
            continue
        records.append(
            {
                "residue_slot_index": int(i),
                "resid": int(residue.resid),
                "resname": str(residue.resname),
                "occupancy": float(occ),
                "n_hits": int(counts[i]),
                "n_frames": int(n_frames),
                "min_distance_angstrom": float(min_distance[i]),
                "oxygen_atom_index": int(oxygen_idx_arr[i]),
            }
        )

    records.sort(key=lambda r: (-float(r["occupancy"]), float(r["min_distance_angstrom"]), int(r["residue_slot_index"])))
    return records


def merge_shell_rankings(
    state0_records: Sequence[Dict[str, Any]],
    state1_records: Sequence[Dict[str, Any]],
    *,
    max_waters: int,
) -> List[Dict[str, Any]]:
    """Merge two occupancy rankings (state0/state1) into a single candidate list."""
    merged: Dict[int, Dict[str, Any]] = {}

    def _upsert(records: Sequence[Dict[str, Any]], key_occ: str) -> None:
        for row in records:
            resid = int(row["resid"])
            if resid not in merged:
                merged[resid] = {
                    "resid": resid,
                    "resname": str(row.get("resname", "")),
                    "residue_slot_index": int(row.get("residue_slot_index", -1)),
                    "occupancy_state0": 0.0,
                    "occupancy_state1": 0.0,
                    "min_distance_angstrom": float(row.get("min_distance_angstrom", np.inf)),
                }
            merged[resid][key_occ] = float(row.get("occupancy", 0.0))
            merged[resid]["min_distance_angstrom"] = min(
                float(merged[resid]["min_distance_angstrom"]),
                float(row.get("min_distance_angstrom", np.inf)),
            )

    _upsert(state0_records, "occupancy_state0")
    _upsert(state1_records, "occupancy_state1")

    combined = []
    for row in merged.values():
        occ0 = float(row["occupancy_state0"])
        occ1 = float(row["occupancy_state1"])
        row["occupancy_sum"] = occ0 + occ1
        row["occupancy_max"] = max(occ0, occ1)
        combined.append(row)

    combined.sort(
        key=lambda r: (
            -float(r["occupancy_sum"]),
            -float(r["occupancy_max"]),
            float(r["min_distance_angstrom"]),
            int(r["residue_slot_index"]),
        )
    )

    return combined[: int(max_waters)]


def build_shell_equivariant_index_spec(
    topology_path: Union[str, Path],
    *,
    solute_selection: str,
    water_resnames_text: str,
    water_oxygen_names_text: str,
    ion_resnames_text: str,
) -> Dict[str, Any]:
    """Build global/local index arrays for shell-equivariant water flows."""
    try:
        import MDAnalysis as mda
    except Exception as exc:
        raise RuntimeError(
            "The shell-equivariant water flow requires MDAnalysis. "
            "Install it with: conda install -c conda-forge mdanalysis"
        ) from exc

    topology = Path(topology_path).expanduser().resolve()
    if not topology.exists():
        raise FileNotFoundError(f"shell-equivariant topology not found: {topology}")

    universe = mda.Universe(str(topology))

    water_resnames = parse_csv_words(water_resnames_text)
    water_oxygen_names = parse_csv_words(water_oxygen_names_text)
    ion_resnames = parse_csv_words(ion_resnames_text)

    solute_global = resolve_solute_atom_indices(
        universe,
        solute_selection=solute_selection,
        water_resnames=water_resnames,
        ion_resnames=ion_resnames,
    )

    water_resname_set = set(water_resnames)
    oxygen_name_set = set(water_oxygen_names)

    water_atom_global_by_mol: List[List[int]] = []
    water_oxygen_global: List[int] = []
    water_h1_global: List[int] = []
    water_h2_global: List[int] = []
    water_resids: List[int] = []
    water_resnames_found: List[str] = []

    for residue in universe.residues:
        if str(residue.resname).strip() not in water_resname_set:
            continue

        oxygens = [atom for atom in residue.atoms if _atom_name(atom) in oxygen_name_set]
        if len(oxygens) != 1:
            continue

        oxygen_index = int(oxygens[0].index)
        atoms = [int(atom.index) for atom in residue.atoms]

        hydrogens = [
            int(atom.index)
            for atom in residue.atoms
            if int(atom.index) != oxygen_index
            and (_atom_name(atom).upper().startswith("H") or str(getattr(atom, "element", "")).upper() == "H")
        ]
        if len(hydrogens) < 2:
            hydrogens = [int(atom.index) for atom in residue.atoms if int(atom.index) != oxygen_index]
        if len(hydrogens) < 2:
            continue

        water_atom_global_by_mol.append(atoms)
        water_oxygen_global.append(oxygen_index)
        water_h1_global.append(int(hydrogens[0]))
        water_h2_global.append(int(hydrogens[1]))
        water_resids.append(int(residue.resid))
        water_resnames_found.append(str(residue.resname))

    if not water_oxygen_global:
        raise ValueError(
            "shell-equivariant flow found zero water oxygens. "
            f"water_resnames={water_resnames}, water_oxygen_names={water_oxygen_names}. "
            "Check residue and atom names in the topology."
        )

    mapped_global = sorted(
        set(int(i) for i in solute_global)
        | set(int(i) for atoms in water_atom_global_by_mol for i in atoms)
    )
    global_to_local = {int(g): int(i) for i, g in enumerate(mapped_global)}

    solute_local = [global_to_local[int(i)] for i in solute_global]

    water_oxygen_local = []
    water_h1_local = []
    water_h2_local = []
    water_atom_local_flat = []
    water_atom_owner = []

    for iw, (o_global, h1_global, h2_global, atom_globals) in enumerate(
        zip(water_oxygen_global, water_h1_global, water_h2_global, water_atom_global_by_mol)
    ):
        if int(o_global) not in global_to_local:
            continue
        if int(h1_global) not in global_to_local or int(h2_global) not in global_to_local:
            continue

        water_oxygen_local.append(global_to_local[int(o_global)])
        water_h1_local.append(global_to_local[int(h1_global)])
        water_h2_local.append(global_to_local[int(h2_global)])

        for g in atom_globals:
            if int(g) in global_to_local:
                water_atom_local_flat.append(global_to_local[int(g)])
                water_atom_owner.append(int(iw))

    if not water_oxygen_local:
        raise ValueError("shell-equivariant flow found zero usable waters after local index remapping")

    return {
        "topology_path": str(topology),
        "mapped_global_indices": np.asarray(mapped_global, dtype=np.int64),
        "solute_local_indices": np.asarray(solute_local, dtype=np.int64),
        "water_oxygen_local_indices": np.asarray(water_oxygen_local, dtype=np.int64),
        "water_h1_local_indices": np.asarray(water_h1_local, dtype=np.int64),
        "water_h2_local_indices": np.asarray(water_h2_local, dtype=np.int64),
        "water_atom_local_flat_indices": np.asarray(water_atom_local_flat, dtype=np.int64),
        "water_atom_owner_indices": np.asarray(water_atom_owner, dtype=np.int64),
        "n_atoms_total": int(len(universe.atoms)),
        "n_mapped_atoms": int(len(mapped_global)),
        "n_solute_atoms": int(len(solute_local)),
        "n_waters": int(len(water_oxygen_local)),
        "water_resids_first20": water_resids[:20],
        "water_resnames_first20": water_resnames_found[:20],
        "water_resnames": water_resnames,
        "water_oxygen_names": water_oxygen_names,
        "ion_resnames": ion_resnames,
        "solute_selection": str(solute_selection),
    }


def compute_shell_slot_conditioning_indices(
    topology_pdb: Union[str, Path],
    *,
    water_selection: str,
    oxygen_names: Sequence[str],
    k_total: int,
    condition_on: str,
) -> List[int]:
    """Conditioning indices for the first ``k_total`` water residues (topology order)."""
    try:
        import MDAnalysis as mda
    except Exception as exc:
        raise RuntimeError(
            "Shell slot conditioning requires MDAnalysis. "
            "Install it with: conda install -c conda-forge mdanalysis"
        ) from exc

    universe = mda.Universe(str(Path(topology_pdb).expanduser().resolve()))
    water_atoms = universe.select_atoms(str(water_selection))
    residues = list(water_atoms.residues)
    residues.sort(key=residue_sort_key)

    if k_total <= 0:
        return []
    if k_total > len(residues):
        raise ValueError(
            f"k_total={k_total} > n_water_res={len(residues)} for selection {water_selection!r}"
        )

    slot_residues = residues[: int(k_total)]
    idx: List[int] = []

    if condition_on == "molecule":
        for residue in slot_residues:
            idx.extend([int(i) for i in residue.atoms.indices])
    elif condition_on == "oxygen":
        for residue in slot_residues:
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
            if oxygen is None:
                raise RuntimeError(
                    f"Could not find oxygen atom in residue {residue} using names {list(oxygen_names)}"
                )
            idx.append(oxygen)
    else:
        raise ValueError("condition_on must be 'molecule' or 'oxygen'")

    return sorted(set(idx))
