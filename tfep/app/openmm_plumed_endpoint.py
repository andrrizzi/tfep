#!/usr/bin/env python3
"""Run a short OpenMM endpoint trajectory with a PLUMED bias.

This module is intentionally separate from TFEP/TBAR estimators. It produces
biased endpoint trajectories plus aligned PLUMED weight tables that can later be
consumed by the explicit reweighting machinery in the TFEP analysis drivers.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import openmm as mm
import openmm.app as app
import openmm.unit as unit
import yaml
from openmm.app.xtcreporter import XTCReporter

try:  # pragma: no cover - import availability is environment-dependent.
    from openmmplumed import PlumedForce
except Exception as exc:  # pragma: no cover
    PlumedForce = None  # type: ignore[assignment]
    _PLUMED_IMPORT_ERROR = exc
else:
    _PLUMED_IMPORT_ERROR = None


KJ_PER_MOL_PER_K = 0.00831446261815324


@dataclass(frozen=True)
class PlumedSpec:
    text: str
    colvar_path: Path
    method: str
    reweight_kind: str
    reweight_column: str
    reweight_offset_column: str | None
    expected_columns: tuple[str, ...]


def _safe_label(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_]", "_", str(value).strip())
    if not label or not label[0].isalpha():
        label = f"cv_{label}"
    return label


def _topology_atom_lookup(
    pdb_path: Path,
    *,
    residue_name: str,
) -> tuple[dict[str, int], list[int]]:
    pdb = app.PDBFile(str(pdb_path))
    solute: dict[str, int] = {}
    water_oxygens: list[int] = []
    for atom in pdb.topology.atoms():
        idx = atom.index + 1
        if atom.residue.name == residue_name:
            if atom.name in solute:
                raise ValueError(f"Duplicate atom name {atom.name!r} in residue {residue_name!r}")
            solute[atom.name] = idx
        elif atom.residue.name in {"HOH", "WAT", "SOL"} and atom.element is not None and atom.element.symbol == "O":
            water_oxygens.append(idx)
    if not solute:
        raise ValueError(f"No atoms found for residue {residue_name!r} in {pdb_path}")
    return solute, water_oxygens


def _named_indices(names: Sequence[str], atom_lookup: dict[str, int], *, label: str) -> list[int]:
    missing = [str(name) for name in names if str(name) not in atom_lookup]
    if missing:
        raise ValueError(f"{label}: atom names not present in endpoint topology: {missing}")
    return [int(atom_lookup[str(name)]) for name in names]


def load_cv_spec(path: Path) -> dict[str, Any]:
    with path.expanduser().resolve().open("r", encoding="utf-8") as handle:
        spec = yaml.safe_load(handle)
    if not isinstance(spec, dict):
        raise ValueError(f"CV specification must be a mapping: {path}")
    if int(spec.get("schema_version", 0)) != 1:
        raise ValueError(f"Unsupported CV specification schema_version in {path}")
    cvs = spec.get("cvs", [])
    if not isinstance(cvs, list) or len(cvs) > 3:
        raise ValueError("CV specification must contain a list of at most three biased CVs")
    return spec


def build_generic_plumed_spec(
    *,
    method: str,
    stage_dir: Path,
    cv_spec: dict[str, Any],
    pdb_path: Path,
    residue_name: str,
    temperature_k: float,
    stride: int,
    opes_barrier_kj_mol: float,
    metad_height_kj_mol: float,
    metad_biasfactor: float,
    restart: bool = False,
) -> PlumedSpec:
    """Render a validated, molecule-specific CV specification to PLUMED."""
    method = str(method).lower().strip()
    atom_lookup, water_oxygens = _topology_atom_lookup(pdb_path, residue_name=residue_name)
    lines = ["UNITS LENGTH=nm ENERGY=kj/mol TIME=ps"]
    if restart:
        lines.append("RESTART")

    labels: list[str] = []
    sigmas: list[float] = []
    grid_min: list[str] = []
    grid_max: list[str] = []
    grid_bin: list[int] = []
    for raw_cv in cv_spec.get("cvs", []):
        if not isinstance(raw_cv, dict):
            raise ValueError("Each CV entry must be a mapping")
        label = _safe_label(str(raw_cv.get("label", "")))
        kind = str(raw_cv.get("type", "")).strip().lower()
        names = [str(x) for x in raw_cv.get("atom_names", [])]
        if kind == "distance":
            atoms = _named_indices(names, atom_lookup, label=label)
            if len(atoms) != 2:
                raise ValueError(f"{label}: distance requires exactly two atom names")
            lines.append(f"{label}: DISTANCE ATOMS={atoms[0]},{atoms[1]}")
        elif kind == "gyration":
            atoms = _named_indices(names, atom_lookup, label=label)
            if len(atoms) < 2:
                raise ValueError(f"{label}: gyration requires at least two atom names")
            lines.append(f"{label}: GYRATION TYPE=RADIUS ATOMS={','.join(map(str, atoms))}")
        elif kind == "torsion":
            atoms = _named_indices(names, atom_lookup, label=label)
            if len(atoms) != 4:
                raise ValueError(f"{label}: torsion requires exactly four atom names")
            lines.append(f"{label}: TORSION ATOMS={','.join(map(str, atoms))}")
        elif kind == "torsion_order":
            torsions = raw_cv.get("torsions", [])
            if not isinstance(torsions, list) or not torsions:
                raise ValueError(f"{label}: torsion_order requires a nonempty torsions list")
            terms: list[str] = []
            for i, torsion_names in enumerate(torsions, start=1):
                atoms = _named_indices([str(x) for x in torsion_names], atom_lookup, label=label)
                if len(atoms) != 4:
                    raise ValueError(f"{label}: every torsion_order component requires four atoms")
                torsion_label = f"{label}_t{i}"
                term_label = f"{label}_g{i}"
                lines.append(f"{torsion_label}: TORSION ATOMS={','.join(map(str, atoms))}")
                lines.append(f"{term_label}: CUSTOM ARG={torsion_label} FUNC=(1+cos(x))/2 PERIODIC=NO")
                terms.append(term_label)
            lines.append(
                f"{label}: COMBINE ARG={','.join(terms)} "
                f"COEFFICIENTS={','.join('1' for _ in terms)} "
                f"PARAMETERS={','.join('0' for _ in terms)} "
                f"POWERS={','.join('1' for _ in terms)} PERIODIC=NO"
            )
        elif kind == "coordination":
            atoms = _named_indices(names, atom_lookup, label=label)
            if not atoms or not water_oxygens:
                raise ValueError(f"{label}: coordination requires solute atoms and solvent water oxygens")
            r0 = float(raw_cv.get("r0_nm", 0.35))
            nn = int(raw_cv.get("nn", 6))
            mm = int(raw_cv.get("mm", 12))
            lines.append(
                f"{label}: COORDINATION GROUPA={','.join(map(str, atoms))} "
                f"GROUPB={','.join(map(str, water_oxygens))} R_0={r0:.8g} NN={nn} MM={mm} NLIST "
                "NL_CUTOFF=0.8 NL_STRIDE=20"
            )
        else:
            raise ValueError(f"Unsupported CV type {kind!r} for {label}")

        sigma = float(raw_cv["sigma"])
        lo = float(raw_cv["grid_min"])
        hi = float(raw_cv["grid_max"])
        bins = int(raw_cv.get("grid_bin", 80))
        if not (math.isfinite(sigma) and sigma > 0.0 and math.isfinite(lo) and math.isfinite(hi) and hi > lo):
            raise ValueError(f"{label}: invalid sigma or grid bounds")
        labels.append(label)
        sigmas.append(sigma)
        if bool(raw_cv.get("periodic", False)):
            grid_min.append("-pi")
            grid_max.append("pi")
        else:
            grid_min.append(f"{lo:.8g}")
            grid_max.append(f"{hi:.8g}")
        grid_bin.append(bins)

    if not labels:
        raise ValueError("WTMetaD/OPES CV specification contains no biased CVs")
    if math.prod(grid_bin) > int(cv_spec.get("max_grid_points", 1_000_000)):
        raise ValueError(f"CV grid has {math.prod(grid_bin)} points, above the configured limit")

    colvar = stage_dir / "COLVAR"
    hills = stage_dir / "HILLS"
    kernels = stage_dir / "KERNELS"
    state = stage_dir / "OPES_STATE"
    args_csv = ",".join(labels)
    sigma_csv = ",".join(f"{x:.8g}" for x in sigmas)
    if method == "wtmetad":
        restart_kw = " RESTART=YES" if restart else ""
        lines.append(
            "metad: METAD "
            f"ARG={args_csv} SIGMA={sigma_csv} HEIGHT={float(metad_height_kj_mol):.8g} "
            f"PACE={int(stride)} BIASFACTOR={float(metad_biasfactor):.8g} TEMP={float(temperature_k):.8g} "
            f"FILE={hills} CALC_RCT GRID_MIN={','.join(grid_min)} "
            f"GRID_MAX={','.join(grid_max)} GRID_BIN={','.join(map(str, grid_bin))}{restart_kw}"
        )
        bias_args = "metad.bias,metad.rct,metad.rbias"
        reweight_kind, reweight_column, reweight_offset = "rbias", "metad.rbias", None
    elif method == "opes":
        lines.append(
            "opes: OPES_METAD "
            f"ARG={args_csv} PACE={int(stride)} TEMP={float(temperature_k):.8g} "
            f"BARRIER={float(opes_barrier_kj_mol):.8g} FILE={kernels} "
            f"STATE_WFILE={state} STATE_WSTRIDE={int(stride)}"
        )
        bias_args = "opes.bias,opes.rct,opes.rbias"
        reweight_kind, reweight_column, reweight_offset = "bias_minus_offset", "opes.bias", "opes.rct"
    else:
        raise ValueError("Generic biased CV specifications require method='wtmetad' or method='opes'")
    restart_kw = " RESTART=YES" if restart else ""
    lines.append(f"PRINT STRIDE={int(stride)} ARG={args_csv},{bias_args} FILE={colvar}{restart_kw}")
    lines.append("FLUSH STRIDE=1")
    return PlumedSpec(
        text="\n".join(lines) + "\n",
        colvar_path=colvar,
        method=method,
        reweight_kind=reweight_kind,
        reweight_column=reweight_column,
        reweight_offset_column=reweight_offset,
        expected_columns=tuple(["time", *labels, *bias_args.split(",")]),
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def load_system(system_xml: Path) -> mm.System:
    return mm.XmlSerializer.deserialize(system_xml.read_text())


def add_barostat(system: mm.System, pressure_bar: float, temperature_K: float, frequency: int) -> None:
    for force in system.getForces():
        if isinstance(force, mm.MonteCarloBarostat):
            return
    system.addForce(mm.MonteCarloBarostat(pressure_bar * unit.bar, temperature_K * unit.kelvin, int(frequency)))


def link_or_copy(src: Path, dst: Path, *, copy: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def prepare_endpoint_layout(source_endpoint: Path, outdir: Path, *, copy_inputs: bool = False) -> None:
    system_dir = outdir / "system"
    system_dir.mkdir(parents=True, exist_ok=True)
    link_or_copy(source_endpoint / "system" / "system.xml", system_dir / "system.xml", copy=copy_inputs)
    link_or_copy(source_endpoint / "system" / "start.pdb", system_dir / "start.pdb", copy=copy_inputs)
    if (source_endpoint / "stages.json").is_file():
        link_or_copy(source_endpoint / "stages.json", outdir / "stages.json", copy=copy_inputs)


def find_equilibrated_state(source_endpoint: Path) -> Path:
    for rel in ("02_npt/state.xml", "01_nvt/state.xml", "03_md/state.xml"):
        path = source_endpoint / rel
        if path.is_file():
            return path
    raise FileNotFoundError(f"No equilibrated state.xml found under {source_endpoint}")


def load_md_stage(source_endpoint: Path) -> dict[str, Any]:
    stages_path = source_endpoint / "stages.json"
    if not stages_path.is_file():
        return {}
    stages = json.loads(stages_path.read_text())
    return dict(stages.get("md", {}))


def decane_carbon_indices(pdb_path: Path, *, residue_name: str = "UNL") -> list[int]:
    pdb = app.PDBFile(str(pdb_path))
    carbons: list[tuple[int, int]] = []
    pattern = re.compile(r"^C(\d+)$")
    for atom in pdb.topology.atoms():
        if atom.residue.name != residue_name:
            continue
        match = pattern.match(atom.name.strip())
        if match:
            carbons.append((int(match.group(1)), atom.index + 1))  # PLUMED is 1-based.
    carbons.sort()
    if len(carbons) < 10:
        raise ValueError(f"Expected at least 10 decane carbon atoms in {pdb_path}, found {len(carbons)}")
    return [idx for _, idx in carbons[:10]]


def named_atom_indices(pdb_path: Path, names: Sequence[str], *, residue_name: str = "UNL") -> list[int]:
    """Return PLUMED 1-based atom indices by atom name.

    The small-molecule FreeSolv endpoints use stable atom names across force
    fields.  Keeping the lookup name-based makes glucose/pyranose CVs explicit
    while avoiding global assumptions about atom ordering.
    """
    pdb = app.PDBFile(str(pdb_path))
    wanted = [str(name).strip() for name in names]
    found: dict[str, int] = {}
    for atom in pdb.topology.atoms():
        if atom.residue.name != residue_name:
            continue
        name = atom.name.strip()
        if name in wanted:
            found[name] = atom.index + 1  # PLUMED is 1-based.
    missing = [name for name in wanted if name not in found]
    if missing:
        raise ValueError(f"Missing atom name(s) {missing} in residue {residue_name!r} of {pdb_path}")
    return [found[name] for name in wanted]


def _torsion_labels(carbon_atoms: Sequence[int]) -> list[tuple[str, tuple[int, int, int, int]]]:
    labels: list[tuple[str, tuple[int, int, int, int]]] = []
    for i in range(0, len(carbon_atoms) - 3):
        labels.append((f"t{i + 1}", tuple(int(x) for x in carbon_atoms[i:i + 4])))
    return labels


def build_glucose_pyranose_plumed_spec(
    *,
    method: str,
    stage_dir: Path,
    atom_lookup: dict[str, int],
    temperature_k: float,
    stride: int,
    opes_barrier_kj_mol: float,
    metad_height_kj_mol: float,
    metad_biasfactor: float,
    sigma_ring_rg_nm: float,
    sigma_exocyclic_radians: float,
    restart: bool = False,
) -> PlumedSpec:
    """Build a conservative WTMetaD/OPES spec for glucose-like pyranose.

    The biased variables are intentionally low-dimensional:

    * ``ring_rg``: radius of gyration of the six-member pyranose ring
      (O1-C2-C3-C4-C5-C6 in this endpoint naming).
    * ``exo``: exocyclic hydroxymethyl torsion O1-C2-C1-O6.

    Additional ring and hydroxyl torsions are printed as diagnostics only so
    we can check whether residual work outliers remain coupled to un-biased
    slow modes without immediately destroying reweighting ESS.
    """
    method = method.lower().strip()
    colvar = stage_dir / "COLVAR"
    kernels = stage_dir / "KERNELS"
    hills = stage_dir / "HILLS"
    state = stage_dir / "OPES_STATE"

    def idx(name: str) -> int:
        return int(atom_lookup[name])

    ring_atoms = [idx(name) for name in ("O1", "C2", "C3", "C4", "C5", "C6")]
    ring_csv = ",".join(str(x) for x in ring_atoms)
    torsions: list[tuple[str, tuple[int, int, int, int]]] = [
        ("r1", (idx("O1"), idx("C2"), idx("C3"), idx("C4"))),
        ("r2", (idx("C2"), idx("C3"), idx("C4"), idx("C5"))),
        ("r3", (idx("C3"), idx("C4"), idx("C5"), idx("C6"))),
        ("r4", (idx("C4"), idx("C5"), idx("C6"), idx("O1"))),
        ("r5", (idx("C5"), idx("C6"), idx("O1"), idx("C2"))),
        ("r6", (idx("C6"), idx("O1"), idx("C2"), idx("C3"))),
        ("exo", (idx("O1"), idx("C2"), idx("C1"), idx("O6"))),
        ("oh1", (idx("C2"), idx("C1"), idx("O6"), idx("H12"))),
        ("oh2", (idx("C5"), idx("C6"), idx("O2"), idx("H8"))),
        ("oh3", (idx("C4"), idx("C5"), idx("O3"), idx("H9"))),
        ("oh4", (idx("C3"), idx("C4"), idx("O4"), idx("H10"))),
        ("oh5", (idx("C2"), idx("C3"), idx("O5"), idx("H11"))),
    ]

    lines = ["UNITS LENGTH=nm ENERGY=kj/mol TIME=ps"]
    if restart:
        lines.append("RESTART")
    lines.append(f"ring_rg: GYRATION TYPE=RADIUS ATOMS={ring_csv}")
    for label, atoms in torsions:
        lines.append(f"{label}: TORSION ATOMS={','.join(str(x) for x in atoms)}")

    biased_arg_csv = "ring_rg,exo"
    sigma_csv = f"{float(sigma_ring_rg_nm):.8g},{float(sigma_exocyclic_radians):.8g}"
    # ``exo`` is periodic, but PLUMED can still build a finite grid for METAD.
    grid_min_csv = "0.10,-pi"
    grid_max_csv = "0.45,pi"
    grid_bin_csv = "100,120"
    print_args = ["ring_rg"] + [label for label, _ in torsions]
    common_args = ",".join(print_args)

    if method == "opes":
        lines.append(
            "opes: OPES_METAD "
            f"ARG={biased_arg_csv} "
            f"PACE={int(stride)} TEMP={float(temperature_k):.8g} "
            f"BARRIER={float(opes_barrier_kj_mol):.8g} "
            f"FILE={kernels} STATE_WFILE={state} STATE_WSTRIDE={int(stride)}"
        )
        bias_args = "opes.bias,opes.rct,opes.rbias"
        reweight_kind = "bias_minus_offset"
        reweight_column = "opes.bias"
        reweight_offset = "opes.rct"
    elif method == "wtmetad":
        restart_kw = " RESTART=YES" if restart else ""
        lines.append(
            "metad: METAD "
            f"ARG={biased_arg_csv} "
            f"SIGMA={sigma_csv} "
            f"HEIGHT={float(metad_height_kj_mol):.8g} "
            f"PACE={int(stride)} BIASFACTOR={float(metad_biasfactor):.8g} "
            f"TEMP={float(temperature_k):.8g} FILE={hills} CALC_RCT "
            f"GRID_MIN={grid_min_csv} GRID_MAX={grid_max_csv} GRID_BIN={grid_bin_csv}"
            f"{restart_kw}"
        )
        bias_args = "metad.bias,metad.rct,metad.rbias"
        reweight_kind = "rbias"
        reweight_column = "metad.rbias"
        reweight_offset = None
    else:
        raise ValueError("method must be one of: opes, wtmetad")

    print_restart_kw = " RESTART=YES" if restart else ""
    lines.append(f"PRINT STRIDE={int(stride)} ARG={common_args},{bias_args} FILE={colvar}{print_restart_kw}")
    lines.append("FLUSH STRIDE=1")
    expected = tuple(["time"] + print_args + bias_args.split(","))
    return PlumedSpec(
        text="\n".join(lines) + "\n",
        colvar_path=colvar,
        method=method,
        reweight_kind=reweight_kind,
        reweight_column=reweight_column,
        reweight_offset_column=reweight_offset,
        expected_columns=expected,
    )


def build_plumed_spec(
    *,
    method: str,
    stage_dir: Path,
    carbon_atoms: Sequence[int],
    temperature_k: float,
    stride: int,
    opes_barrier_kj_mol: float,
    metad_height_kj_mol: float,
    metad_biasfactor: float,
    sigma_distance_nm: float,
    sigma_rg_nm: float,
    metad_include_torsion_order: bool = False,
    sigma_torsion_order: float = 0.6,
    restart: bool = False,
) -> PlumedSpec:
    method = method.lower().strip()
    colvar = stage_dir / "COLVAR"
    kernels = stage_dir / "KERNELS"
    hills = stage_dir / "HILLS"
    state = stage_dir / "OPES_STATE"
    c1 = int(carbon_atoms[0])
    c10 = int(carbon_atoms[-1])
    carbon_csv = ",".join(str(x) for x in carbon_atoms)
    torsions = _torsion_labels(carbon_atoms)

    lines = ["UNITS LENGTH=nm ENERGY=kj/mol TIME=ps"]
    if restart:
        lines.append("RESTART")
    lines.extend([
        f"d_end: DISTANCE ATOMS={c1},{c10}",
        f"rg: GYRATION TYPE=RADIUS ATOMS={carbon_csv}",
    ])
    for label, atoms in torsions:
        lines.append(f"{label}: TORSION ATOMS={','.join(str(x) for x in atoms)}")

    biased_args = ["d_end", "rg"]
    biased_sigmas = [float(sigma_distance_nm), float(sigma_rg_nm)]
    grid_min = ["0.2", "0.10"]
    grid_max = ["1.45", "0.65"]
    grid_bin = ["150", "120"]
    if metad_include_torsion_order:
        compact_terms: list[str] = []
        for label, _ in torsions:
            compact_label = f"g_{label}"
            lines.append(f"{compact_label}: CUSTOM ARG={label} FUNC=(1+cos(x))/2 PERIODIC=NO")
            compact_terms.append(compact_label)
        if compact_terms:
            arg_csv = ",".join(compact_terms)
            coeff_csv = ",".join("1" for _ in compact_terms)
            param_csv = ",".join("0" for _ in compact_terms)
            power_csv = ",".join("1" for _ in compact_terms)
            lines.append(
                "ncompact: COMBINE "
                f"ARG={arg_csv} COEFFICIENTS={coeff_csv} PARAMETERS={param_csv} "
                f"POWERS={power_csv} PERIODIC=NO"
            )
            biased_args.append("ncompact")
            biased_sigmas.append(float(sigma_torsion_order))
            grid_min.append("0.0")
            grid_max.append(str(float(len(compact_terms))))
            # Keep the 3D grid moderate. This is a pilot diagnostic CV, not a
            # high-resolution final FES calculation.
            grid_bin = ["100", "70", "70"]

    torsion_args = ",".join(label for label, _ in torsions)
    print_args = ["d_end", "rg"]
    if "ncompact" in biased_args:
        print_args.append("ncompact")
    print_args.extend(label for label, _ in torsions)
    common_args = ",".join(print_args)
    biased_arg_csv = ",".join(biased_args)
    sigma_csv = ",".join(f"{x:.8g}" for x in biased_sigmas)
    grid_min_csv = ",".join(grid_min)
    grid_max_csv = ",".join(grid_max)
    grid_bin_csv = ",".join(grid_bin)

    if method == "opes":
        lines.append(
            "opes: OPES_METAD "
            f"ARG={biased_arg_csv} "
            f"PACE={int(stride)} TEMP={float(temperature_k):.8g} "
            f"BARRIER={float(opes_barrier_kj_mol):.8g} "
            f"FILE={kernels} STATE_WFILE={state} STATE_WSTRIDE={int(stride)}"
        )
        # OPES component names vary slightly by PLUMED version. We request the
        # common columns and let the preflight/run fail clearly if unsupported.
        bias_args = "opes.bias,opes.rct,opes.rbias"
        reweight_kind = "bias_minus_offset"
        reweight_column = "opes.bias"
        reweight_offset = "opes.rct"
    elif method == "wtmetad":
        restart_kw = " RESTART=YES" if restart else ""
        lines.append(
            "metad: METAD "
            f"ARG={biased_arg_csv} "
            f"SIGMA={sigma_csv} "
            f"HEIGHT={float(metad_height_kj_mol):.8g} "
            f"PACE={int(stride)} BIASFACTOR={float(metad_biasfactor):.8g} "
            f"TEMP={float(temperature_k):.8g} FILE={hills} CALC_RCT "
            f"GRID_MIN={grid_min_csv} GRID_MAX={grid_max_csv} GRID_BIN={grid_bin_csv}"
            f"{restart_kw}"
        )
        bias_args = "metad.bias,metad.rct,metad.rbias"
        reweight_kind = "rbias"
        reweight_column = "metad.rbias"
        reweight_offset = None
    else:
        raise ValueError("method must be one of: opes, wtmetad")

    print_restart_kw = " RESTART=YES" if restart else ""
    lines.append(f"PRINT STRIDE={int(stride)} ARG={common_args},{bias_args} FILE={colvar}{print_restart_kw}")
    lines.append("FLUSH STRIDE=1")
    expected = tuple(["time"] + print_args + bias_args.split(","))
    return PlumedSpec(
        text="\n".join(lines) + "\n",
        colvar_path=colvar,
        method=method,
        reweight_kind=reweight_kind,
        reweight_column=reweight_column,
        reweight_offset_column=reweight_offset,
        expected_columns=expected,
    )


def available_plumed_actions(plumed_binary: str = "plumed") -> set[str]:
    try:
        proc = subprocess.run([plumed_binary, "manual"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    except FileNotFoundError:
        return set()
    actions: set[str] = set()
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if stripped and re.match(r"^[A-Z][A-Z0-9_]+$", stripped):
            actions.add(stripped)
    return actions


def check_method_available(method: str, plumed_binary: str = "plumed") -> tuple[bool, str]:
    if str(method).lower() == "unbiased":
        return True, "Unbiased endpoint production does not require a PLUMED bias action"
    actions = available_plumed_actions(plumed_binary)
    if not actions:
        return False, f"No PLUMED actions discovered with {plumed_binary!r}"
    required = "OPES_METAD" if method == "opes" else "METAD"
    if required not in actions:
        return False, f"PLUMED action {required} is unavailable in {plumed_binary!r}"
    return True, f"PLUMED action {required} is available"


def read_table_fields(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#! FIELDS"):
                return line.split()[2:]
    raise ValueError(f"No '#! FIELDS' header found in {path}")


def read_table_rows(path: Path) -> list[str]:
    rows: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#!") or not line.strip():
                continue
            rows.append(line.rstrip("\n"))
    return rows


def count_xtc_frames(topology_path: Path, traj_path: Path, expected: int) -> int:
    try:
        import MDAnalysis as mda
    except Exception:
        return int(expected)
    try:
        universe = mda.Universe(str(topology_path), str(traj_path))
        return int(len(universe.trajectory))
    except Exception:
        return int(expected)


def align_colvar_to_trajectory(colvar: Path, aligned: Path, *, n_frames: int) -> dict[str, Any]:
    fields = read_table_fields(colvar)
    rows = read_table_rows(colvar)
    dropped_first = False
    dropped_initial_rows = 0
    if len(rows) > n_frames and len(rows) - n_frames <= 5:
        dropped_initial_rows = len(rows) - n_frames
        rows = rows[dropped_initial_rows:]
        dropped_first = bool(dropped_initial_rows)
    if len(rows) != n_frames:
        raise ValueError(f"Cannot align {colvar}: {len(rows)} COLVAR rows for {n_frames} trajectory frames")
    aligned.parent.mkdir(parents=True, exist_ok=True)
    with aligned.open("w", encoding="utf-8") as handle:
        handle.write("#! FIELDS " + " ".join(fields) + "\n")
        for row in rows:
            handle.write(row + "\n")
    return {
        "fields": fields,
        "n_rows_in": len(read_table_rows(colvar)),
        "n_rows_out": len(rows),
        "dropped_first_row": dropped_first,
        "dropped_initial_rows": int(dropped_initial_rows),
    }


def log_weight_diagnostics_from_colvar(
    aligned_colvar: Path,
    *,
    column: str,
    offset_column: str | None,
    kind: str,
    temperature_k: float,
) -> dict[str, Any]:
    fields = read_table_fields(aligned_colvar)
    data = []
    with aligned_colvar.open("r", encoding="utf-8") as handle:
        reader = csv.reader((line for line in handle if not line.startswith("#!") and line.strip()), delimiter=" ")
        for row in reader:
            data.append([float(x) for x in row if x != ""])
    arr = list(zip(*data)) if data else []
    col_map = {name: i for i, name in enumerate(fields)}
    if column not in col_map:
        raise ValueError(f"Column {column!r} not found in {aligned_colvar}; fields={fields}")
    values = [float(x) for x in arr[col_map[column]]]
    if kind == "rbias" or kind == "bias":
        logw = [v / (KJ_PER_MOL_PER_K * float(temperature_k)) for v in values]
    elif kind == "bias_minus_offset":
        if offset_column is None or offset_column not in col_map:
            raise ValueError(f"Offset column {offset_column!r} missing from {aligned_colvar}")
        offsets = [float(x) for x in arr[col_map[offset_column]]]
        kt = KJ_PER_MOL_PER_K * float(temperature_k)
        logw = [(v - o) / kt for v, o in zip(values, offsets)]
    else:
        logw = values
    finite = [x for x in logw if math.isfinite(x)]
    if not finite:
        return {"n": len(logw), "n_finite": 0, "ess": math.nan, "ess_ratio": math.nan, "log_weight_span": math.nan}
    max_logw = max(finite)
    weights = [math.exp(x - max_logw) for x in finite]
    norm = sum(weights)
    weights = [w / norm for w in weights]
    ess = 1.0 / sum(w * w for w in weights)
    return {
        "n": len(logw),
        "n_finite": len(finite),
        "ess": ess,
        "ess_ratio": ess / len(finite),
        "log_weight_min": min(finite),
        "log_weight_max": max(finite),
        "log_weight_span": max(finite) - min(finite),
    }


def run_biased_md(args: argparse.Namespace) -> dict[str, Any]:
    unbiased = str(args.method).lower() == "unbiased"
    if not unbiased and PlumedForce is None:
        raise RuntimeError(f"openmmplumed is not importable: {_PLUMED_IMPORT_ERROR!r}")

    source_endpoint = Path(args.source_endpoint_dir).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    stage_dir = outdir / "03_md"
    stage_dir.mkdir(parents=True, exist_ok=True)
    prepare_endpoint_layout(source_endpoint, outdir, copy_inputs=bool(args.copy_inputs))

    start_pdb = outdir / "system" / "start.pdb"
    system_xml = outdir / "system" / "system.xml"
    md_stage = load_md_stage(source_endpoint)
    temperature_k = float(args.temperature_k if args.temperature_k is not None else md_stage.get("temperature_K", 298.15))
    dt_ps = float(args.dt_ps if args.dt_ps is not None else md_stage.get("dt_ps", 0.004))
    friction_ps = float(args.friction_ps if args.friction_ps is not None else md_stage.get("friction_inv_ps", 1.0))
    constraint_tolerance = float(md_stage.get("constraint_tolerance", 1.0e-6))

    if unbiased:
        msg = "Unbiased endpoint production; PLUMED action check is not applicable"
    elif bool(args.skip_plumed_action_check):
        msg = "PLUMED action check skipped by configuration; runtime context creation remains authoritative"
    else:
        ok, msg = check_method_available(args.method, args.plumed_binary)
        if not ok:
            raise RuntimeError(msg)

    cv_mode = "spec" if args.cv_spec else str(args.cv_mode).strip().lower()
    carbon_atoms: list[int] | None = None
    glucose_atom_lookup: dict[str, int] | None = None
    cv_spec_data: dict[str, Any] | None = None
    spec: PlumedSpec | None = None
    biased_cv_note = "Unbiased endpoint production with dimensionless unit weights."
    if unbiased:
        pass
    elif args.cv_spec:
        cv_spec_path = Path(args.cv_spec).expanduser().resolve()
        cv_spec_data = load_cv_spec(cv_spec_path)
        spec = build_generic_plumed_spec(
            method=args.method,
            stage_dir=stage_dir,
            cv_spec=cv_spec_data,
            pdb_path=start_pdb,
            residue_name=args.residue_name,
            temperature_k=temperature_k,
            stride=int(args.colvar_every),
            opes_barrier_kj_mol=float(args.opes_barrier_kj_mol),
            metad_height_kj_mol=float(args.metad_height_kj_mol),
            metad_biasfactor=float(args.metad_biasfactor),
            restart=bool(args.plumed_restart),
        )
        biased_cv_note = str(cv_spec_data.get("decision", {}).get("rationale", "Molecule-specific generated CV specification."))
    elif cv_mode == "decane":
        carbon_atoms = decane_carbon_indices(start_pdb, residue_name=args.residue_name)
        spec = build_plumed_spec(
            method=args.method,
            stage_dir=stage_dir,
            carbon_atoms=carbon_atoms,
            temperature_k=temperature_k,
            stride=int(args.colvar_every),
            opes_barrier_kj_mol=float(args.opes_barrier_kj_mol),
            metad_height_kj_mol=float(args.metad_height_kj_mol),
            metad_biasfactor=float(args.metad_biasfactor),
            sigma_distance_nm=float(args.sigma_distance_nm),
            sigma_rg_nm=float(args.sigma_rg_nm),
            metad_include_torsion_order=bool(args.metad_include_torsion_order),
            sigma_torsion_order=float(args.sigma_torsion_order),
            restart=bool(args.plumed_restart),
        )
        biased_cv_note = (
            "WTMetaD/OPES bias acts on d_end and rg plus ncompact when "
            "metad_include_torsion_order is true. ncompact is sum((1+cos(phi_i))/2) "
            "over the seven decane carbon-chain torsions."
        )
    elif cv_mode in {"glucose-pyranose", "glucose_pyranose"}:
        glucose_names = ("C1", "C2", "C3", "C4", "C5", "C6", "O1", "O2", "O3", "O4", "O5", "O6", "H8", "H9", "H10", "H11", "H12")
        glucose_atom_lookup = dict(
            zip(
                glucose_names,
                named_atom_indices(start_pdb, glucose_names, residue_name=args.residue_name),
            )
        )
        spec = build_glucose_pyranose_plumed_spec(
            method=args.method,
            stage_dir=stage_dir,
            atom_lookup=glucose_atom_lookup,
            temperature_k=temperature_k,
            stride=int(args.colvar_every),
            opes_barrier_kj_mol=float(args.opes_barrier_kj_mol),
            metad_height_kj_mol=float(args.metad_height_kj_mol),
            metad_biasfactor=float(args.metad_biasfactor),
            sigma_ring_rg_nm=float(args.sigma_glucose_ring_rg_nm),
            sigma_exocyclic_radians=float(args.sigma_glucose_exocyclic_radians),
            restart=bool(args.plumed_restart),
        )
        biased_cv_note = (
            "WTMetaD/OPES bias acts on glucose-like pyranose ring_rg "
            "(O1-C2-C3-C4-C5-C6 radius of gyration) and exo "
            "(O1-C2-C1-O6 torsion). Ring and hydroxyl torsions are printed "
            "as diagnostics only."
        )
    else:
        raise ValueError("cv_mode must be one of: decane, glucose-pyranose")
    plumed_path = stage_dir / "plumed.dat"
    if spec is not None:
        plumed_path.write_text(spec.text, encoding="utf-8")

    if args.dry_run:
        return {
            "dry_run": True,
            "plumed_dat": str(plumed_path) if spec is not None else None,
            "message": msg,
            "cv_mode": cv_mode,
            "carbon_atoms": carbon_atoms,
            "glucose_atom_lookup": glucose_atom_lookup,
            "plumed_text": spec.text if spec is not None else None,
            "cv_spec": cv_spec_data,
        }

    if not args.overwrite and (stage_dir / "state.xml").is_file() and (stage_dir / "COLVAR_aligned.dat").is_file():
        return {"skipped_complete": True, "outdir": str(outdir)}

    system = load_system(system_xml)
    if bool(md_stage.get("add_barostat", False)):
        add_barostat(
            system,
            pressure_bar=float(md_stage.get("pressure_bar", 1.0)),
            temperature_K=temperature_k,
            frequency=int(md_stage.get("barostat_frequency", 25)),
        )
    if spec is not None:
        system.addForce(PlumedForce(spec.text))

    pdb = app.PDBFile(str(start_pdb))
    topology = pdb.topology
    integrator = mm.LangevinMiddleIntegrator(
        temperature_k * unit.kelvin,
        friction_ps / unit.picosecond,
        dt_ps * unit.picoseconds,
    )
    integrator.setRandomNumberSeed(int(args.seed))
    integrator.setConstraintTolerance(constraint_tolerance)

    platform = mm.Platform.getPlatformByName(str(args.platform))
    props: dict[str, str] = {}
    if str(args.platform).upper() == "CUDA":
        props["DeviceIndex"] = str(args.device)
        props["CudaPrecision"] = str(args.precision)
    elif str(args.platform).upper() == "OPENCL":
        props["OpenCLDeviceIndex"] = str(args.device)
        props["OpenCLPrecision"] = str(args.precision)

    sim = app.Simulation(topology, system, integrator, platform, props)
    source_state = mm.XmlSerializer.deserialize(find_equilibrated_state(source_endpoint).read_text())
    try:
        sim.context.setPeriodicBoxVectors(*source_state.getPeriodicBoxVectors())
    except Exception:
        pass
    sim.context.setPositions(source_state.getPositions())
    try:
        sim.context.setVelocities(source_state.getVelocities())
    except Exception:
        sim.context.setVelocitiesToTemperature(temperature_k * unit.kelvin, int(args.seed))

    ckpt = stage_dir / "checkpoint.chk"
    if args.resume and ckpt.is_file():
        sim.loadCheckpoint(str(ckpt))

    sim.reporters.append(XTCReporter(str(stage_dir / "traj.xtc"), int(args.traj_every), append=bool(args.append_output)))
    sim.reporters.append(
        app.StateDataReporter(
            str(stage_dir / "log.txt"),
            int(args.log_every),
            step=True,
            time=True,
            potentialEnergy=True,
            kineticEnergy=True,
            totalEnergy=True,
            temperature=True,
            volume=True,
            density=True,
            speed=True,
            separator="\t",
            append=bool(args.append_output),
        )
    )
    sim.reporters.append(app.CheckpointReporter(str(ckpt), int(args.ckpt_every)))
    start_step = int(sim.currentStep)
    requested_steps = int(args.nsteps)
    existing_frames = 0
    trajectory_path = stage_dir / "traj.xtc"
    if args.resume and trajectory_path.is_file():
        existing_frames = count_xtc_frames(start_pdb, trajectory_path, 0)
    if args.target_total_steps:
        target_frames = requested_steps // int(args.traj_every)
        if existing_frames > target_frames:
            raise ValueError(
                f"Existing trajectory has {existing_frames} frames, above target {target_frames}"
            )
        # Target the remaining output count. This avoids excess frames when a
        # periodic checkpoint trails the most recently reported trajectory frame.
        steps_to_run = (target_frames - existing_frames) * int(args.traj_every)
    else:
        steps_to_run = requested_steps
    sim.step(int(steps_to_run))

    state = sim.context.getState(getPositions=True, getVelocities=True, getEnergy=True, enforcePeriodicBox=True)
    pe = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    ke = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
    if not math.isfinite(pe) or not math.isfinite(ke):
        raise FloatingPointError(f"Non-finite energy after biased MD: PE={pe}, KE={ke}")

    with (stage_dir / "final.pdb").open("w", encoding="utf-8") as handle:
        app.PDBFile.writeFile(sim.topology, state.getPositions(), handle)
    sim.saveState(str(stage_dir / "state.xml"))

    expected_frames = int(args.nsteps) // int(args.traj_every)
    n_frames = count_xtc_frames(start_pdb, stage_dir / "traj.xtc", expected_frames)
    if spec is None:
        colvar_path = stage_dir / "COLVAR"
        aligned_path = stage_dir / "COLVAR_aligned.dat"
        rows = [f"{(i + 1) * int(args.traj_every) * dt_ps:.8g} 0" for i in range(n_frames)]
        table = "#! FIELDS time log_weight\n" + "\n".join(rows) + "\n"
        colvar_path.write_text(table, encoding="utf-8")
        aligned_path.write_text(table, encoding="utf-8")
        alignment = {
            "fields": ["time", "log_weight"],
            "n_rows_in": int(n_frames),
            "n_rows_out": int(n_frames),
            "dropped_first_row": False,
            "dropped_initial_rows": 0,
        }
        reweight_kind = "log_weight"
        reweight_column = "log_weight"
        reweight_offset = None
        reweight_energy_unit = "kT"
    else:
        aligned_path = stage_dir / "COLVAR_aligned.dat"
        alignment = align_colvar_to_trajectory(spec.colvar_path, aligned_path, n_frames=n_frames)
        reweight_kind = spec.reweight_kind
        reweight_column = spec.reweight_column
        reweight_offset = spec.reweight_offset_column
        reweight_energy_unit = "kJ/mol"
    weight_diag = log_weight_diagnostics_from_colvar(
        aligned_path,
        column=reweight_column,
        offset_column=reweight_offset,
        kind=reweight_kind,
        temperature_k=temperature_k,
    )
    metadata = {
        "source_endpoint": str(source_endpoint),
        "outdir": str(outdir),
        "method": args.method,
        "temperature_k": temperature_k,
        "dt_ps": dt_ps,
        "nsteps": int(args.nsteps),
        "start_step": int(start_step),
        "resume_existing_frames": int(existing_frames),
        "steps_run": int(steps_to_run),
        "target_total_steps": bool(args.target_total_steps),
        "traj_every": int(args.traj_every),
        "colvar_every": int(args.colvar_every),
        "frames": int(n_frames),
        "cv_mode": cv_mode,
        "cv_spec_path": str(Path(args.cv_spec).expanduser().resolve()) if args.cv_spec else None,
        "cv_spec": cv_spec_data,
        "carbon_atoms_plumed_1based": carbon_atoms,
        "glucose_atom_lookup_plumed_1based": glucose_atom_lookup,
        "biased_cv_note": biased_cv_note,
        "metad_include_torsion_order": bool(args.metad_include_torsion_order),
        "sigma_torsion_order": float(args.sigma_torsion_order),
        "resume": bool(args.resume),
        "append_output": bool(args.append_output),
        "plumed_restart": bool(args.plumed_restart),
        "plumed_dat": str(plumed_path) if spec is not None else None,
        "plumed_action_check": msg,
        "reweighting": {
            "file": str(stage_dir / "COLVAR_aligned.dat"),
            "kind": reweight_kind,
            "column": reweight_column,
            "offset_column": reweight_offset,
            "energy_unit": reweight_energy_unit,
            "statistical_note": "Use these weights for equilibrium estimates; unweighted biased-trajectory estimates are diagnostics only.",
        },
        "alignment": alignment,
        "weight_diagnostics": weight_diag,
    }
    (stage_dir / "bias_metadata.json").write_text(json.dumps(_json_ready(metadata), indent=2), encoding="utf-8")
    (stage_dir / "weight_diagnostics.json").write_text(json.dumps(_json_ready(weight_diag), indent=2), encoding="utf-8")
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source-endpoint-dir", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--method", choices=["opes", "wtmetad", "unbiased"], required=True)
    parser.add_argument("--platform", choices=["CUDA", "OpenCL", "CPU"], default="OpenCL")
    parser.add_argument("--device", default="0")
    parser.add_argument("--precision", choices=["single", "mixed", "double"], default="mixed")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--nsteps", type=int, default=1_250_000)
    parser.add_argument("--traj-every", type=int, default=500)
    parser.add_argument("--colvar-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=5000)
    parser.add_argument("--ckpt-every", type=int, default=62500)
    parser.add_argument("--temperature-k", type=float, default=None)
    parser.add_argument("--dt-ps", type=float, default=None)
    parser.add_argument("--friction-ps", type=float, default=None)
    parser.add_argument("--opes-barrier-kj-mol", type=float, default=20.0)
    parser.add_argument("--metad-height-kj-mol", type=float, default=0.35)
    parser.add_argument("--metad-biasfactor", type=float, default=10.0)
    parser.add_argument("--sigma-distance-nm", type=float, default=0.05)
    parser.add_argument("--sigma-rg-nm", type=float, default=0.03)
    parser.add_argument("--metad-include-torsion-order", action="store_true")
    parser.add_argument("--sigma-torsion-order", type=float, default=0.6)
    parser.add_argument("--cv-mode", choices=["decane", "glucose-pyranose"], default="decane")
    parser.add_argument("--cv-spec", default=None, help="Versioned YAML CV specification; overrides --cv-mode")
    parser.add_argument("--sigma-glucose-ring-rg-nm", type=float, default=0.025)
    parser.add_argument("--sigma-glucose-exocyclic-radians", type=float, default=0.35)
    parser.add_argument("--residue-name", default="UNL")
    parser.add_argument("--plumed-binary", default="plumed")
    parser.add_argument("--skip-plumed-action-check", action="store_true")
    parser.add_argument("--copy-inputs", action="store_true", help="Copy system/start files instead of symlinking them")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--target-total-steps", action="store_true", help="Treat --nsteps as the desired final step when resuming")
    parser.add_argument("--append-output", action="store_true", help="Append reporters/COLVAR to existing output files for continuation runs")
    parser.add_argument("--plumed-restart", action="store_true", help="Enable PLUMED restart/append semantics in generated plumed.dat")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_biased_md(args)
    print(json.dumps(_json_ready(result), indent=2))


if __name__ == "__main__":
    main()
