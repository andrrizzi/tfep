#!/usr/bin/env python3
"""Template entrypoint for external solvated bidirectional TFEP wrappers.

This script is intentionally lightweight and library-oriented: it demonstrates
how an external project can assemble shell conditioning and shell-equivariant
specs using reusable tfep interfaces.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from tfep.solvation import (
    build_shell_equivariant_index_spec,
    compute_shell_slot_conditioning_indices,
)


@dataclass
class TemplateConfig:
    topology_pdb: str
    state0_traj: str
    state1_traj: str
    mapped_atoms: str = "AUTO"
    conditioning_mode: str = "solute_only"  # solute_only | shell_slots
    shell_water_selection: str = "water"
    shell_oxygen_names: List[str] = field(default_factory=lambda: ["O", "OW"])
    shell_k1: int = 0
    shell_k2: int = 0
    shell_condition_on: str = "molecule"
    shell_equivariant_enabled: bool = False
    shell_equiv_solute_selection: str = "AUTO"
    shell_equiv_water_resnames: str = "HOH,SOL,WAT"
    shell_equiv_water_oxygen_names: str = "O,OW"
    shell_equiv_ion_resnames: str = "NA,CL"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Template for external wrappers: resolves shell conditioning and "
            "shell-equivariant mappings without enforcing any project-specific runner."
        )
    )
    parser.add_argument("--config-json", type=Path, required=True, help="Path to JSON config")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve/print all derived fields without launching training.",
    )
    return parser


def load_config(path: Path) -> TemplateConfig:
    payload = json.loads(path.read_text())
    return TemplateConfig(**payload)


def derive_runtime_plan(config: TemplateConfig) -> Dict[str, object]:
    topology = Path(config.topology_pdb).expanduser().resolve()
    if not topology.exists():
        raise FileNotFoundError(f"Topology file not found: {topology}")

    runtime: Dict[str, object] = {
        "topology_pdb": str(topology),
        "state0_traj": str(Path(config.state0_traj).expanduser().resolve()),
        "state1_traj": str(Path(config.state1_traj).expanduser().resolve()),
        "mapped_atoms": config.mapped_atoms,
        "conditioning_mode": config.conditioning_mode,
        "shell_k1": int(config.shell_k1),
        "shell_k2": int(config.shell_k2),
    }

    k_total = int(config.shell_k1) + int(config.shell_k2)
    if config.conditioning_mode == "shell_slots" and k_total > 0:
        runtime["conditioning_atom_indices"] = compute_shell_slot_conditioning_indices(
            topology,
            water_selection=config.shell_water_selection,
            oxygen_names=config.shell_oxygen_names,
            k_total=k_total,
            condition_on=config.shell_condition_on,
        )
    else:
        runtime["conditioning_atom_indices"] = []

    if bool(config.shell_equivariant_enabled):
        runtime["shell_equivariant_spec"] = build_shell_equivariant_index_spec(
            topology,
            solute_selection=config.shell_equiv_solute_selection,
            water_resnames_text=config.shell_equiv_water_resnames,
            water_oxygen_names_text=config.shell_equiv_water_oxygen_names,
            ion_resnames_text=config.shell_equiv_ion_resnames,
        )
    else:
        runtime["shell_equivariant_spec"] = None

    return runtime


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config_json)
    runtime = derive_runtime_plan(config)

    summary = {
        "config": asdict(config),
        "runtime": runtime,
        "notes": [
            "This template only resolves a training plan and derived indices/specs.",
            "External projects should call their own map/trainer constructors.",
            "Use --dry-run for CI/smoke tests of wrapper wiring.",
        ],
    }

    print(json.dumps(summary, indent=2, default=str))

    if not args.dry_run:
        raise RuntimeError(
            "Template mode does not launch training. Keep wrapper execution in your external project."
        )


if __name__ == "__main__":
    main()
