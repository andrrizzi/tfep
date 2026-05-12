#!/usr/bin/env python

import numpy as np
import pytest

from tfep.solvation import (
    build_shell_equivariant_index_spec,
    compute_shell_slot_conditioning_indices,
    merge_shell_rankings,
    rank_shell_waters_by_occupancy,
    resolve_solute_atom_indices,
)
from tfep.io.dataset.shell import SolvationShellPermutingTrajectoryDataset


def _build_simple_solvated_universe(n_waters: int = 3):
    import MDAnalysis

    atom_resindex = np.concatenate([[0], np.repeat(np.arange(1, n_waters + 1), 3)])
    universe = MDAnalysis.Universe.empty(
        n_atoms=n_waters * 3 + 1,
        n_residues=n_waters + 1,
        atom_resindex=atom_resindex,
        residue_segindex=[0] * (n_waters + 1),
        trajectory=True,
    )

    atom_names = ["C"] + ["O", "H1", "H2"] * n_waters
    elements = ["C"] + ["O", "H", "H"] * n_waters
    resnames = ["LIG"] + ["WAT"] * n_waters
    resids = [1] + list(range(2, n_waters + 2))

    universe.add_TopologyAttr("name", atom_names)
    universe.add_TopologyAttr("type", elements)
    universe.add_TopologyAttr("element", elements)
    universe.add_TopologyAttr("resname", resnames)
    universe.add_TopologyAttr("resid", resids)
    universe.add_TopologyAttr("segid", ["SYS"])

    template = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.95, 0.0, 0.0],
            [-0.24, 0.93, 0.0],
        ],
        dtype=float,
    )

    positions = np.zeros((n_waters * 3 + 1, 3), dtype=float)
    positions[0] = [0.0, 0.0, 0.0]
    centers = [[10.0, 0.0, 0.0], [1.0, 0.0, 0.0], [5.0, 0.0, 0.0]]
    for i, center in enumerate(centers[:n_waters]):
        start = 1 + i * 3
        positions[start : start + 3] = template + np.asarray(center)

    universe.atoms.positions = positions
    return universe


def test_resolve_solute_indices_auto_and_explicit():
    universe = _build_simple_solvated_universe(n_waters=2)

    auto_idx = resolve_solute_atom_indices(
        universe,
        solute_selection="AUTO",
        water_resnames=("WAT",),
        ion_resnames=("NA", "CL"),
    )
    assert auto_idx == [0]

    explicit_idx = resolve_solute_atom_indices(
        universe,
        solute_selection="resname LIG",
        water_resnames=("WAT",),
        ion_resnames=(),
    )
    assert explicit_idx == [0]


def test_shell_slot_conditioning_indices(tmp_path):
    universe = _build_simple_solvated_universe(n_waters=3)
    top_path = tmp_path / "sys.pdb"
    universe.atoms.write(str(top_path))

    molecule_idx = compute_shell_slot_conditioning_indices(
        top_path,
        water_selection="water",
        oxygen_names=["O"],
        k_total=2,
        condition_on="molecule",
    )
    oxygen_idx = compute_shell_slot_conditioning_indices(
        top_path,
        water_selection="water",
        oxygen_names=["O"],
        k_total=2,
        condition_on="oxygen",
    )

    assert len(molecule_idx) == 6
    assert oxygen_idx == [1, 4]


def test_rank_and_merge_shell_candidates():
    universe = _build_simple_solvated_universe(n_waters=3)
    records = rank_shell_waters_by_occupancy(
        universe,
        solute_atom_indices=[0],
        water_selection="water",
        oxygen_names=["O"],
        cutoff_angstrom=2.0,
        min_occupancy=0.0,
    )
    assert len(records) == 3
    assert records[0]["resid"] == 3

    merged = merge_shell_rankings(records[:2], records[1:], max_waters=2)
    assert len(merged) == 2
    assert merged[0]["occupancy_sum"] >= merged[1]["occupancy_sum"]


def test_shell_permuting_dataset_moves_nearest_water_to_first_slot():
    universe = _build_simple_solvated_universe(n_waters=3)

    dataset = SolvationShellPermutingTrajectoryDataset(
        universe,
        center_atoms=[0],
        water_selection="water",
        oxygen_names=["O"],
        k1=1,
        k2=0,
    )

    sample = dataset[0]
    xyz = sample["positions"].numpy().reshape(-1, 3)

    # First water slot oxygen is atom index 1. It should be moved near the solute center (x~1).
    assert pytest.approx(xyz[1, 0], abs=1e-3) == 1.0


def test_build_shell_equivariant_index_spec(tmp_path):
    universe = _build_simple_solvated_universe(n_waters=2)
    top_path = tmp_path / "sys.pdb"
    universe.atoms.write(str(top_path))

    spec = build_shell_equivariant_index_spec(
        top_path,
        solute_selection="AUTO",
        water_resnames_text="WAT",
        water_oxygen_names_text="O",
        ion_resnames_text="NA,CL",
    )

    assert spec["n_solute_atoms"] == 1
    assert spec["n_waters"] == 2
    assert spec["n_mapped_atoms"] == 7
    assert spec["mapped_global_indices"].dtype == np.int64
