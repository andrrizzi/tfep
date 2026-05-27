"""Tests for diagnostic-only short relaxation utilities."""
from __future__ import annotations

import json

import numpy as np
import pytest

from tfep.analysis.short_relaxation import (
    RelaxationDiagnosticConfig,
    run_short_relaxation_diagnostic,
    write_relaxation_outputs,
)


def _write_two_atom_openmm_endpoint(tmp_path):
    openmm = pytest.importorskip("openmm")
    openmm_app = pytest.importorskip("openmm.app")
    unit = openmm.unit

    system = openmm.System()
    system.addParticle(12.0 * unit.amu)
    system.addParticle(12.0 * unit.amu)
    force = openmm.HarmonicBondForce()
    force.addBond(
        0,
        1,
        0.15 * unit.nanometer,
        1000.0 * unit.kilojoule_per_mole / unit.nanometer**2,
    )
    system.addForce(force)

    topology = openmm_app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("UNL", chain)
    element = openmm_app.Element.getByAtomicNumber(6)
    atom0 = topology.addAtom("C1", element, residue)
    atom1 = topology.addAtom("C2", element, residue)
    topology.addBond(atom0, atom1)

    positions = [
        openmm.Vec3(0.0, 0.0, 0.0),
        openmm.Vec3(0.30, 0.0, 0.0),
    ] * unit.nanometer

    system_xml = tmp_path / "system.xml"
    topology_pdb = tmp_path / "start.pdb"
    system_xml.write_text(openmm.XmlSerializer.serialize(system), encoding="utf-8")
    with topology_pdb.open("w", encoding="utf-8") as handle:
        openmm_app.PDBFile.writeFile(topology, positions, handle)
    return system_xml, topology_pdb


def _one_frame(positions_angstrom):
    return {
        "state0_to_state1": [
            {
                "frame_index": 0,
                "source_state": 0,
                "target_state": 1,
                "dataset_sample_index": 7,
                "trajectory_sample_index": 11,
                "positions_angstrom": np.asarray(positions_angstrom, dtype=float),
                "deterministic_tfep_work": 3.14,
                "log_det_J": 0.25,
            }
        ]
    }


def test_noop_relaxation_keeps_coordinates_and_work_metadata(tmp_path):
    system_xml, topology_pdb = _write_two_atom_openmm_endpoint(tmp_path)
    positions = np.asarray([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=float)
    config = RelaxationDiagnosticConfig(mode="noop", output_dir=str(tmp_path / "diag"))

    result = run_short_relaxation_diagnostic(
        mapped_frames=_one_frame(positions),
        target_system_xml_paths={"state0_to_state1": system_xml},
        target_topology_pdb_paths={"state0_to_state1": topology_pdb},
        config=config,
        mapped_atom_indices=[0, 1],
        solute_selection="mapped",
    )

    record = result["records"][0]
    assert record.success
    assert record.delta_u_relax == pytest.approx(0.0, abs=1e-10)
    assert record.deterministic_tfep_work == pytest.approx(3.14)
    np.testing.assert_allclose(result["mapped_coordinates"]["state0_to_state1"], result["relaxed_coordinates"]["state0_to_state1"])


def test_minimization_reduces_simple_bond_energy(tmp_path):
    system_xml, topology_pdb = _write_two_atom_openmm_endpoint(tmp_path)
    positions = np.asarray([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=float)
    config = RelaxationDiagnosticConfig(mode="minimize", steps=100, output_dir=str(tmp_path / "diag"))

    result = run_short_relaxation_diagnostic(
        mapped_frames=_one_frame(positions),
        target_system_xml_paths={"state0_to_state1": system_xml},
        target_topology_pdb_paths={"state0_to_state1": topology_pdb},
        config=config,
        mapped_atom_indices=[0, 1],
        solute_selection="mapped",
    )

    record = result["records"][0]
    assert record.success
    assert record.target_energy_after_kj_mol <= record.target_energy_before_kj_mol + 1e-8
    assert record.delta_u_relax <= 1e-8


def test_failed_frame_is_recorded_when_not_strict(tmp_path):
    system_xml, topology_pdb = _write_two_atom_openmm_endpoint(tmp_path)
    bad_positions = np.asarray([[0.0, 0.0, 0.0]], dtype=float)
    config = RelaxationDiagnosticConfig(mode="minimize", strict=False, output_dir=str(tmp_path / "diag"))

    result = run_short_relaxation_diagnostic(
        mapped_frames=_one_frame(bad_positions),
        target_system_xml_paths={"state0_to_state1": system_xml},
        target_topology_pdb_paths={"state0_to_state1": topology_pdb},
        config=config,
        mapped_atom_indices=[0, 1],
        solute_selection="mapped",
    )

    record = result["records"][0]
    assert not record.success
    assert "Exception" in record.exception or "OpenMM" in record.exception


def test_outputs_include_readme_and_metadata_warning(tmp_path):
    system_xml, topology_pdb = _write_two_atom_openmm_endpoint(tmp_path)
    positions = np.asarray([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=float)
    outdir = tmp_path / "diag"
    config = RelaxationDiagnosticConfig(mode="noop", output_dir=str(outdir))

    result = run_short_relaxation_diagnostic(
        mapped_frames=_one_frame(positions),
        target_system_xml_paths={"state0_to_state1": system_xml},
        target_topology_pdb_paths={"state0_to_state1": topology_pdb},
        config=config,
        mapped_atom_indices=[0, 1],
        solute_selection="mapped",
    )
    write_relaxation_outputs(result, output_dir=outdir, config=config)

    assert (outdir / "relaxation_summary.csv").is_file()
    assert (outdir / "energy_before_after.npy").is_file()
    readme = (outdir / "README.md").read_text(encoding="utf-8")
    assert "diagnostic post-processing outputs only" in readme
    metadata = json.loads((outdir / "metadata.json").read_text(encoding="utf-8"))
    assert "not used in the deterministic TFEP/BAR/TMBAR estimator" in metadata["statistical_meaning"]
