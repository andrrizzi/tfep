#!/usr/bin/env python

import warnings
import json
import os
import subprocess
import sys
from pathlib import Path

from tfep.tests import DATA_DIR_PATH


def test_contrib_losses_shim_imports():
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        from tfep.contrib.losses import BARLikeRegularizer, fep_forward_df, fep_reverse_df

    assert BARLikeRegularizer is not None
    assert fep_forward_df is not None
    assert fep_reverse_df is not None


def test_contrib_triatomic_shim_imports():
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        from tfep.contrib.flows.triatomic_zmat import TriatomicZMatrixFlow, VectorCouplingFlow

    assert TriatomicZMatrixFlow is not None
    assert VectorCouplingFlow is not None


def test_ozone_tmbar_compat_class_available():
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        from tfep.contrib.ozone.tmbar_map import OzoneTMBARMap

    assert OzoneTMBARMap is not None


def test_solvated_template_dry_run_smoke(tmp_path):
    script_path = Path(__file__).resolve().parents[2] / "contrib" / "examples" / "solvated_bidirectional_template.py"
    cfg_path = tmp_path / "cfg.json"
    topology = Path(DATA_DIR_PATH) / "chloro-fluoromethane.pdb"

    payload = {
        "topology_pdb": str(topology),
        "state0_traj": str(topology),
        "state1_traj": str(topology),
        "conditioning_mode": "solute_only",
        "shell_k1": 0,
        "shell_k2": 0,
        "shell_equivariant_enabled": False,
    }
    cfg_path.write_text(json.dumps(payload))

    completed = subprocess.run(
        [sys.executable, str(script_path), "--config-json", str(cfg_path), "--dry-run"],
        capture_output=True,
        text=True,
        env={
            **dict(os.environ),
            "PYTHONPATH": str(Path(__file__).resolve().parents[3]),
        },
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    data = json.loads(completed.stdout)
    assert data["runtime"]["conditioning_atom_indices"] == []
    assert data["runtime"]["shell_equivariant_spec"] is None
