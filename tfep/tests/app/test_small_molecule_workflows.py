#!/usr/bin/env python

import subprocess
import sys
import tomllib
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "module_name",
    [
        "tfep.app.small_molecule_tmbar",
        "tfep.analysis.tbar_cv_bootstrap",
        "tfep.analysis.tbar_holdout",
        "tfep.analysis.tbar_plots",
    ],
)
def test_packaged_workflow_help_is_importable_without_openmm(module_name):
    result = subprocess.run(
        [sys.executable, "-m", module_name, "--help"],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_packaged_workflow_entrypoints_are_declared():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["scripts"] == {
        "tfep-small-molecule-tmbar": "tfep.app.small_molecule_tmbar:main",
        "tfep-tbar-cv-bootstrap": "tfep.analysis.tbar_cv_bootstrap:main",
        "tfep-tbar-holdout": "tfep.analysis.tbar_holdout:main",
        "tfep-tbar-plots": "tfep.analysis.tbar_plots:main",
    }


def test_openmm_runtime_guard_has_actionable_message_when_openmm_is_missing(tmp_path):
    from tfep.app import small_molecule_tmbar

    if small_molecule_tmbar._OPENMM_IMPORT_ERROR is None:
        pytest.skip("OpenMM is installed in this environment")

    with pytest.raises(RuntimeError, match="requires OpenMM"):
        small_molecule_tmbar._load_openmm_system(tmp_path / "missing.xml")
