#!/usr/bin/env python

import numpy as np
import pytest

from tfep.utils.plumed.reweighting import kt_in_unit, read_plumed_log_weights


def _write_colvar(path):
    path.write_text(
        "\n".join([
            "#! FIELDS time bias rbias opes.bias opes.rct logw",
            "0.0 1.0 0.5 2.0 0.2 -3.0",
            "1.0 2.0 1.5 3.0 0.4 -2.0",
            "2.0 3.0 2.5 4.0 0.6 -1.0",
        ]) + "\n",
        encoding="utf-8",
    )


def test_plumed_log_weight_kind_uses_column_directly(tmp_path):
    colvar = tmp_path / "COLVAR"
    _write_colvar(colvar)

    result = read_plumed_log_weights(
        colvar,
        kind="log_weight",
        column="logw",
        temperature_k=300.0,
        energy_unit="kJ/mol",
    )

    assert np.allclose(result.log_weights, [-3.0, -2.0, -1.0])
    assert result.metadata["kind"] == "log_weight"
    assert result.metadata["weighted"] is True


@pytest.mark.parametrize("kind,column", [("bias", "bias"), ("rbias", "rbias")])
def test_plumed_bias_like_columns_are_divided_by_kT(tmp_path, kind, column):
    colvar = tmp_path / "COLVAR"
    _write_colvar(colvar)
    kt = kt_in_unit(300.0, "kJ/mol")

    result = read_plumed_log_weights(
        colvar,
        kind=kind,
        column=column,
        temperature_k=300.0,
        energy_unit="kJ/mol",
    )

    expected = np.array([0.5, 1.5, 2.5]) / kt if column == "rbias" else np.array([1.0, 2.0, 3.0]) / kt
    assert np.allclose(result.log_weights, expected)


def test_plumed_bias_minus_offset_uses_difference_over_kT(tmp_path):
    colvar = tmp_path / "COLVAR"
    _write_colvar(colvar)
    kt = kt_in_unit(300.0, "kJ/mol")

    result = read_plumed_log_weights(
        colvar,
        kind="bias_minus_offset",
        column="opes.bias",
        offset_column="opes.rct",
        temperature_k=300.0,
        energy_unit="kJ/mol",
    )

    assert np.allclose(result.log_weights, (np.array([2.0, 3.0, 4.0]) - np.array([0.2, 0.4, 0.6])) / kt)


def test_plumed_bias_minus_offset_requires_offset_column(tmp_path):
    colvar = tmp_path / "COLVAR"
    _write_colvar(colvar)

    with pytest.raises(ValueError, match="offset_column"):
        read_plumed_log_weights(
            colvar,
            kind="bias_minus_offset",
            column="opes.bias",
            temperature_k=300.0,
        )
