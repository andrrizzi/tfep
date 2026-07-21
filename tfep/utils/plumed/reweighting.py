"""Utilities to convert PLUMED/OPES/metadynamics bias columns to log weights."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from tfep.analysis.reweighting import log_weight_diagnostics
from tfep.utils.plumed.io import read_table


KJ_PER_MOL_PER_K = 0.00831446261815324
KCAL_PER_MOL_PER_K = 0.00198720425864083


@dataclass(frozen=True)
class PlumedLogWeights:
    """Dimensionless log weights loaded from a PLUMED-style table."""

    log_weights: np.ndarray
    metadata: dict[str, Any]


def kt_in_unit(temperature_k: float, energy_unit: str) -> float:
    """Return ``kT`` in the requested energy unit."""
    unit = str(energy_unit).lower().replace("_", "/")
    if unit == "kt":
        return 1.0
    if unit in {"kj/mol", "kjmol", "kilojoule/mol", "kilojoule_per_mole"}:
        return KJ_PER_MOL_PER_K * float(temperature_k)
    if unit in {"kcal/mol", "kcalmol", "kilocalorie/mol", "kilocalorie_per_mole"}:
        return KCAL_PER_MOL_PER_K * float(temperature_k)
    raise ValueError(f"Unsupported reweight energy unit: {energy_unit!r}")


def read_plumed_log_weights(
    file_path: str | Path,
    *,
    kind: str,
    column: str,
    temperature_k: float,
    energy_unit: str = "kJ/mol",
    offset_column: Optional[str] = None,
    remove_duplicates: bool = True,
) -> PlumedLogWeights:
    """Read a PLUMED table and return dimensionless log weights.

    Parameters
    ----------
    kind
        One of ``log_weight``, ``bias``, ``rbias``, or ``bias_minus_offset``.
        ``bias`` and ``rbias`` are divided by ``kT``. ``log_weight`` is used as
        already dimensionless.
    column
        PLUMED column used as the weight source, e.g. ``metad.rbias`` or
        ``opes.bias``.
    offset_column
        Required only for ``bias_minus_offset``; the returned weight is
        ``(column - offset_column) / kT``.
    """
    path = Path(file_path).expanduser().resolve()
    mode = str(kind).strip().lower().replace("-", "_")
    if mode not in {"log_weight", "bias", "rbias", "bias_minus_offset"}:
        raise ValueError("kind must be one of: log_weight, bias, rbias, bias_minus_offset")

    cols = [str(column)]
    if mode == "bias_minus_offset":
        if offset_column is None:
            raise ValueError("offset_column is required for kind='bias_minus_offset'")
        cols.append(str(offset_column))

    data = read_table(path, col_names=cols, remove_duplicates=remove_duplicates)
    values = np.asarray(data[str(column)], dtype=np.float64).reshape(-1)
    kt = kt_in_unit(float(temperature_k), energy_unit)

    if mode == "log_weight":
        logw = values
    elif mode in {"bias", "rbias"}:
        logw = values / kt
    else:
        offset = np.asarray(data[str(offset_column)], dtype=np.float64).reshape(-1)
        logw = (values - offset) / kt

    metadata: dict[str, Any] = {
        "source_file": str(path),
        "kind": mode,
        "column": str(column),
        "offset_column": str(offset_column) if offset_column is not None else None,
        "temperature_k": float(temperature_k),
        "energy_unit": str(energy_unit),
        "kT_in_energy_unit": float(kt),
        "n_rows": int(logw.size),
    }
    metadata.update(log_weight_diagnostics(logw))
    return PlumedLogWeights(log_weights=np.asarray(logw, dtype=np.float64), metadata=metadata)
