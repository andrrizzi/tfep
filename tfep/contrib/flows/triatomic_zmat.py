"""Deprecated compatibility shim for contrib triatomic flow import path."""

from __future__ import annotations

import warnings

from tfep.nn.flows.triatomic_zmatrix import TriatomicZMatrixFlow, VectorCouplingFlow

warnings.warn(
    "tfep.contrib.flows.triatomic_zmat is deprecated; import from "
    "tfep.nn.flows.triatomic_zmatrix instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["TriatomicZMatrixFlow", "VectorCouplingFlow"]
