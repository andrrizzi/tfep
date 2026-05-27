"""Serialization helpers for stochastic path-weighted TFEP outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch

from .path import StochasticPathBatch
from .work import effective_sample_size_from_log_weights


README_TEXT = """# Experimental stochastic path-weighted TFEP output

This directory was produced by the experimental path-weighted stochastic TFEP
estimator. Work values include deterministic log-Jacobian terms and stochastic
forward/reverse transition probability terms.

Do not mix these path works with deterministic TFEP/BAR arrays unless the
stochastic estimator was explicitly selected and the sign convention has been
validated for the run.
"""


def _cpu_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def path_rows(path: StochasticPathBatch) -> list[dict[str, Any]]:
    """Convert a path batch to CSV-friendly rows."""
    path = path.detach()
    rows = []
    for i in range(path.batch_size):
        rows.append(
            {
                "path_id": i,
                "direction": path.direction,
                "u_source_x0": float(path.u_source_x0[i].cpu()),
                "u_target_xK": float(path.u_target_xK[i].cpu()),
                "sum_logJ": float(path.sum_logJ[i].cpu()),
                "sum_logq_forward": float(path.sum_logq_forward[i].cpu()),
                "sum_logq_reverse": float(path.sum_logq_reverse[i].cpu()),
                "path_work": float(path.path_work[i].cpu()),
                "log_weight": float(path.log_weight[i].cpu()),
                "success": bool(path.success[i].cpu()) if path.success is not None else True,
                "exception_message": path.exception_messages[i] if path.exception_messages else "",
            }
        )
    return rows


def write_path_outputs(
    output_dir: Union[str, Path],
    *,
    forward: Optional[StochasticPathBatch] = None,
    reverse: Optional[StochasticPathBatch] = None,
    config: Optional[dict[str, Any]] = None,
    extra_results: Optional[dict[str, Any]] = None,
) -> None:
    """Write path arrays, CSV summaries, and metadata for auditability."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "README.md").write_text(README_TEXT)
    (output / "config.json").write_text(json.dumps(config or {}, indent=2, sort_keys=True))

    diagnostics: dict[str, Any] = {"experimental": True}
    if extra_results:
        diagnostics.update(extra_results)

    for label, path in (("forward", forward), ("reverse", reverse)):
        if path is None:
            continue
        detached = path.detach()
        np.save(output / f"work_{label}.npy", _cpu_numpy(detached.path_work))
        np.savez(
            output / f"log_terms_{label}.npz",
            sum_logJ=_cpu_numpy(detached.sum_logJ),
            sum_logq_forward=_cpu_numpy(detached.sum_logq_forward),
            sum_logq_reverse=_cpu_numpy(detached.sum_logq_reverse),
            log_weight=_cpu_numpy(detached.log_weight),
        )
        rows = path_rows(detached)
        with (output / f"paths_{label}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
            writer.writeheader()
            writer.writerows(rows)
        diagnostics[f"{label}_n_paths"] = detached.batch_size
        diagnostics[f"{label}_ess"] = float(effective_sample_size_from_log_weights(detached.log_weight).cpu())
        diagnostics[f"{label}_metadata"] = dict(detached.metadata)

    (output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2, sort_keys=True))
