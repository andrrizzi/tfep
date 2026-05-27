"""Molecular evaluation helpers for experimental stochastic path TFEP.

These helpers bridge existing TFEP/TMBAR map objects to the standalone
path-weighted stochastic estimator. The code is evaluation-only: it computes
stochastic path works beside the deterministic work arrays and never feeds
stochastic final coordinates back into deterministic BAR/TMBAR logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import json
from pathlib import Path

import numpy as np
import torch

try:
    import MDAnalysis as mda
except Exception:  # pragma: no cover - MDAnalysis is a required tfep dependency in practice.
    mda = None

from .kernels import GaussianRandomWalkKernel, KernelContext, UnadjustedLangevinKernel
from .work import compute_path_work


BatchMover = Callable[[Any, torch.device], Any]
ArrayConverter = Callable[[Any], np.ndarray]
PotentialEvaluator = Callable[[int, torch.Tensor, Optional[torch.Tensor]], torch.Tensor]


@dataclass(frozen=True)
class MolecularStochasticConfig:
    """Configuration for molecular stochastic path evaluation."""

    estimator: str = "deterministic-tfep"
    kernel: str = "gaussian-rw"
    noise_sigma: Optional[float] = None
    step_size: Optional[float] = None
    diffusion: float = 1.0
    num_blocks: int = 1
    steps_per_block: int = 1
    gradient_policy: str = "stop-gradient"
    seed: int = 123
    max_frames: Optional[int] = None
    every_n_frames: int = 1
    selected_atoms: Optional[str] = None
    apply_to: str = "selected"
    output_dir: str = "stochastic_tfep_outputs"
    strict: bool = False
    allow_molecular_ula: bool = False
    allow_constrained_cartesian_ula: bool = False

    @property
    def enabled(self) -> bool:
        return str(self.estimator) == "stochastic-path-tfep"

    @property
    def n_stochastic_steps(self) -> int:
        return max(1, int(self.num_blocks)) * max(1, int(self.steps_per_block))


def atom_indices_to_flat_indices(atom_indices: Sequence[int]) -> np.ndarray:
    atom_indices = np.asarray(atom_indices, dtype=int).reshape(-1)
    flat = np.empty(atom_indices.size * 3, dtype=int)
    flat[0::3] = 3 * atom_indices
    flat[1::3] = 3 * atom_indices + 1
    flat[2::3] = 3 * atom_indices + 2
    return flat


def _read_indices_file(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path).astype(int).reshape(-1)
    values = []
    for token in path.read_text(encoding="utf-8").replace(",", " ").split():
        values.append(int(token))
    return np.asarray(values, dtype=int)


def _selection_to_atom_indices(topology_path: Union[str, Path], selection: str) -> np.ndarray:
    if mda is None:
        raise RuntimeError("MDAnalysis is required to resolve atom selections for stochastic TFEP")
    universe = mda.Universe(str(topology_path))
    atoms = universe.select_atoms(str(selection))
    if len(atoms) == 0:
        raise ValueError(f"SNF atom selection returned zero atoms: {selection!r}")
    return np.asarray(atoms.indices, dtype=int)


def _water_shell_atom_indices(path: Union[str, Path]) -> np.ndarray:
    data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    atoms: list[int] = []
    for record in data.get("selected_water_records", []):
        atoms.extend(int(i) for i in record.get("atom_indices", []))
    if not atoms:
        raise ValueError(f"No selected water atom indices found in shell mapping: {path}")
    return np.asarray(sorted(set(atoms)), dtype=int)


def resolve_snf_flat_indices(
    *,
    model: Any,
    topology_path: Union[str, Path],
    config: MolecularStochasticConfig,
    mapped_atom_indices: Optional[Sequence[int]] = None,
    water_shell_mapping_path: Optional[Union[str, Path]] = None,
) -> Optional[np.ndarray]:
    """Resolve stochastic-kernel flattened coordinate indices.

    ``None`` means all flattened coordinates. Returned indices are flattened
    coordinate indices, not atom indices.
    """
    apply_to = str(config.apply_to).lower()
    if apply_to == "all":
        return None

    atom_indices: Optional[np.ndarray] = None
    selected = config.selected_atoms
    if selected:
        selected_path = Path(str(selected)).expanduser()
        if selected_path.is_file():
            atom_indices = _read_indices_file(selected_path)
        else:
            atom_indices = _selection_to_atom_indices(topology_path, str(selected))
    elif apply_to == "shell":
        if water_shell_mapping_path is None:
            raise ValueError("--snf-apply-to shell requires a water_shell_mapping.json path")
        atom_indices = _water_shell_atom_indices(water_shell_mapping_path)
    elif mapped_atom_indices is not None:
        atom_indices = np.asarray(mapped_atom_indices, dtype=int).reshape(-1)
    else:
        try:
            mapped = model.get_mapped_indices(idx_type="atom", remove_fixed=False)
            atom_indices = np.asarray(mapped.detach().cpu().numpy() if torch.is_tensor(mapped) else mapped, dtype=int).reshape(-1)
        except Exception as exc:
            raise ValueError(
                "Could not infer SNF selected atoms. Pass --snf-selected-atoms or use --snf-apply-to all."
            ) from exc

    if atom_indices is None or atom_indices.size == 0:
        raise ValueError("SNF selected atom set is empty")
    return atom_indices_to_flat_indices(atom_indices)


def _make_generator(seed: int, device: torch.device) -> torch.Generator:
    generator_device = "cpu" if device.type == "cpu" else device.type
    try:
        gen = torch.Generator(device=generator_device)
    except TypeError:
        gen = torch.Generator()
    gen.manual_seed(int(seed))
    return gen


def _kernel_from_config(config: MolecularStochasticConfig, selected_flat_indices: Optional[Sequence[int]]):
    kernel_name = str(config.kernel).lower()
    if kernel_name == "gaussian-rw":
        sigma = 0.01 if config.noise_sigma is None else float(config.noise_sigma)
        if sigma <= 0.0:
            raise ValueError("SNF gaussian-rw requires --snf-noise-sigma > 0")
        return GaussianRandomWalkKernel(sigma=sigma, selected_indices=selected_flat_indices)
    if kernel_name == "ula":
        if not (bool(config.allow_molecular_ula) or bool(config.allow_constrained_cartesian_ula)):
            raise RuntimeError(
                "Molecular ULA is gated because constraints/PBC can invalidate the Cartesian transition density. "
                "Pass --snf-allow-molecular-ula for validated unconstrained-coordinate smoke tests, or "
                "--snf-allow-constrained-cartesian-ula for explicitly experimental constrained endpoint runs."
            )
        if config.step_size is None or float(config.step_size) <= 0.0:
            raise ValueError("SNF ULA requires --snf-step-size > 0")
        return UnadjustedLangevinKernel(
            step_size=float(config.step_size),
            diffusion=float(config.diffusion),
            selected_indices=selected_flat_indices,
            gradient_policy=str(config.gradient_policy),
        )
    raise ValueError(f"Unsupported molecular stochastic kernel: {config.kernel}")


def _finite_concat(chunks: Iterable[np.ndarray]) -> np.ndarray:
    chunks = [np.asarray(chunk) for chunk in chunks]
    if not chunks:
        return np.empty(0, dtype=float)
    return np.concatenate(chunks)


def _displacement_stats(x_final: torch.Tensor, x_mapped: torch.Tensor, selected_flat_indices: Optional[Sequence[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    if selected_flat_indices is not None:
        idx = torch.as_tensor(selected_flat_indices, dtype=torch.long, device=x_final.device)
        x_final = x_final.index_select(1, idx)
        x_mapped = x_mapped.index_select(1, idx)
    disp = (x_final - x_mapped).reshape(x_final.shape[0], -1, 3)
    per_atom = torch.linalg.norm(disp, dim=-1)
    rmsd = torch.sqrt(torch.mean(per_atom.pow(2), dim=1))
    max_disp = torch.amax(per_atom, dim=1)
    return rmsd, max_disp


def evaluate_stochastic_direction(
    *,
    model: Any,
    loader: torch.utils.data.DataLoader,
    state_from: int,
    state_to: int,
    direction: str,
    config: MolecularStochasticConfig,
    selected_flat_indices: Optional[Sequence[int]],
    move_batch_to_device: BatchMover,
    to_numpy: ArrayConverter,
    seed_offset: int = 0,
) -> Dict[str, np.ndarray]:
    """Evaluate stochastic path works for one molecular direction."""
    device = model.device
    rng = _make_generator(int(config.seed) + int(seed_offset), device)
    kernel = _kernel_from_config(config, selected_flat_indices)

    records: Dict[str, list[np.ndarray]] = {
        "dataset_sample_index": [],
        "trajectory_sample_index": [],
        "snf_work": [],
        "u_from": [],
        "u_to_snf": [],
        "u_to_mapped": [],
        "log_det_J": [],
        "sum_logq_forward": [],
        "sum_logq_reverse": [],
        "path_log_weight": [],
        "mapped_to_snf_rmsd_angstrom": [],
        "mapped_to_snf_max_disp_angstrom": [],
    }

    n_seen = 0
    n_kept = 0
    every = max(1, int(config.every_n_frames))
    max_frames = None if config.max_frames is None else int(config.max_frames)

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        batch_size = int(batch["positions"].shape[0])
        keep_rows = []
        for row in range(batch_size):
            keep = (n_seen % every) == 0 and (max_frames is None or n_kept < max_frames)
            if keep:
                keep_rows.append(row)
                n_kept += 1
            n_seen += 1
        if not keep_rows:
            if max_frames is not None and n_kept >= max_frames:
                break
            continue

        idx_rows = torch.as_tensor(keep_rows, dtype=torch.long, device=device)
        batch_eval = {}
        for key, value in batch.items():
            if torch.is_tensor(value) and value.shape[:1] == (batch_size,):
                batch_eval[key] = value.index_select(0, idx_rows)
            else:
                batch_eval[key] = value

        dims = batch_eval.get("dimensions", None)
        positions = batch_eval["positions"]
        sum_logq_forward = torch.zeros(positions.shape[0], dtype=positions.dtype, device=positions.device)
        sum_logq_reverse = torch.zeros_like(sum_logq_forward)
        with torch.no_grad():
            u_from = model._eval_potential(state_from, positions, dims) / model._kT

        if direction == "forward":
            with torch.no_grad():
                mapped = model.forward(batch_eval)
                x_mapped = mapped["positions"]
                log_det_J = mapped["log_det_J"].reshape(-1)
                u_to_mapped = model._eval_potential(state_to, x_mapped, dims) / model._kT

            def target_reduced_potential(x: torch.Tensor) -> torch.Tensor:
                return model._eval_potential(state_to, x, dims) / model._kT

            context = KernelContext(reduced_potential=target_reduced_potential)
            x_current = x_mapped
            for _ in range(config.n_stochastic_steps):
                x_next, logq_f, aux = kernel.forward(x_current, context=context, rng=rng)
                logq_r = kernel.reverse_log_prob(x_current, x_next, context=context, aux_info=aux)
                sum_logq_forward = sum_logq_forward + logq_f.reshape(-1)
                sum_logq_reverse = sum_logq_reverse + logq_r.reshape(-1)
                x_current = x_next

            with torch.no_grad():
                u_to_snf = model._eval_potential(state_to, x_current, dims) / model._kT
                snf_work = compute_path_work(u_from, u_to_snf, log_det_J, sum_logq_forward, sum_logq_reverse)
                rmsd, max_disp = _displacement_stats(x_current, x_mapped, selected_flat_indices)
        elif direction == "inverse":
            # Reverse protocol for a forward path x_A -> M(x_A) -> stochastic B
            # samples the stochastic B-space variable first and only then applies
            # the inverse deterministic map. This preserves the path-probability
            # ratio used to derive the -logJ + logq_forward - logq_reverse work.
            def source_reduced_potential(x: torch.Tensor) -> torch.Tensor:
                return model._eval_potential(state_from, x, dims) / model._kT

            context = KernelContext(reduced_potential=source_reduced_potential)
            x_current = positions
            for _ in range(config.n_stochastic_steps):
                y_prev, logq_f, aux = kernel.reverse(x_current, context=context, rng=rng)
                logq_r = kernel.forward_log_prob(x_current, y_prev, context=context, aux_info=aux)
                sum_logq_forward = sum_logq_forward + logq_f.reshape(-1)
                sum_logq_reverse = sum_logq_reverse + logq_r.reshape(-1)
                x_current = y_prev

            inverse_batch = dict(batch_eval)
            inverse_batch["positions"] = x_current
            with torch.no_grad():
                mapped = model.inverse(inverse_batch)
                x_mapped = mapped["positions"]
                log_det_J = mapped["log_det_J"].reshape(-1)
                u_to_mapped = model._eval_potential(state_to, x_mapped, dims) / model._kT
                u_to_snf = u_to_mapped
                snf_work = compute_path_work(u_from, u_to_snf, log_det_J, sum_logq_forward, sum_logq_reverse)
                rmsd, max_disp = _displacement_stats(x_current, positions, selected_flat_indices)
        else:
            raise ValueError(f"Unsupported direction: {direction}")

        records["dataset_sample_index"].append(to_numpy(batch_eval["dataset_sample_index"]))
        records["trajectory_sample_index"].append(to_numpy(batch_eval["trajectory_sample_index"]))
        records["snf_work"].append(to_numpy(snf_work))
        records["u_from"].append(to_numpy(u_from))
        records["u_to_snf"].append(to_numpy(u_to_snf))
        records["u_to_mapped"].append(to_numpy(u_to_mapped))
        records["log_det_J"].append(to_numpy(log_det_J))
        records["sum_logq_forward"].append(to_numpy(sum_logq_forward))
        records["sum_logq_reverse"].append(to_numpy(sum_logq_reverse))
        records["path_log_weight"].append(to_numpy(-snf_work))
        records["mapped_to_snf_rmsd_angstrom"].append(to_numpy(rmsd))
        records["mapped_to_snf_max_disp_angstrom"].append(to_numpy(max_disp))

        if max_frames is not None and n_kept >= max_frames:
            break

    return {key: _finite_concat(values) for key, values in records.items()}


def metadata_from_config(config: MolecularStochasticConfig, selected_flat_indices: Optional[Sequence[int]]) -> Dict[str, Any]:
    return {
        "enabled": bool(config.enabled),
        "estimator": config.estimator,
        "kernel": config.kernel,
        "noise_sigma_coordinate_units": config.noise_sigma if config.noise_sigma is not None else 0.01,
        "step_size": config.step_size,
        "diffusion": config.diffusion,
        "gradient_policy": config.gradient_policy,
        "num_blocks": int(config.num_blocks),
        "steps_per_block": int(config.steps_per_block),
        "n_stochastic_steps": int(config.n_stochastic_steps),
        "seed": int(config.seed),
        "max_frames": config.max_frames,
        "every_n_frames": int(config.every_n_frames),
        "apply_to": config.apply_to,
        "selected_flat_coordinate_count": None if selected_flat_indices is None else int(len(selected_flat_indices)),
        "work_convention": "u_target-u_source-logJ+logq_forward-logq_reverse",
        "statistical_note": "These are path-weighted stochastic works, not deterministic TFEP works.",
        "molecular_ula_allowed": bool(config.allow_molecular_ula),
        "constrained_cartesian_ula_allowed": bool(config.allow_constrained_cartesian_ula),
        "constrained_cartesian_ula_warning": (
            "Cartesian ULA on constrained/PBC molecular endpoints is experimental and not a rigorously "
            "derived constrained-manifold transition density."
            if bool(config.allow_constrained_cartesian_ula) else None
        ),
    }
