"""Train-time stochastic path work utilities for bidirectional TFEP.

This module is intentionally opt-in.  It computes path-weighted stochastic
works for BAR/hybrid training while leaving deterministic TFEP/TMBAR behavior
unchanged unless a caller explicitly provides an enabled
``StochasticTrainingConfig``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import torch

from tfep.utils.misc import atom_to_flattened_indices

from .kernels import GaussianRandomWalkKernel, KernelContext, UnadjustedLangevinKernel
from .work import assert_finite_path_terms, compute_path_work


@dataclass(frozen=True)
class StochasticTrainingConfig:
    """Configuration for opt-in stochastic-path BAR/hybrid training."""

    enabled: bool = False
    kernel: str = "gaussian-rw"
    noise_sigma: Optional[float] = None
    step_size: Optional[float] = None
    diffusion: float = 1.0
    num_blocks: int = 1
    steps_per_block: int = 1
    train_mc_samples: int = 1
    gradient_policy: str = "stop-gradient"
    apply_to: str = "selected"
    allow_constrained_cartesian_ula: bool = False
    allow_molecular_ula: bool = False
    strict: bool = True

    @classmethod
    def from_any(cls, value: Optional[Any]) -> "StochasticTrainingConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls(**dict(value))
        raise TypeError("stochastic_training_config must be None, a mapping, or StochasticTrainingConfig")

    @property
    def n_stochastic_steps(self) -> int:
        return max(1, int(self.num_blocks)) * max(1, int(self.steps_per_block))

    def validate(self) -> None:
        if not self.enabled:
            return
        kernel = str(self.kernel).lower()
        if kernel not in {"none", "gaussian-rw", "ula"}:
            raise ValueError("Unsupported stochastic training kernel. Expected none, gaussian-rw, or ula.")
        if int(self.train_mc_samples) < 1:
            raise ValueError("train_mc_samples must be >= 1")
        if str(self.gradient_policy).lower() not in {"stop-gradient", "full"}:
            raise ValueError("gradient_policy must be 'stop-gradient' or 'full'")
        if kernel == "gaussian-rw":
            sigma = 0.01 if self.noise_sigma is None else float(self.noise_sigma)
            if sigma <= 0.0:
                raise ValueError("gaussian-rw stochastic training requires noise_sigma > 0")
        if kernel == "ula":
            if self.step_size is None or float(self.step_size) <= 0.0:
                raise ValueError("ULA stochastic training requires step_size > 0")
            if float(self.diffusion) <= 0.0:
                raise ValueError("ULA stochastic training requires diffusion > 0")
            if not (bool(self.allow_constrained_cartesian_ula) or bool(self.allow_molecular_ula)):
                raise RuntimeError(
                    "Molecular Cartesian ULA is experimental for constrained/PBC endpoints. "
                    "Set allow_constrained_cartesian_ula=True only after acknowledging this limitation."
                )


@dataclass
class StochasticTrainingWorks:
    """Bidirectional stochastic path work tensors and diagnostics."""

    w01: torch.Tensor
    w10: torch.Tensor
    logJ01: torch.Tensor
    logJ10: torch.Tensor
    sum_logq_forward_01: torch.Tensor
    sum_logq_reverse_01: torch.Tensor
    sum_logq_forward_10: torch.Tensor
    sum_logq_reverse_10: torch.Tensor
    u1_xK: torch.Tensor
    u0_xK: torch.Tensor
    mapped_to_snf_rmsd_01: torch.Tensor
    mapped_to_snf_rmsd_10: torch.Tensor
    mapped_to_snf_max_disp_01: torch.Tensor
    mapped_to_snf_max_disp_10: torch.Tensor

    def metrics(self) -> dict[str, torch.Tensor]:
        """Return scalar train metrics safe for Lightning logging."""
        return {
            "snf_logq_forward_01_mean": self.sum_logq_forward_01.mean(),
            "snf_logq_reverse_01_mean": self.sum_logq_reverse_01.mean(),
            "snf_logq_forward_10_mean": self.sum_logq_forward_10.mean(),
            "snf_logq_reverse_10_mean": self.sum_logq_reverse_10.mean(),
            "snf_work_01_mean": self.w01.mean(),
            "snf_work_10_mean": self.w10.mean(),
            "snf_mapped_to_snf_rmsd_01_mean": self.mapped_to_snf_rmsd_01.mean(),
            "snf_mapped_to_snf_rmsd_10_mean": self.mapped_to_snf_rmsd_10.mean(),
            "snf_mapped_to_snf_max_disp_01_mean": self.mapped_to_snf_max_disp_01.mean(),
            "snf_mapped_to_snf_max_disp_10_mean": self.mapped_to_snf_max_disp_10.mean(),
        }


def resolve_training_flat_indices(model: Any, config: StochasticTrainingConfig) -> Optional[tuple[int, ...]]:
    """Resolve flattened coordinate indices for train-time stochastic kernels.

    ``None`` means all coordinates.  The default ``selected`` and ``solute``
    modes use the model's mapped atoms, which keeps the first molecular
    implementation focused on the solute mapping.
    """
    apply_to = str(config.apply_to).lower()
    if apply_to == "all":
        return None
    if apply_to == "shell":
        raise ValueError("Train-time --snf-apply-to shell requires a dedicated shell index resolver; use selected for now.")
    if apply_to not in {"selected", "solute"}:
        raise ValueError("stochastic training apply_to must be all, selected, solute, or shell")
    atom_indices = model.get_mapped_indices(idx_type="atom", remove_fixed=False)
    if not torch.is_tensor(atom_indices):
        atom_indices = torch.as_tensor(atom_indices, dtype=torch.long)
    flat = atom_to_flattened_indices(atom_indices.detach().cpu().long())
    return tuple(int(i) for i in flat.reshape(-1).tolist())


def _make_kernel(config: StochasticTrainingConfig, selected_flat_indices: Optional[Sequence[int]]):
    kernel_name = str(config.kernel).lower()
    if kernel_name == "none":
        return None
    if kernel_name == "gaussian-rw":
        sigma = 0.01 if config.noise_sigma is None else float(config.noise_sigma)
        return GaussianRandomWalkKernel(sigma=sigma, selected_indices=selected_flat_indices)
    if kernel_name == "ula":
        return UnadjustedLangevinKernel(
            step_size=float(config.step_size),
            diffusion=float(config.diffusion),
            selected_indices=selected_flat_indices,
            gradient_policy=str(config.gradient_policy).lower(),
        )
    raise ValueError(f"Unsupported stochastic training kernel: {config.kernel}")


def _zeros_like_batch(x: torch.Tensor) -> torch.Tensor:
    return torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)


def _displacement_stats(x_final: torch.Tensor, x_ref: torch.Tensor, selected_flat_indices: Optional[Sequence[int]]):
    if selected_flat_indices is not None:
        idx = torch.as_tensor(selected_flat_indices, dtype=torch.long, device=x_final.device)
        x_final = x_final.index_select(1, idx)
        x_ref = x_ref.index_select(1, idx)
    disp = (x_final - x_ref).reshape(x_final.shape[0], -1, 3)
    per_atom = torch.linalg.norm(disp, dim=-1)
    return torch.sqrt(torch.mean(per_atom.pow(2), dim=1)), torch.amax(per_atom, dim=1)


def _eval_reduced(model: Any, state: int, positions: torch.Tensor, dimensions: Optional[torch.Tensor]) -> torch.Tensor:
    return model._eval_potential(int(state), positions, dimensions) / model._kT


def _forward_path(
    *,
    model: Any,
    batch0: Mapping[str, torch.Tensor],
    mapped01: Mapping[str, torch.Tensor],
    u0_x0: torch.Tensor,
    config: StochasticTrainingConfig,
    selected_flat_indices: Optional[Sequence[int]],
):
    dims = batch0.get("dimensions", None)
    x_mapped = mapped01["positions"]
    logJ = mapped01["log_det_J"].reshape(-1)
    kernel = _make_kernel(config, selected_flat_indices)
    sum_logq_forward = _zeros_like_batch(x_mapped)
    sum_logq_reverse = torch.zeros_like(sum_logq_forward)

    def target_reduced_potential(x: torch.Tensor) -> torch.Tensor:
        return _eval_reduced(model, 1, x, dims)

    context = KernelContext(
        reduced_potential=target_reduced_potential,
        gradient_policy=str(config.gradient_policy).lower(),
        metadata={"state": 1, "direction": "0_to_1"},
    )
    x_current = x_mapped
    if kernel is not None:
        for _ in range(config.n_stochastic_steps):
            x_next, logq_f, aux = kernel.forward(x_current, context=context, rng=None)
            logq_r = kernel.reverse_log_prob(x_current, x_next, context=context, aux_info=aux)
            sum_logq_forward = sum_logq_forward + logq_f.reshape(-1)
            sum_logq_reverse = sum_logq_reverse + logq_r.reshape(-1)
            x_current = x_next
    u1_xK = target_reduced_potential(x_current)
    work = compute_path_work(u0_x0, u1_xK, logJ, sum_logq_forward, sum_logq_reverse)
    rmsd, max_disp = _displacement_stats(x_current, x_mapped, selected_flat_indices)
    return work, logJ, sum_logq_forward, sum_logq_reverse, u1_xK, rmsd, max_disp


def _reverse_path(
    *,
    model: Any,
    batch1: Mapping[str, torch.Tensor],
    u1_x1: torch.Tensor,
    config: StochasticTrainingConfig,
    selected_flat_indices: Optional[Sequence[int]],
):
    dims = batch1.get("dimensions", None)
    positions = batch1["positions"]
    kernel = _make_kernel(config, selected_flat_indices)
    sum_logq_forward = _zeros_like_batch(positions)
    sum_logq_reverse = torch.zeros_like(sum_logq_forward)

    def source_reduced_potential(x: torch.Tensor) -> torch.Tensor:
        return _eval_reduced(model, 1, x, dims)

    context = KernelContext(
        reduced_potential=source_reduced_potential,
        gradient_policy=str(config.gradient_policy).lower(),
        metadata={"state": 1, "direction": "1_to_0_reverse_kernel"},
    )
    y_current = positions
    if kernel is not None:
        for _ in range(config.n_stochastic_steps):
            y_prev, logq_generating, aux = kernel.reverse(y_current, context=context, rng=None)
            logq_counter = kernel.forward_log_prob(y_current, y_prev, context=context, aux_info=aux)
            sum_logq_forward = sum_logq_forward + logq_generating.reshape(-1)
            sum_logq_reverse = sum_logq_reverse + logq_counter.reshape(-1)
            y_current = y_prev

    inverse_batch = dict(batch1)
    inverse_batch["positions"] = y_current
    mapped10 = model.inverse(inverse_batch)
    x_a = mapped10["positions"]
    logJ = mapped10["log_det_J"].reshape(-1)
    u0_xK = _eval_reduced(model, 0, x_a, dims)
    work = compute_path_work(u1_x1, u0_xK, logJ, sum_logq_forward, sum_logq_reverse)
    rmsd, max_disp = _displacement_stats(y_current, positions, selected_flat_indices)
    return work, logJ, sum_logq_forward, sum_logq_reverse, u0_xK, rmsd, max_disp


def compute_bidirectional_stochastic_training_works(
    *,
    model: Any,
    batch0: Mapping[str, torch.Tensor],
    batch1: Mapping[str, torch.Tensor],
    mapped01: Mapping[str, torch.Tensor],
    u0_x0: torch.Tensor,
    u1_x1: torch.Tensor,
    config: StochasticTrainingConfig,
) -> StochasticTrainingWorks:
    """Compute stochastic path works for one bidirectional training minibatch."""
    config = StochasticTrainingConfig.from_any(config)
    config.validate()
    selected_flat_indices = resolve_training_flat_indices(model, config)

    chunks: dict[str, list[torch.Tensor]] = {
        "w01": [], "w10": [], "logJ01": [], "logJ10": [],
        "sum_logq_forward_01": [], "sum_logq_reverse_01": [],
        "sum_logq_forward_10": [], "sum_logq_reverse_10": [],
        "u1_xK": [], "u0_xK": [],
        "rmsd01": [], "rmsd10": [], "max01": [], "max10": [],
    }
    for _ in range(max(1, int(config.train_mc_samples))):
        f = _forward_path(
            model=model,
            batch0=batch0,
            mapped01=mapped01,
            u0_x0=u0_x0,
            config=config,
            selected_flat_indices=selected_flat_indices,
        )
        r = _reverse_path(
            model=model,
            batch1=batch1,
            u1_x1=u1_x1,
            config=config,
            selected_flat_indices=selected_flat_indices,
        )
        w01, logJ01, lqf01, lqr01, u1_xK, rmsd01, max01 = f
        w10, logJ10, lqf10, lqr10, u0_xK, rmsd10, max10 = r
        chunks["w01"].append(w01)
        chunks["w10"].append(w10)
        chunks["logJ01"].append(logJ01)
        chunks["logJ10"].append(logJ10)
        chunks["sum_logq_forward_01"].append(lqf01)
        chunks["sum_logq_reverse_01"].append(lqr01)
        chunks["sum_logq_forward_10"].append(lqf10)
        chunks["sum_logq_reverse_10"].append(lqr10)
        chunks["u1_xK"].append(u1_xK)
        chunks["u0_xK"].append(u0_xK)
        chunks["rmsd01"].append(rmsd01)
        chunks["rmsd10"].append(rmsd10)
        chunks["max01"].append(max01)
        chunks["max10"].append(max10)

    result = StochasticTrainingWorks(
        w01=torch.cat(chunks["w01"]),
        w10=torch.cat(chunks["w10"]),
        logJ01=torch.cat(chunks["logJ01"]),
        logJ10=torch.cat(chunks["logJ10"]),
        sum_logq_forward_01=torch.cat(chunks["sum_logq_forward_01"]),
        sum_logq_reverse_01=torch.cat(chunks["sum_logq_reverse_01"]),
        sum_logq_forward_10=torch.cat(chunks["sum_logq_forward_10"]),
        sum_logq_reverse_10=torch.cat(chunks["sum_logq_reverse_10"]),
        u1_xK=torch.cat(chunks["u1_xK"]),
        u0_xK=torch.cat(chunks["u0_xK"]),
        mapped_to_snf_rmsd_01=torch.cat(chunks["rmsd01"]),
        mapped_to_snf_rmsd_10=torch.cat(chunks["rmsd10"]),
        mapped_to_snf_max_disp_01=torch.cat(chunks["max01"]),
        mapped_to_snf_max_disp_10=torch.cat(chunks["max10"]),
    )
    assert_finite_path_terms(
        snf_w01=result.w01,
        snf_w10=result.w10,
        sum_logq_forward_01=result.sum_logq_forward_01,
        sum_logq_reverse_01=result.sum_logq_reverse_01,
        sum_logq_forward_10=result.sum_logq_forward_10,
        sum_logq_reverse_10=result.sum_logq_reverse_10,
    )
    return result
