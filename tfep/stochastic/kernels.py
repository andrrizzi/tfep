"""Stochastic transition kernels for path-weighted TFEP.

The kernels in this module return exact log transition densities for the
coordinates they perturb. They do not modify deterministic TFEP work arrays;
they are building blocks for an explicit path-weighted estimator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence, Union

import math
import torch


TensorCallable = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class KernelContext:
    """Context required by a stochastic kernel.

    Parameters
    ----------
    reduced_potential : callable, optional
        Function returning reduced potentials ``u(x)`` with shape ``(batch,)``.
    reduced_potential_gradient : callable, optional
        Function returning ``grad_x u(x)`` in the same coordinate units used by
        the kernel. If omitted, gradients are computed with torch autograd from
        ``reduced_potential``. This is intended for evaluation/toy systems; it
        intentionally avoids any production molecular assumptions.
    gradient_policy : str, optional
        ``"stop-gradient"`` detaches reduced-potential gradients before they are
        used in the transition mean. ``"full"`` keeps the autograd graph and may
        require second derivatives through the potential backend.
    metadata : mapping, optional
        Reproducibility metadata such as units, lambda values, or backend names.
    """

    reduced_potential: Optional[TensorCallable] = None
    reduced_potential_gradient: Optional[TensorCallable] = None
    gradient_policy: str = "stop-gradient"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def gradient(self, x: torch.Tensor, gradient_policy: Optional[str] = None) -> torch.Tensor:
        """Return the gradient of the reduced potential at ``x``.

        This method is deliberately explicit about reduced-potential gradients.
        Molecular callers must ensure units and constraints make the transition
        density valid before using gradient-based kernels.
        """
        policy = str(gradient_policy or self.gradient_policy).lower()
        if policy not in {"stop-gradient", "full"}:
            raise ValueError("gradient_policy must be 'stop-gradient' or 'full'")

        if self.reduced_potential_gradient is not None:
            grad = self.reduced_potential_gradient(x)
            if grad.shape != x.shape:
                raise ValueError("reduced_potential_gradient must return a tensor with x.shape")
            return grad if policy == "full" else grad.detach()
        if self.reduced_potential is None:
            raise ValueError("A reduced potential or reduced-potential gradient is required")

        with torch.enable_grad():
            if policy == "full":
                x_req = x
                if not bool(x_req.requires_grad):
                    x_req = x.detach().clone().requires_grad_(True)
            else:
                x_req = x.detach().clone().requires_grad_(True)
            u = self.reduced_potential(x_req).reshape(-1)
            if u.shape[0] != x.shape[0]:
                raise ValueError("reduced_potential must return one value per batch element")
            grad, = torch.autograd.grad(
                u.sum(),
                x_req,
                create_graph=(policy == "full"),
                retain_graph=(policy == "full"),
            )
        return grad if policy == "full" else grad.detach()


class StochasticKernel(ABC):
    """Abstract stochastic kernel with explicit forward/reverse densities."""

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[KernelContext] = None,
        rng: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Sample ``x_next`` and return ``(x_next, log_q_forward, aux_info)``."""

    @abstractmethod
    def reverse(
        self,
        x_next: torch.Tensor,
        context: Optional[KernelContext] = None,
        rng: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Sample the reverse stochastic variable and its log probability."""

    @abstractmethod
    def forward_log_prob(
        self,
        x_next: torch.Tensor,
        x: torch.Tensor,
        context: Optional[KernelContext] = None,
        aux_info: Optional[Mapping[str, Any]] = None,
    ) -> torch.Tensor:
        """Return ``log K^F(x_next | x)``."""

    @abstractmethod
    def reverse_log_prob(
        self,
        x: torch.Tensor,
        x_next: torch.Tensor,
        context: Optional[KernelContext] = None,
        aux_info: Optional[Mapping[str, Any]] = None,
    ) -> torch.Tensor:
        """Return ``log K^R(x | x_next)``."""

    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        """Return serializable metadata needed for reproducibility."""


def _selected(x: torch.Tensor, selected_indices: Optional[Sequence[int]]) -> torch.Tensor:
    if selected_indices is None:
        return x
    idx = torch.as_tensor(selected_indices, dtype=torch.long, device=x.device)
    return x.index_select(dim=1, index=idx)


def _replace_selected(
    base: torch.Tensor,
    values: torch.Tensor,
    selected_indices: Optional[Sequence[int]],
) -> torch.Tensor:
    if selected_indices is None:
        return values
    idx = torch.as_tensor(selected_indices, dtype=torch.long, device=base.device)
    out = base.clone()
    out[:, idx] = values
    return out


def _randn_like(x: torch.Tensor, rng: Optional[torch.Generator]) -> torch.Tensor:
    return torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=rng)


def _gaussian_log_prob(value: torch.Tensor, mean: torch.Tensor, variance: Union[torch.Tensor, float]) -> torch.Tensor:
    if value.shape != mean.shape:
        raise ValueError("value and mean must have the same shape")
    if value.ndim < 2:
        raise ValueError("expected a batched tensor with shape (batch, features)")

    variance_tensor = torch.as_tensor(variance, dtype=value.dtype, device=value.device)
    if torch.any(variance_tensor <= 0):
        raise ValueError("Gaussian variance must be positive")

    diff = (value - mean).reshape(value.shape[0], -1)
    n_features = diff.shape[1]
    quad = diff.pow(2).sum(dim=1) / variance_tensor
    norm = n_features * torch.log(2.0 * torch.as_tensor(math.pi, dtype=value.dtype, device=value.device) * variance_tensor)
    return -0.5 * (quad + norm)


class GaussianRandomWalkKernel(StochasticKernel):
    """Symmetric Gaussian random-walk kernel.

    This kernel is mathematically simple and is the preferred first kernel for
    path-probability validation. It is not intended to be a physically optimal
    molecular relaxation model.
    """

    def __init__(self, sigma: float, selected_indices: Optional[Sequence[int]] = None) -> None:
        if sigma <= 0.0:
            raise ValueError("sigma must be positive")
        self.sigma = float(sigma)
        self.selected_indices = None if selected_indices is None else tuple(int(i) for i in selected_indices)

    @property
    def variance(self) -> float:
        return self.sigma * self.sigma

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[KernelContext] = None,
        rng: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        x_active = _selected(x, self.selected_indices)
        x_next_active = x_active + self.sigma * _randn_like(x_active, rng)
        x_next = _replace_selected(x, x_next_active, self.selected_indices)
        logq = _gaussian_log_prob(x_next_active, x_active, self.variance)
        return x_next, logq, {"kernel": "gaussian-rw"}

    def reverse(
        self,
        x_next: torch.Tensor,
        context: Optional[KernelContext] = None,
        rng: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        x_next_active = _selected(x_next, self.selected_indices)
        x_active = x_next_active + self.sigma * _randn_like(x_next_active, rng)
        x = _replace_selected(x_next, x_active, self.selected_indices)
        logq = _gaussian_log_prob(x_active, x_next_active, self.variance)
        return x, logq, {"kernel": "gaussian-rw", "reverse": True}

    def forward_log_prob(
        self,
        x_next: torch.Tensor,
        x: torch.Tensor,
        context: Optional[KernelContext] = None,
        aux_info: Optional[Mapping[str, Any]] = None,
    ) -> torch.Tensor:
        return _gaussian_log_prob(
            _selected(x_next, self.selected_indices),
            _selected(x, self.selected_indices),
            self.variance,
        )

    def reverse_log_prob(
        self,
        x: torch.Tensor,
        x_next: torch.Tensor,
        context: Optional[KernelContext] = None,
        aux_info: Optional[Mapping[str, Any]] = None,
    ) -> torch.Tensor:
        return _gaussian_log_prob(
            _selected(x, self.selected_indices),
            _selected(x_next, self.selected_indices),
            self.variance,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "kernel": "gaussian-rw",
            "sigma": self.sigma,
            "selected_indices": self.selected_indices,
            "transition_density": "cartesian_isotropic_gaussian",
        }


class UnadjustedLangevinKernel(StochasticKernel):
    """Unadjusted Langevin kernel with explicit Gaussian transition density.

    The update is written in reduced-potential coordinates:

    ``x_next = x - step_size * diffusion * grad u(x) + sqrt(2*diffusion*step_size) * eta``.

    This implementation is safe for toy models and selected unconstrained
    Cartesian coordinates. It does not make constrained/PBC molecular dynamics
    valid by itself.
    """

    def __init__(
        self,
        step_size: float,
        diffusion: float = 1.0,
        selected_indices: Optional[Sequence[int]] = None,
        gradient_policy: str = "stop-gradient",
    ) -> None:
        if step_size <= 0.0:
            raise ValueError("step_size must be positive")
        if diffusion <= 0.0:
            raise ValueError("diffusion must be positive")
        if str(gradient_policy).lower() not in {"stop-gradient", "full"}:
            raise ValueError("gradient_policy must be 'stop-gradient' or 'full'")
        self.step_size = float(step_size)
        self.diffusion = float(diffusion)
        self.selected_indices = None if selected_indices is None else tuple(int(i) for i in selected_indices)
        self.gradient_policy = str(gradient_policy).lower()

    @property
    def variance(self) -> float:
        return 2.0 * self.diffusion * self.step_size

    def _mean(self, x: torch.Tensor, context: Optional[KernelContext]) -> torch.Tensor:
        if context is None:
            raise ValueError("UnadjustedLangevinKernel requires a KernelContext")
        grad = context.gradient(x, gradient_policy=self.gradient_policy)
        mean = x.clone()
        if self.selected_indices is None:
            return x - self.step_size * self.diffusion * grad
        idx = torch.as_tensor(self.selected_indices, dtype=torch.long, device=x.device)
        mean[:, idx] = x[:, idx] - self.step_size * self.diffusion * grad[:, idx]
        return mean

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[KernelContext] = None,
        rng: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        mean = self._mean(x, context)
        mean_active = _selected(mean, self.selected_indices)
        x_next_active = mean_active + math.sqrt(self.variance) * _randn_like(mean_active, rng)
        x_next = _replace_selected(x, x_next_active, self.selected_indices)
        logq = _gaussian_log_prob(x_next_active, mean_active, self.variance)
        return x_next, logq, {"kernel": "ula"}

    def reverse(
        self,
        x_next: torch.Tensor,
        context: Optional[KernelContext] = None,
        rng: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        mean = self._mean(x_next, context)
        mean_active = _selected(mean, self.selected_indices)
        x_active = mean_active + math.sqrt(self.variance) * _randn_like(mean_active, rng)
        x = _replace_selected(x_next, x_active, self.selected_indices)
        logq = _gaussian_log_prob(x_active, mean_active, self.variance)
        return x, logq, {"kernel": "ula", "reverse": True}

    def forward_log_prob(
        self,
        x_next: torch.Tensor,
        x: torch.Tensor,
        context: Optional[KernelContext] = None,
        aux_info: Optional[Mapping[str, Any]] = None,
    ) -> torch.Tensor:
        mean = self._mean(x, context)
        return _gaussian_log_prob(
            _selected(x_next, self.selected_indices),
            _selected(mean, self.selected_indices),
            self.variance,
        )

    def reverse_log_prob(
        self,
        x: torch.Tensor,
        x_next: torch.Tensor,
        context: Optional[KernelContext] = None,
        aux_info: Optional[Mapping[str, Any]] = None,
    ) -> torch.Tensor:
        mean = self._mean(x_next, context)
        return _gaussian_log_prob(
            _selected(x, self.selected_indices),
            _selected(mean, self.selected_indices),
            self.variance,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "kernel": "ula",
            "step_size": self.step_size,
            "diffusion": self.diffusion,
            "variance": self.variance,
            "gradient_policy": self.gradient_policy,
            "selected_indices": self.selected_indices,
            "transition_density": "cartesian_isotropic_gaussian_with_reduced_potential_drift",
            "limitations": [
                "not valid for constrained coordinates unless the constrained density is derived",
                "not valid across periodic boundaries unless PBC path density is handled explicitly",
            ],
        }
