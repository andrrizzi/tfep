"""Protocol blocks for stochastic path-weighted TFEP."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Union

import torch

from .kernels import KernelContext, StochasticKernel


@dataclass
class BlockResult:
    """Single stochastic-flow-block result for a batch of paths."""

    x_next: torch.Tensor
    y: torch.Tensor
    log_det_J: torch.Tensor
    logq_forward: torch.Tensor
    logq_reverse: torch.Tensor
    aux_info: dict[str, Any] = field(default_factory=dict)


class DeterministicFlowBlock:
    """Adapter around an existing deterministic invertible TFEP flow."""

    def __init__(self, flow: torch.nn.Module, name: Optional[str] = None) -> None:
        self.flow = flow
        self.name = name or flow.__class__.__name__

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y, logj = self.flow.forward(x)
        return y, logj.reshape(-1)

    def inverse(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, logj = self.flow.inverse(y)
        return x, logj.reshape(-1)

    def metadata(self) -> dict[str, Any]:
        return {"name": self.name, "class": self.flow.__class__.__name__}


class IdentityFlowBlock:
    """Identity deterministic block with zero log-Jacobian."""

    name = "identity"

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x, torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)

    def inverse(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return y, torch.zeros(y.shape[0], dtype=y.dtype, device=y.device)

    def metadata(self) -> dict[str, Any]:
        return {"name": self.name, "class": self.__class__.__name__}


class AffineFlowBlock:
    """Simple diagonal affine block for toy-model validation."""

    name = "affine"

    def __init__(self, scale: Union[float, Sequence[float], torch.Tensor], shift: Union[float, Sequence[float], torch.Tensor] = 0.0) -> None:
        scale_tensor = torch.as_tensor(scale)
        if torch.any(scale_tensor == 0):
            raise ValueError("scale must be nonzero")
        self.scale = scale_tensor
        self.shift = torch.as_tensor(shift)

    def _params(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale = self.scale.to(dtype=x.dtype, device=x.device)
        shift = self.shift.to(dtype=x.dtype, device=x.device)
        return scale, shift

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale, shift = self._params(x)
        y = x * scale + shift
        logj_scalar = torch.log(torch.abs(scale)).sum() if scale.ndim > 0 else x.shape[1] * torch.log(torch.abs(scale))
        return y, torch.full((x.shape[0],), logj_scalar, dtype=x.dtype, device=x.device)

    def inverse(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale, shift = self._params(y)
        x = (y - shift) / scale
        logj_scalar = -torch.log(torch.abs(scale)).sum() if scale.ndim > 0 else -y.shape[1] * torch.log(torch.abs(scale))
        return x, torch.full((y.shape[0],), logj_scalar, dtype=y.dtype, device=y.device)

    def metadata(self) -> dict[str, Any]:
        return {"name": self.name, "class": self.__class__.__name__}


class StochasticFlowBlock:
    """One deterministic invertible block optionally followed by a stochastic kernel."""

    def __init__(
        self,
        deterministic_flow: Union[DeterministicFlowBlock, IdentityFlowBlock, AffineFlowBlock],
        kernel: Optional[StochasticKernel] = None,
        name: Optional[str] = None,
    ) -> None:
        self.deterministic_flow = deterministic_flow
        self.kernel = kernel
        self.name = name or getattr(deterministic_flow, "name", deterministic_flow.__class__.__name__)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[KernelContext] = None,
        rng: Optional[torch.Generator] = None,
    ) -> BlockResult:
        y, logj = self.deterministic_flow.forward(x)
        if self.kernel is None:
            zeros = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
            return BlockResult(x_next=y, y=y, log_det_J=logj, logq_forward=zeros, logq_reverse=zeros)
        x_next, logq_forward, aux = self.kernel.forward(y, context=context, rng=rng)
        logq_reverse = self.kernel.reverse_log_prob(y, x_next, context=context, aux_info=aux)
        return BlockResult(
            x_next=x_next,
            y=y,
            log_det_J=logj,
            logq_forward=logq_forward.reshape(-1),
            logq_reverse=logq_reverse.reshape(-1),
            aux_info=aux,
        )

    def reverse(
        self,
        x_next: torch.Tensor,
        context: Optional[KernelContext] = None,
        rng: Optional[torch.Generator] = None,
    ) -> BlockResult:
        if self.kernel is None:
            y = x_next
            logq_generating = torch.zeros(x_next.shape[0], dtype=x_next.dtype, device=x_next.device)
            logq_counter = torch.zeros_like(logq_generating)
            aux: dict[str, Any] = {}
        else:
            y, logq_generating, aux = self.kernel.reverse(x_next, context=context, rng=rng)
            logq_counter = self.kernel.forward_log_prob(x_next, y, context=context, aux_info=aux)
        x_prev, logj_inverse = self.deterministic_flow.inverse(y)
        return BlockResult(
            x_next=x_prev,
            y=y,
            log_det_J=logj_inverse.reshape(-1),
            logq_forward=logq_generating.reshape(-1),
            logq_reverse=logq_counter.reshape(-1),
            aux_info=aux,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "deterministic_flow": self.deterministic_flow.metadata(),
            "kernel": None if self.kernel is None else self.kernel.metadata(),
        }
