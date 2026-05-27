"""Path data containers for stochastic TFEP."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import torch


@dataclass
class StochasticPathBatch:
    """Batch of stochastic paths with all terms needed to audit work values."""

    direction: str
    x0: torch.Tensor
    xK: torch.Tensor
    u_source_x0: torch.Tensor
    u_target_xK: torch.Tensor
    sum_logJ: torch.Tensor
    sum_logq_forward: torch.Tensor
    sum_logq_reverse: torch.Tensor
    path_work: torch.Tensor
    log_weight: torch.Tensor
    block_names: list[str] = field(default_factory=list)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    intermediates: Optional[dict[str, list[torch.Tensor]]] = None
    success: Optional[torch.Tensor] = None
    exception_messages: Optional[list[str]] = None

    def __post_init__(self) -> None:
        batch_size = self.x0.shape[0]
        fields = [
            self.xK,
            self.u_source_x0,
            self.u_target_xK,
            self.sum_logJ,
            self.sum_logq_forward,
            self.sum_logq_reverse,
            self.path_work,
            self.log_weight,
        ]
        for field_value in fields:
            if field_value.shape[0] != batch_size:
                raise ValueError("All path tensors must have the same batch dimension")
        if self.success is None:
            self.success = torch.ones(batch_size, dtype=torch.bool, device=self.x0.device)
        if self.exception_messages is None:
            self.exception_messages = [""] * batch_size

    @property
    def batch_size(self) -> int:
        return int(self.x0.shape[0])

    def detach(self) -> "StochasticPathBatch":
        """Return a detached copy suitable for serialization."""
        intermediates = None
        if self.intermediates is not None:
            intermediates = {
                key: [value.detach() for value in values]
                for key, values in self.intermediates.items()
            }
        return StochasticPathBatch(
            direction=self.direction,
            x0=self.x0.detach(),
            xK=self.xK.detach(),
            u_source_x0=self.u_source_x0.detach(),
            u_target_xK=self.u_target_xK.detach(),
            sum_logJ=self.sum_logJ.detach(),
            sum_logq_forward=self.sum_logq_forward.detach(),
            sum_logq_reverse=self.sum_logq_reverse.detach(),
            path_work=self.path_work.detach(),
            log_weight=self.log_weight.detach(),
            block_names=list(self.block_names),
            metadata=dict(self.metadata),
            intermediates=intermediates,
            success=self.success.detach() if self.success is not None else None,
            exception_messages=list(self.exception_messages or []),
        )
