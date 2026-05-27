"""Evaluation-only path-weighted stochastic TFEP estimator."""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import torch

from .kernels import KernelContext
from .path import StochasticPathBatch
from .protocols import StochasticFlowBlock
from .work import assert_finite_path_terms, compute_path_work

ReducedPotential = Callable[[torch.Tensor], torch.Tensor]


class PathWeightedTFEP:
    """Experimental evaluator for path-weighted stochastic TFEP.

    This class deliberately does not inherit from ``TFEPMapBase``. It computes
    path-weighted work values separately from the deterministic production
    estimator so stochastic coordinates cannot be consumed by BAR/TMBAR without
    the required transition-density terms.
    """

    def __init__(
        self,
        blocks: Sequence[StochasticFlowBlock],
        source_potential: ReducedPotential,
        target_potential: ReducedPotential,
        kernel_contexts: Optional[Sequence[Optional[KernelContext]]] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        self.blocks = list(blocks)
        self.source_potential = source_potential
        self.target_potential = target_potential
        self.kernel_contexts = list(kernel_contexts) if kernel_contexts is not None else [None] * len(self.blocks)
        if len(self.kernel_contexts) != len(self.blocks):
            raise ValueError("kernel_contexts must have one entry per block")
        self.metadata = dict(metadata or {})

    def sample_forward(
        self,
        x0: torch.Tensor,
        rng: Optional[torch.Generator] = None,
        save_intermediates: bool = False,
        strict: bool = True,
    ) -> StochasticPathBatch:
        """Generate A->B paths and compute path-weighted ``w01``."""
        return self._sample_direction(
            initial=x0,
            source_potential=self.source_potential,
            target_potential=self.target_potential,
            direction="A_to_B",
            reverse=False,
            rng=rng,
            save_intermediates=save_intermediates,
            strict=strict,
        )

    def sample_reverse(
        self,
        xK: torch.Tensor,
        rng: Optional[torch.Generator] = None,
        save_intermediates: bool = False,
        strict: bool = True,
    ) -> StochasticPathBatch:
        """Generate B->A paths and compute path-weighted ``w10`` convention."""
        return self._sample_direction(
            initial=xK,
            source_potential=self.target_potential,
            target_potential=self.source_potential,
            direction="B_to_A",
            reverse=True,
            rng=rng,
            save_intermediates=save_intermediates,
            strict=strict,
        )

    def _sample_direction(
        self,
        initial: torch.Tensor,
        source_potential: ReducedPotential,
        target_potential: ReducedPotential,
        direction: str,
        reverse: bool,
        rng: Optional[torch.Generator],
        save_intermediates: bool,
        strict: bool,
    ) -> StochasticPathBatch:
        x = initial
        batch_size = x.shape[0]
        sum_logJ = torch.zeros(batch_size, dtype=x.dtype, device=x.device)
        sum_logq_forward = torch.zeros_like(sum_logJ)
        sum_logq_reverse = torch.zeros_like(sum_logJ)
        block_names: list[str] = []
        intermediates = {"x": [x.detach().clone()], "y": []} if save_intermediates else None

        block_iter = list(enumerate(self.blocks))
        if reverse:
            block_iter = list(reversed(block_iter))

        try:
            for idx, block in block_iter:
                context = self.kernel_contexts[idx]
                result = block.reverse(x, context=context, rng=rng) if reverse else block.forward(x, context=context, rng=rng)
                x = result.x_next
                sum_logJ = sum_logJ + result.log_det_J
                sum_logq_forward = sum_logq_forward + result.logq_forward
                sum_logq_reverse = sum_logq_reverse + result.logq_reverse
                block_names.append(block.name)
                if intermediates is not None:
                    intermediates["y"].append(result.y.detach().clone())
                    intermediates["x"].append(x.detach().clone())
        except Exception:
            if strict:
                raise
            success = torch.zeros(batch_size, dtype=torch.bool, device=initial.device)
            empty = torch.full((batch_size,), float("nan"), dtype=initial.dtype, device=initial.device)
            return StochasticPathBatch(
                direction=direction,
                x0=initial,
                xK=x,
                u_source_x0=empty,
                u_target_xK=empty,
                sum_logJ=sum_logJ,
                sum_logq_forward=sum_logq_forward,
                sum_logq_reverse=sum_logq_reverse,
                path_work=empty,
                log_weight=empty,
                block_names=block_names,
                metadata=self._metadata(),
                intermediates=intermediates,
                success=success,
                exception_messages=["path generation failed"] * batch_size,
            )

        u_source = source_potential(initial).reshape(-1)
        u_target = target_potential(x).reshape(-1)
        path_work = compute_path_work(u_source, u_target, sum_logJ, sum_logq_forward, sum_logq_reverse)
        assert_finite_path_terms(
            u_source_x0=u_source,
            u_target_xK=u_target,
            sum_logJ=sum_logJ,
            sum_logq_forward=sum_logq_forward,
            sum_logq_reverse=sum_logq_reverse,
            path_work=path_work,
        )
        return StochasticPathBatch(
            direction=direction,
            x0=initial,
            xK=x,
            u_source_x0=u_source,
            u_target_xK=u_target,
            sum_logJ=sum_logJ,
            sum_logq_forward=sum_logq_forward,
            sum_logq_reverse=sum_logq_reverse,
            path_work=path_work,
            log_weight=-path_work,
            block_names=block_names,
            metadata=self._metadata(),
            intermediates=intermediates,
        )

    def _metadata(self) -> dict:
        metadata = dict(self.metadata)
        metadata.update(
            {
                "estimator": "path-weighted-stochastic-tfep",
                "experimental": True,
                "work_convention": "u_target-u_source-logJ+logq_forward-logq_reverse",
                "blocks": [block.metadata() for block in self.blocks],
            }
        )
        return metadata
