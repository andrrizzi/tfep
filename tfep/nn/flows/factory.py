"""Developer-facing flow factory.

This module provides a small registry to create flows from a declarative spec.
The goal is to make it easy to swap flows at the *developer* level without
modifying map code.

Currently ships with:
- cartesian_maf: sequential masked autoregressive flows (MAF)
- triatomic_zmat: internal-coordinate flow for triatomics (gas phase)
- wrappers: centered_centroid, oriented

You can register new builders by calling ``FlowFactory.register(name, fn)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch

from tfep.nn.conditioners import generate_degrees
from tfep.nn.flows import MAF, OrientedFlow, CenteredCentroidFlow, SequentialFlow

from .triatomic_zmatrix import TriatomicZMatrixFlow, VectorCouplingFlow


@dataclass
class FlowSpec:
    """Declarative flow spec.

    Parameters
    ----------
    name
        Registered flow name.
    kwargs
        Passed to the builder.
    wrappers
        Optional list of wrapper specs applied *outside* the built flow.
        Wrapper specs use the same FlowSpec class but must have a registered
        wrapper builder (e.g., "oriented").
    """

    name: str
    kwargs: Dict[str, Any] = field(default_factory=dict)
    wrappers: List["FlowSpec"] = field(default_factory=list)


BuilderFn = Callable[[FlowSpec, Dict[str, Any]], torch.nn.Module]


class FlowFactory:
    """Registry-based flow factory."""

    _REGISTRY: Dict[str, BuilderFn] = {}

    @classmethod
    def register(cls, name: str, fn: BuilderFn) -> None:
        cls._REGISTRY[str(name)] = fn

    @classmethod
    def build(cls, spec: FlowSpec, context: Optional[Dict[str, Any]] = None) -> torch.nn.Module:
        context = {} if context is None else dict(context)
        if spec.name not in cls._REGISTRY:
            raise KeyError(
                f"Unknown flow spec name={spec.name!r}. Registered: {sorted(cls._REGISTRY)}"
            )

        flow = cls._REGISTRY[spec.name](spec, context)

        # Apply wrappers (outside-in): base flow first, then wrappers in order.
        for wspec in spec.wrappers:
            if wspec.name not in cls._REGISTRY:
                raise KeyError(
                    f"Unknown wrapper spec name={wspec.name!r}. Registered: {sorted(cls._REGISTRY)}"
                )
            # Wrapper builder receives the current flow in context.
            wctx = dict(context)
            wctx["flow"] = flow
            flow = cls._REGISTRY[wspec.name](wspec, wctx)

        return flow


# -----------------------------------------------------------------------------
# Builders
# -----------------------------------------------------------------------------

def _build_cartesian_maf(spec: FlowSpec, ctx: Dict[str, Any]) -> torch.nn.Module:
    n_features = int(ctx["n_nonfixed_dofs"])
    conditioning_indices = ctx.get("conditioning_indices", None)

    # Guard: if everything is conditioning, nothing left to map.
    if conditioning_indices is not None:
        n_cond = int(conditioning_indices.numel())
        if n_cond >= n_features:
            raise RuntimeError(
                f"Conditioning DOFs cover all features (n_features={n_features}, n_cond={n_cond})."
            )

    degrees_in = generate_degrees(
        n_features=n_features,
        conditioning_indices=conditioning_indices,
        order=spec.kwargs.get("order", "ascending"),
    )

    n_layers = int(spec.kwargs.get("n_layers", 6))
    hidden_layers = int(spec.kwargs.get("hidden_layers", 2))
    weight_norm = bool(spec.kwargs.get("weight_norm", True))
    initialize_identity = bool(spec.kwargs.get("initialize_identity", True))

    maf_layers = []
    for _ in range(n_layers):
        degrees_in = degrees_in.flip(dims=(0,))
        maf_layers.append(
            MAF(
                degrees_in=degrees_in,
                hidden_layers=hidden_layers,
                weight_norm=weight_norm,
                initialize_identity=initialize_identity,
            )
        )

    return SequentialFlow(*maf_layers)


def _build_triatomic_zmat(spec: FlowSpec, ctx: Dict[str, Any]) -> torch.nn.Module:
    # Indices within the triatomic (0..2). Resolve from kwargs, else from ctx.
    origin_idx = int(spec.kwargs.get("origin_idx", ctx.get("origin_idx", 1)))
    axis_idx = int(spec.kwargs.get("axis_idx", ctx.get("axis_idx", 0)))
    plane_idx = int(spec.kwargs.get("plane_idx", ctx.get("plane_idx", 2)))

    z_flow = VectorCouplingFlow(
        dim=3,
        n_layers=int(spec.kwargs.get("zmat_n_layers", 6)),
        hidden_dim=int(spec.kwargs.get("zmat_hidden_dim", 64)),
        n_hidden=int(spec.kwargs.get("zmat_n_hidden", 2)),
        scale=float(spec.kwargs.get("zmat_scale", 0.8)),
    )

    return TriatomicZMatrixFlow(
        z_flow=z_flow,
        origin_idx=origin_idx,
        axis_idx=axis_idx,
        plane_idx=plane_idx,
        eps=float(spec.kwargs.get("eps", 1e-7)),
    )


def _wrap_centered_centroid(spec: FlowSpec, ctx: Dict[str, Any]) -> torch.nn.Module:
    flow = ctx["flow"]
    subset_point_indices = spec.kwargs.get("subset_point_indices", None)
    fixed_point_idx = int(spec.kwargs.get("fixed_point_idx", 0))
    space_dimension = int(spec.kwargs.get("space_dimension", 3))
    translate_back = bool(spec.kwargs.get("translate_back", True))
    return CenteredCentroidFlow(
        flow,
        subset_point_indices=subset_point_indices,
        fixed_point_idx=fixed_point_idx,
        space_dimension=space_dimension,
        translate_back=translate_back,
    )


def _wrap_oriented(spec: FlowSpec, ctx: Dict[str, Any]) -> torch.nn.Module:
    flow = ctx["flow"]
    axis_point_idx = spec.kwargs.get("axis_point_idx", None)
    plane_point_idx = spec.kwargs.get("plane_point_idx", None)
    axis = spec.kwargs.get("axis", "z")
    plane = spec.kwargs.get("plane", "xz")
    rotate_back = bool(spec.kwargs.get("rotate_back", True))

    return OrientedFlow(
        flow,
        axis_point_idx=axis_point_idx,
        plane_point_idx=plane_point_idx,
        axis=axis,
        plane=plane,
        rotate_back=rotate_back,
    )


# Register default builders.
FlowFactory.register("cartesian_maf", _build_cartesian_maf)
FlowFactory.register("triatomic_zmat", _build_triatomic_zmat)
FlowFactory.register("centered_centroid", _wrap_centered_centroid)
FlowFactory.register("oriented", _wrap_oriented)
