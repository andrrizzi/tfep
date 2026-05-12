"""Compatibility layer for the historical ozone contrib TMBAR map.

The canonical implementation is now :class:`tfep.contrib.ozone.map.OzoneBidirectionalTMBARMap`.
This module keeps the previous import path and constructor shape working.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Sequence, Union

import pint

from tfep.nn.flows.factory import FlowSpec
from tfep.contrib.ozone.map import OzoneBidirectionalTMBARMap


class OzoneTMBARMap(OzoneBidirectionalTMBARMap):
    """Backward-compatible wrapper around ``OzoneBidirectionalTMBARMap``."""

    def __init__(
        self,
        potential_0,
        potential_1,
        topology_file_path: str,
        coordinates_file_path: Union[str, Sequence[str]],
        coordinates_file_path_2: Union[str, Sequence[str]],
        temperature: pint.Quantity,
        batch_size: int = 1,
        flow_spec: Optional[Dict[str, Any]] = None,
        flow_factory=None,
        bar_regularizer=None,
        mapped_atoms=None,
        conditioning_atoms=None,
        origin_atom=None,
        axes_atoms=None,
        tfep_logger_dir_path: str = "tfep_logs",
        dataloader_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        warnings.warn(
            "tfep.contrib.ozone.tmbar_map.OzoneTMBARMap is deprecated; "
            "use tfep.contrib.ozone.map.OzoneBidirectionalTMBARMap instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        if flow_factory is not None:
            warnings.warn(
                "flow_factory argument is ignored by the compatibility wrapper; "
                "flow construction is handled by tfep.nn.flows.FlowFactory.",
                DeprecationWarning,
                stacklevel=2,
            )

        parsed_flow_spec = _parse_flow_spec(flow_spec)

        # Prefer explicit object if provided; fallback to scalar weight.
        if bar_regularizer is not None:
            bar_reg_weight = 0.0
        else:
            bar_reg_weight = float(kwargs.pop("bar_reg_weight", 0.0))

        super().__init__(
            potential_0=potential_0,
            potential_1=potential_1,
            topology_file_path=topology_file_path,
            coordinates_file_path=coordinates_file_path,
            coordinates_file_path_2=coordinates_file_path_2,
            temperature=temperature,
            batch_size=batch_size,
            mapped_atoms=mapped_atoms,
            conditioning_atoms=conditioning_atoms,
            origin_atom=origin_atom,
            axes_atoms=axes_atoms,
            tfep_logger_dir_path=tfep_logger_dir_path,
            dataloader_kwargs=dataloader_kwargs,
            flow_spec=parsed_flow_spec,
            bar_reg_weight=bar_reg_weight,
            **kwargs,
        )

        if bar_regularizer is not None:
            self._bar_regularizer = bar_regularizer


def _parse_flow_spec(flow_spec) -> Optional[FlowSpec]:
    """Accept historical dict-like flow specs and normalize to ``FlowSpec``."""
    if flow_spec is None:
        return None

    if isinstance(flow_spec, FlowSpec):
        return flow_spec

    if isinstance(flow_spec, dict):
        name = str(flow_spec.get("type", flow_spec.get("name", "triatomic_zmat"))).strip()
        kwargs = dict(flow_spec)
        kwargs.pop("type", None)
        kwargs.pop("name", None)
        kwargs.pop("wrappers", None)
        wrappers = flow_spec.get("wrappers", [])

        parsed_wrappers = []
        for wrapper in wrappers:
            if isinstance(wrapper, FlowSpec):
                parsed_wrappers.append(wrapper)
            elif isinstance(wrapper, dict):
                w_name = str(wrapper.get("type", wrapper.get("name", ""))).strip()
                w_kwargs = dict(wrapper)
                w_kwargs.pop("type", None)
                w_kwargs.pop("name", None)
                parsed_wrappers.append(FlowSpec(name=w_name, kwargs=w_kwargs, wrappers=[]))

        return FlowSpec(name=name, kwargs=kwargs, wrappers=parsed_wrappers)

    raise TypeError("flow_spec must be None, FlowSpec, or dict")
