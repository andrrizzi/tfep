"""Compatibility re-exports for the former contrib flow factory.

The flow factory and related specs are now part of the core library under
``tfep.nn.flows``.
"""

from tfep.nn.flows.factory import FlowFactory, FlowSpec

__all__ = ["FlowFactory", "FlowSpec"]
