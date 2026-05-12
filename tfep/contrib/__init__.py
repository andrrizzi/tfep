"""tfep.contrib

Contributed (non-core) utilities and examples.

This namespace is meant for:
- experiment-specific flows/maps (e.g., ozone demos)
- optional regularizers and diagnostics
- developer-facing flow factories/registries

Nothing in here is required by the core tfep package.
"""

from __future__ import annotations

__all__ = [
    "flows",
    "losses",
    "ozone",
    "regularizers",
]
