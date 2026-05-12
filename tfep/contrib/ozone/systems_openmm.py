"""Deprecated compatibility wrapper for historical ozone OpenMM system helpers.

The implementation was moved to :mod:`tfep.contrib.ozone.systems` with lazy
OpenMM imports. This module re-exports the same symbols to keep old paths
working without forcing OpenMM at import time.
"""

from __future__ import annotations

import warnings

from tfep.contrib.ozone.systems import (
    get_platform,
    ozone_topology,
    run_md,
    starting_positions,
    system_reference_harmonic,
    system_target_morse_anharm,
)

warnings.warn(
    "tfep.contrib.ozone.systems_openmm is deprecated; "
    "use tfep.contrib.ozone.systems instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "get_platform",
    "ozone_topology",
    "run_md",
    "starting_positions",
    "system_reference_harmonic",
    "system_target_morse_anharm",
]
