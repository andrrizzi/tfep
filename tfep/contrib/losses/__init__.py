"""Deprecated compatibility aliases for historical contrib losses imports."""

from __future__ import annotations

import warnings

from tfep.regularizers.bar import BARLikeRegularizer, fep_forward_df, fep_reverse_df

warnings.warn(
    "tfep.contrib.losses is deprecated; use tfep.regularizers instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["BARLikeRegularizer", "fep_forward_df", "fep_reverse_df"]
