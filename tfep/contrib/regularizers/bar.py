"""Compatibility re-exports for the former contrib BAR-like regularizer."""

from tfep.regularizers.bar import BARLikeRegularizer, fep_forward_df, fep_reverse_df

__all__ = ["BARLikeRegularizer", "fep_forward_df", "fep_reverse_df"]
