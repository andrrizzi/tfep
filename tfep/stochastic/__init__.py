"""Experimental stochastic path-weighted TFEP utilities.

This package is intentionally separate from :mod:`tfep.app` so the production
Deterministic TFEP/TMBAR estimator cannot accidentally consume stochastic final
coordinates without the required path-probability terms.
"""

from .kernels import (
    GaussianRandomWalkKernel,
    KernelContext,
    StochasticKernel,
    UnadjustedLangevinKernel,
)
from .molecular import MolecularStochasticConfig, evaluate_stochastic_direction, metadata_from_config, resolve_snf_flat_indices
from .path import StochasticPathBatch
from .protocols import (
    AffineFlowBlock,
    DeterministicFlowBlock,
    IdentityFlowBlock,
    StochasticFlowBlock,
)
from .work import (
    bar_delta_f,
    compute_path_work,
    effective_sample_size_from_log_weights,
    jarzynski_delta_f_forward,
    jarzynski_delta_f_reverse,
)
from .estimator import PathWeightedTFEP
from .training import (
    StochasticTrainingConfig,
    StochasticTrainingWorks,
    compute_bidirectional_stochastic_training_works,
)

__all__ = [
    "AffineFlowBlock",
    "DeterministicFlowBlock",
    "GaussianRandomWalkKernel",
    "IdentityFlowBlock",
    "KernelContext",
    "MolecularStochasticConfig",
    "PathWeightedTFEP",
    "StochasticFlowBlock",
    "StochasticKernel",
    "StochasticPathBatch",
    "UnadjustedLangevinKernel",
    "bar_delta_f",
    "StochasticTrainingConfig",
    "StochasticTrainingWorks",
    "compute_bidirectional_stochastic_training_works",
    "evaluate_stochastic_direction",
    "compute_path_work",
    "effective_sample_size_from_log_weights",
    "jarzynski_delta_f_forward",
    "jarzynski_delta_f_reverse",
    "metadata_from_config",
    "resolve_snf_flat_indices",
]
