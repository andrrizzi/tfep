"""CLI helpers for experimental stochastic TFEP options."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class StochasticCLIConfig:
    estimator: str = "deterministic-tfep"
    snf_config: Optional[str] = None
    snf_output_dir: str = "stochastic_tfep_outputs"
    snf_direction: str = "bidirectional"
    snf_eval_only: bool = False
    snf_train: bool = False
    snf_train_objective: Optional[str] = None
    snf_train_mc_samples: int = 1
    snf_save_paths: bool = False
    snf_save_intermediates: bool = False
    snf_strict: bool = False
    snf_max_frames: Optional[int] = None
    snf_every_n_frames: int = 1
    snf_seed: int = 123
    snf_kernel: str = "gaussian-rw"
    snf_num_blocks: int = 1
    snf_steps_per_block: int = 1
    snf_noise_sigma: Optional[float] = None
    snf_step_size: Optional[float] = None
    snf_diffusion: float = 1.0
    snf_gradient_policy: str = "stop-gradient"
    snf_allow_molecular_ula: bool = False
    snf_allow_constrained_cartesian_ula: bool = False
    snf_selected_atoms: Optional[str] = None
    snf_apply_to: str = "selected"

    @classmethod
    def from_namespace(cls, namespace: argparse.Namespace) -> "StochasticCLIConfig":
        values = {field: getattr(namespace, field) for field in cls.__dataclass_fields__ if hasattr(namespace, field)}
        return cls(**values)


def add_stochastic_tfep_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add backwards-compatible experimental stochastic TFEP flags."""
    parser.add_argument("--estimator", choices=["deterministic-tfep", "stochastic-path-tfep"], default="deterministic-tfep")
    parser.add_argument("--snf-config", default=None)
    parser.add_argument("--snf-output-dir", default="stochastic_tfep_outputs")
    parser.add_argument("--snf-direction", choices=["forward", "reverse", "bidirectional"], default="bidirectional")
    parser.add_argument("--snf-eval-only", action="store_true")
    parser.add_argument("--snf-train", action="store_true",
                        help="Use stochastic path works in BAR/hybrid training. Experimental and opt-in.")
    parser.add_argument("--snf-train-objective", choices=["bar", "hybrid"], default=None,
                        help="Override --objective when --snf-train is enabled.")
    parser.add_argument("--snf-train-mc-samples", type=int, default=1,
                        help="Monte Carlo stochastic path samples per training frame.")
    parser.add_argument("--snf-save-paths", action="store_true")
    parser.add_argument("--snf-save-intermediates", action="store_true")
    parser.add_argument("--snf-strict", action="store_true")
    parser.add_argument("--snf-max-frames", type=int, default=None)
    parser.add_argument("--snf-every-n-frames", type=int, default=1)
    parser.add_argument("--snf-seed", type=int, default=123)
    parser.add_argument("--snf-kernel", choices=["gaussian-rw", "ula"], default="gaussian-rw")
    parser.add_argument("--snf-num-blocks", type=int, default=1)
    parser.add_argument("--snf-steps-per-block", type=int, default=1)
    parser.add_argument("--snf-noise-sigma", type=float, default=None)
    parser.add_argument("--snf-step-size", type=float, default=None)
    parser.add_argument("--snf-diffusion", type=float, default=1.0)
    parser.add_argument("--snf-gradient-policy", choices=["stop-gradient", "full"], default="stop-gradient")
    parser.add_argument("--snf-allow-molecular-ula", action="store_true")
    parser.add_argument("--snf-allow-constrained-cartesian-ula", action="store_true",
                        help="Allow experimental Cartesian molecular ULA on constrained/PBC endpoints.")
    parser.add_argument("--snf-selected-atoms", default=None)
    parser.add_argument("--snf-apply-to", choices=["all", "selected", "solute", "shell"], default="selected")
    return parser
