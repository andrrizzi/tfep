# Stochastic Path-Weighted TFEP Design

This document describes the experimental stochastic normalizing-flow estimator. It is intentionally separate from the production deterministic TFEP/TMBAR code path.

## Architecture

The implementation lives under:

```text
tfep/stochastic/
```

Main objects:

```text
StochasticKernel
GaussianRandomWalkKernel
UnadjustedLangevinKernel
DeterministicFlowBlock
StochasticFlowBlock
StochasticPathBatch
PathWeightedTFEP
```

`PathWeightedTFEP` is evaluation-only in the MVP and does not inherit from `TFEPMapBase` or `TMBARMapBase`. This prevents stochastic final coordinates from entering deterministic work arrays by accident.

## Kernels

`GaussianRandomWalkKernel` is the first validation kernel:

```text
x_next = x + sigma eta
eta ~ N(0, I)
```

The forward and reverse log probabilities are explicit Gaussian densities.

`UnadjustedLangevinKernel` is implemented for toy and selected unconstrained coordinate tests:

```text
x_next = x - eps D grad u(x) + sqrt(2 D eps) eta
```

It must not be used blindly for constrained or periodic molecular systems.

## Data Flow

Forward direction:

```text
source samples -> deterministic flow block -> stochastic kernel -> target proposal -> path work
```

Reverse direction:

```text
target samples -> reverse stochastic kernel -> inverse deterministic flow -> source proposal -> reverse path work
```

Each path stores:

```text
u_source_x0
u_target_xK
sum_logJ
sum_logq_forward
sum_logq_reverse
path_work
log_weight
```

## CLI

A helper function adds backwards-compatible experimental flags:

```text
--estimator deterministic-tfep
--estimator stochastic-path-tfep
```

Default remains deterministic.

## Output

Serialization helpers write:

```text
README.md
config.json
paths_forward.csv
paths_reverse.csv
work_forward.npy
work_reverse.npy
log_terms_forward.npz
log_terms_reverse.npz
diagnostics.json
```

Each README warns that stochastic path works include transition probability terms and must not be mixed into deterministic BAR arrays without explicit estimator selection.

## Molecular Limitations

The MVP does not claim full molecular physical validity. For solvated molecular systems, PBC, constraints, rigid waters, COM removal, shell-water identity, and units must be validated before ULA or any physically meaningful stochastic kernel is used scientifically.

## Molecular Integration Status

The molecular scripts now expose the stochastic estimator only through explicit flags:

```text
--estimator stochastic-path-tfep
--snf-kernel gaussian-rw
--snf-noise-sigma 0.01
--snf-apply-to selected
```

For bromomethane holdout/FreeSolv campaigns, stochastic path works are written beside the deterministic validation arrays:

```text
stochastic_tfep_outputs/metadata.json
stochastic_tfep_outputs/stochastic_work_arrays.npz
stochastic_tfep_outputs/paths_forward.csv
stochastic_tfep_outputs/paths_reverse.csv
```

The holdout summary stores the result under `held_out.stochastic_path_tfep`, and `validation_work_arrays.npz` includes `snf_w01` and `snf_w10` when enabled.

The FreeSolv orchestrator passes SNF options from the YAML config and the analysis orchestrator aggregates SNF leg and cycle metrics.

## Trainable SNF Status

Trainable stochastic-path BAR/hybrid objectives are opt-in:

```text
--snf-train
--snf-train-objective {bar,hybrid}
--snf-train-mc-samples INT
--snf-gradient-policy {stop-gradient,full}
--snf-allow-constrained-cartesian-ula
```

When enabled, the BAR/hybrid objective consumes path-weighted stochastic works:

```text
w = u_target(xK) - u_source(x0) - logJ + logq_forward - logq_reverse
```

Deterministic KL losses and deterministic validation arrays remain separate.

For molecular ULA, `stop-gradient` is the intended default. The drift is computed from OpenMM/GROMACS forces but detached before optimization, avoiding unsupported Hessian/double-backpropagation through molecular potentials. Full-gradient ULA is reserved for toy/Torch potentials.

Molecular Cartesian ULA remains scientifically experimental for constrained/PBC endpoints. It requires explicit acknowledgment with `--snf-allow-constrained-cartesian-ula`, and run metadata records that the transition density is not a rigorously derived constrained-manifold density.
