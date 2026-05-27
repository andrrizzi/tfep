# Stochastic TFEP Validation Plan

The stochastic estimator is not scientifically usable until all validation gates pass.

## Required Unit Tests

```text
Gaussian random-walk log probability equals manual Gaussian formula
ULA log probability equals manual Gaussian formula
seed reproducibility
batch shape consistency
failure handling
```

## Required Analytic Tests

```text
identity flow + no stochasticity gives ordinary FEP
deterministic affine flow limit matches TFEP formula
1D Gaussian free energy is recovered within statistical uncertainty
shifted harmonic oscillator free energy is recovered within statistical uncertainty
bidirectional BAR sign convention returns the analytic free energy
```

## Required Molecular Tests

```text
molecular smoke paths have finite u, logJ, logq, work
coordinate units are recorded
gradient units are documented
PBC use either has a valid path density or fails explicitly
constraints either have a valid constrained density or fail explicitly
```

## Acceptance Gates

```text
deterministic estimator unchanged by default
deterministic limit equals current TFEP work
forward/reverse BAR toy tests pass
path log probabilities are auditable
molecular smoke tests are finite
metadata can reproduce every path work
```

## Non-Goals For MVP

```text
no production molecular ULA claim
no MALA/HMC production implementation
no replacement of deterministic TFEP/TMBAR defaults
```

## Trainable SNF Gates

Before treating trainable molecular SNF scientifically, verify:

```text
--snf-train absent gives identical deterministic behavior
--snf-train with kernel=none matches deterministic TFEP work
ULA fails unless explicitly acknowledged for constrained Cartesian molecular endpoints
stop-gradient ULA produces finite first-order gradients without Hessians
train-time stochastic BAR logs snf_bar_obj and snf_df_bar_obj
validation still writes deterministic tfep_w01/tfep_w10 separately from snf_w01/snf_w10
```

## Current Molecular Smoke Gate

Before scientific use on FreeSolv or methane/bromomethane, run at least one small capped evaluation:

```bash
python SCRIPT.py \
  ... existing training args ... \
  --estimator stochastic-path-tfep \
  --snf-kernel gaussian-rw \
  --snf-noise-sigma 0.01 \
  --snf-max-frames 100 \
  --snf-apply-to selected
```

Check that:

```text
stochastic_tfep_outputs/metadata.json exists
stochastic_tfep_outputs/stochastic_work_arrays.npz exists
all snf_w01/snf_w10 values are finite
SNF works are reported separately from deterministic TFEP works
```

For the experimental trainable molecular ULA branch, start with:

```bash
python SCRIPT.py \
  ... existing training args ... \
  --objective bar \
  --estimator stochastic-path-tfep \
  --snf-train \
  --snf-train-objective bar \
  --snf-kernel ula \
  --snf-step-size 1.0e-6 \
  --snf-diffusion 1.0 \
  --snf-gradient-policy stop-gradient \
  --snf-apply-to selected \
  --snf-allow-constrained-cartesian-ula
```
