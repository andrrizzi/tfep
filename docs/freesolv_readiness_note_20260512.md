# FreeSolv Readiness Note (Post-Refactor)

## Why this refactor enables FreeSolv subset tests
- Solute-only mapping is now cleanly supported as a first-class reusable pattern.
- Solute + shell-slot conditioning is reusable via fixed-size shell slot indices and dynamic shell permutation datasets.
- Optional shell-equivariant internal-water flow can be activated molecule-by-molecule without changing wrappers.
- Bidirectional objective options (`kl`, `bar`, `hybrid`) and Jacobian penalty controls stay centralized in `TMBARMapBase`.

## Recommended first FreeSolv subset protocol
1. Start with solute-only mapping baseline across a curated subset.
2. Add shell-slot conditioning (`k1/k2`) for compounds where solvent coupling dominates.
3. Keep shell-equivariant internal-water flow as optional diagnostic path, not default.
4. Compare cycle/overlap/sigma metrics with identical objective settings across seeds.

## Compatibility posture
- Legacy contrib import paths continue to work via deprecation shims.
- External project wrappers remain outside `tfep`, with the template example as migration guidance.
