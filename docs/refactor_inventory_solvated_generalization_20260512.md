# Refactor Inventory Matrix (Solvated Generalization)

This matrix maps implementation snippets from local bromomethane workflows to their target location in `tfep`.

| Source snippet family | New home in `tfep` | Status | Notes |
|---|---|---|---|
| Solute AUTO selection excluding water/ions | `tfep/solvation/mapping.py::resolve_solute_atom_indices` | Ported | Generic across solvated small molecules. |
| Shell water occupancy ranking | `tfep/solvation/mapping.py::rank_shell_waters_by_occupancy` | Ported | Reusable for wrapper-level water candidate selection. |
| State0/state1 shell ranking merge | `tfep/solvation/mapping.py::merge_shell_rankings` | Ported | Wrapper-level composition retained outside core runners. |
| First/second shell slot conditioning indices | `tfep/solvation/mapping.py::compute_shell_slot_conditioning_indices` | Ported | Supports `molecule` and `oxygen` conditioning modes. |
| Shell-equivariant index spec build | `tfep/solvation/mapping.py::build_shell_equivariant_index_spec` | Ported | Generic mapped/local indexing contract. |
| Dynamic shell permutation dataset | `tfep/io/dataset/shell.py::SolvationShellPermutingTrajectoryDataset` | Ported | Fixed-size shell conditioning with per-frame permutation. |
| Internal shell-water equivariant flow | `tfep/nn/flows/shell_water.py::ShellEquivariantWaterInternalFlow` | Ported | Optional modular flow, no bromomethane constants. |
| Joint solute MAF + internal shell-water flow | `tfep/nn/flows/shell_water.py::JointSoluteMAFAndShellWaterInternalFlow` | Ported | Optional diagnostic composition block. |
| Bidirectional objective and BAR/logJ logic | `tfep/app/base.py::TMBARMapBase` | Refactored | Extracted reusable helper methods; behavior preserved. |
| Ozone contrib duplicated dataloader/loss logic | `tfep/contrib/ozone/tmbar_map.py` | Shimmed | Compatibility wrapper now delegates to canonical map. |
| Legacy contrib losses import path | `tfep/contrib/losses/__init__.py` | Shimmed | Deprecation shim to `tfep.regularizers`. |
| Legacy contrib triatomic flow module | `tfep/contrib/flows/triatomic_zmat.py` | Shimmed | Deprecation shim to `tfep.nn.flows.triatomic_zmatrix`. |
| Generic small-molecule TMBAR training wrapper | `tfep/app/small_molecule_tmbar.py` + `tfep-small-molecule-tmbar` | Packaged | Former methane/bromomethane training logic is now a reusable OpenMM workflow module. |
| Generic TMBAR CV/holdout/plot wrappers | `tfep/analysis/tbar_cv_bootstrap.py`, `tfep/analysis/tbar_holdout.py`, `tfep/analysis/tbar_plots.py` + console scripts | Packaged | Cross-validation, single holdout, and plot/report generation install with the library. |

## Scope boundary kept intentionally external
- Bromomethane-specific launch orchestration, folder conventions, endpoint campaign scripts, and generated simulation outputs remain outside the library.
- The packaged wrappers are generic small-molecule workflows. Local project scripts may remain as compatibility shims that delegate to these modules.
