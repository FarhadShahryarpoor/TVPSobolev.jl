# Reproducibility

Every table and figure in the paper is built by a script under
`scripts/` acting on outputs under `results/`. Scripts seed their RNGs
from `mc.seed` in `scripts/config.yaml`, so re-running with the same
config is bit-for-bit identical.

## Environments

Two committed Julia environments:

- `Project.toml` (top-level) is the `TVPSobolev` library. Stdlib-only,
  no external deps. Test-time deps (`Aqua`, `Test`, `TestItemRunner`)
  are constrained by `[compat]` bounds, so `Pkg.test()` resolves a fresh
  env within the tested major versions.
- `scripts/Project.toml` + `scripts/Manifest.toml` for the drivers.
  Adds `Plots`, `LaTeXStrings`, `DelimitedFiles`, `Dates`. Every
  transitive dep is pinned by the committed Manifest.

Invoke scripts with `--project=scripts`, not `--project=.`.

## Pipeline

1. **Monte Carlo.**

   ```bash
   julia --project=scripts --threads=4 scripts/run_mc.jl
   ```

   Writes one row per replication to
   `results/mc_sobolev_m1/<design>_T<T>.tsv` and a per-``\tau``
   per-rep CSV alongside it. The same files go under `results/mc_ncs/`
   for the natural cubic spline. Filter with `--design lzh
   --method ncs --T 200` to run a single cell.

2. **Monte Carlo figures.**

   ```bash
   julia --project=scripts --threads=4 scripts/make_figures.jl
   ```

   Writes the paper's per-``\tau`` MSE and median-replication plots
   to `figures/mc/` (natural linear spline) and `figures/mc_ncs/`
   (natural cubic spline). Pass `--diagnostics` for MAD, coverage,
   variance-ratio, and STR median-replication panels (useful for
   checking, not cited in the paper).

3. **NKPC application.**

   ```bash
   julia --project=scripts --threads=4 scripts/run_nkpc.jl
   ```

   Writes coefficient-path estimates, metadata, and figures to
   `results/nkpc_core/` (natural linear spline) and
   `results/nkpc_core_ncs/` (natural cubic spline). The paper reads
   NKPC figures directly from these folders, so this step must precede
   `latexmk`.

## Deterministic seeding

`run_mc.jl` calls `Random.seed!(base_seed)` once per `(design, T, method)`
cell, and every replication draws from the shared stream (DGP and
bootstrap multipliers in one sequence). Two runs with the same
`config.yaml` give bit-identical outputs.

The MC driver uses iid Gaussian bootstrap multipliers
(`block_length = 1`). The NKPC driver adds two serial-correlation
corrections: Newey–West HAC with 12 lags in [`pointwise_se`](@ref)
(via `hac_lags = 12`) and block-Gaussian bootstrap multipliers with
`block_length = ⌊T^{1/3}⌋` in [`bootstrap_scb`](@ref). Both are keyword
arguments.

## Data

`data/data_NKPC_coreCPI.csv` is from FRED: core CPI (`CPILFESL`) and
civilian unemployment (`UNRATE`), 1958 to 2025, monthly. Pre-built
differences and lags are included as additional columns.
