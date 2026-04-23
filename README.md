# TVPSobolev.jl

[![Build Status](https://github.com/FarhadShahryarpoor/TVPSobolev.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/FarhadShahryarpoor/TVPSobolev.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/FarhadShahryarpoor/TVPSobolev.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/FarhadShahryarpoor/TVPSobolev.jl)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://FarhadShahryarpoor.github.io/TVPSobolev.jl/dev/)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

Penalised SMD estimators for time-varying coefficient regression under
endogeneity. Two estimators (a natural linear spline and a natural cubic
spline) share the same instrument kernel, the Gao–Tsay bias correction,
and the multiplier-bootstrap simultaneous confidence band. Written for
ECON 622 at UBC; the paper is `project_report_622.pdf`.

## Contents

```
src/         — TVPSobolev.jl library (stdlib-only): instrument kernel,
               two estimators, bias correction, inference, DGPs
test/        — per-function tests via TestItemRunner
scripts/     — drivers: run_mc.jl, run_nkpc.jl, make_figures.jl
docs/        — Documenter.jl site (Home, Methodology,
               Reproducibility, API Reference)
data/        — NKPC core-CPI CSV input
results/     — MC and NKPC output (gitignored, regenerated from scripts)
figures/     — MC plots from make_figures.jl, plus two static
               illustrations (theta_paths.pdf, design2_panels.png).
               NKPC plots live at results/nkpc_core*/figures/.
project_report_622.tex/pdf — the paper
```

## Usage

```julia
using TVPSobolev, Random

data  = generate_lzh(500; rng = MersenneTwister(0))
K_F   = build_fsmd_kernel(data.Z)
K_ncs = ncs_penalty(500)
cv    = select_lambda_loocv(data.y, data.X, exp.(range(-12, 2, length=60));
                             method = :ncs, K_F = K_F, K_ncs = K_ncs)
θ̂     = ncs_estimate(data.y, data.X, K_F, K_ncs, cv.λ)
θ̂_c   = bias_correct(θ̂, data.X, cv.λ;
                      method = :ncs, K_F = K_F, K_ncs = K_ncs)
pw    = pointwise_se(data.y, data.X, θ̂_c, cv.λ;
                      method = :ncs, K_F = K_F, K_ncs = K_ncs)
scb   = bootstrap_scb(data.y, data.X, θ̂_c, pw.σ_V, pw.W, pw.u_c;
                       n_boot = 1000, rng = MersenneTwister(7))
```

## Reproducing the paper

The library (`src/`) is stdlib-only, so it has no external deps to
rot. The drivers pull in `Plots`, `LaTeXStrings`, `DelimitedFiles`,
and `Dates`; those live in a separate pinned environment
(`scripts/Project.toml` + `scripts/Manifest.toml`) that fixes every
transitive dep at the versions the paper was built with.

```bash
julia --project=scripts -e 'using Pkg; Pkg.instantiate()'  # once
julia --project=scripts --threads=4 scripts/run_mc.jl       # ~2 h full grid
julia --project=scripts --threads=4 scripts/run_nkpc.jl     # ~2 min
julia --project=scripts --threads=4 scripts/make_figures.jl # ~30 s
latexmk -pdf project_report_622.tex
```

Order matters: `run_nkpc.jl` must run before `latexmk` (the paper
reads NKPC plots from `results/nkpc_core*/figures/`), and
`make_figures.jl` must run before it too (MC plots in `figures/`).
Pass `--diagnostics` to `make_figures.jl` to also get the MAD /
coverage / variance-ratio / STR median-rep panels (not in the paper).

All script parameters live in `scripts/config.yaml`. Seeds are fixed;
outputs are bit-for-bit reproducible across runs.

## Testing

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Test deps (`Aqua`, `Test`, `TestItemRunner`) are pinned by compat bounds
in `Project.toml`, not by a committed Manifest, so the test env resolves
fresh each run while staying within tested major versions.

**753 assertions across 43 `@testitem` blocks**, in four layers:

*Per-function coverage* — a dedicated testitem for every exported function
(17 exports, all tested individually), with correctness checks beyond shape:
smoother–estimator consistency (`Sθ · y ≈ sobolev_m1_estimate`,
`S · y ≈ ncs_estimate` reshaped), null-space recovery (`sobolev_m1_estimate`
recovers constants, `ncs_estimate` recovers linear paths) on noiseless data,
Gao–Tsay bias-correction identity for both estimator branches, LOOCV argmin
consistency, the pointwise SE formula term-by-term, and SCB critical-value
monotonicity in nominal level plus the `θ̂_c ± c_α σ_V` band identity.

*Paper-load-bearing identities* — a numerical verification of Lemma 2 of the
accompanying paper (the twicing identity `W·D - I = -(I - S·D)²` that drops
the smoothing bias from `O(λ)` to `O(λ²)`), superposition (linearity of both
estimators in `y`), null-space limits `λ → ∞` for both penalties, hat-matrix
effective-degrees-of-freedom bounds and monotonicity in `λ`, instrument
kernel scale invariance under `Z → c·Z`, kernel-ridge monotonicity on the
minimum
eigenvalue, and a full-pipeline integration run.

*Statistical behaviour (small Monte Carlo)* — NCS IMSE decreases with
`T` at the theoretical `T^{-4/5}` pace, the Gao–Tsay correction reduces
raw MC IMSE on curved truth, m=1 and NCS agree on linear truth, and a
50-replication pointwise-coverage MC on null-space truth tracks nominal
within MC tolerance. A full-pipeline run is also bit-for-bit
reproducible across identical-seed re-runs.

*Quality assurance* — [Aqua.jl](https://github.com/JuliaTesting/Aqua.jl)
checks (method ambiguities, unbound type parameters, stale dependencies,
undefined exports), the docstring examples that ship `jl-doctest` blocks
are run in CI, and a DGP structural-sanity test (Design 2's
regime-transition function crosses 0.5 at τ ∈ {1/4, 3/4}, matching the
paper's weak-ID construction).

GitHub Actions runs the suite on Julia 1.10 and the current release on
Linux and macOS, uploads coverage to Codecov, and rebuilds the docs.

Code style: [Blue](https://github.com/invenia/BlueStyle), enforced via
`JuliaFormatter.jl` and the `.JuliaFormatter.toml` config at the project root.

## License

MIT, see `LICENSE`.
