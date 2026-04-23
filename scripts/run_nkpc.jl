#!/usr/bin/env julia
# Hybrid NKPC on core CPI. Optional `--method {sobolev_m1|ncs}` filter.

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using TVPSobolev
using LinearAlgebra, Statistics, Random, DelimitedFiles, Dates, Plots, LaTeXStrings
gr()

filter_method = nothing
let i = 1
  while i ≤ length(ARGS)
    a = ARGS[i]
    if a == "--method"; global filter_method = ARGS[i + 1]; i += 2
    else; i += 1
    end
  end
end

"""
    load_nkpc_core(datapath)

Load the core-CPI NKPC dataset. Returns `(y, X, Z, dates)` with the
hybrid NKPC design: X includes a constant, Δπ_{t+1}, Δπ_{t-1}, Δu_t,
and Z is four lags each of Δπ and Δu.
"""
function load_nkpc_core(datapath::AbstractString)
  raw, _ = readdlm(datapath, ','; header=true)
  dates_all = Date.(String.(raw[:, 1]), "yyyy-mm-dd")
  d_I = Float64.(raw[:, 5])
  d_U = Float64.(raw[:, 6])
  N = length(d_I)
  t_start, t_end = 5, N - 1
  idx = t_start:t_end
  dates = dates_all[idx]
  y = d_I[idx]
  X = hcat(ones(length(idx)), d_I[idx .+ 1], d_I[idx .- 1], d_U[idx])
  Z = hcat(d_I[idx .- 1], d_I[idx .- 2], d_I[idx .- 3], d_I[idx .- 4],
           d_U[idx .- 1], d_U[idx .- 2], d_U[idx .- 3], d_U[idx .- 4])
  return (y=y, X=X, Z=Z, dates=dates)
end

"""
    ols_2sls(y, X, Z)

Constant-coefficient 2SLS: first-stage projection `X̂ = P_Z X`, then OLS of
`y` on `X̂`. Returns `(β, se)` as the benchmark the time-varying paths are
compared against.
"""
function ols_2sls(y::AbstractVector, X::AbstractMatrix, Z::AbstractMatrix)
  n, p = size(X)
  # First stage projections: X̂ = Z (Z'Z)^{-1} Z' X
  P_Z = Z * ((Z'Z) \ Matrix(Z'))
  X̂ = P_Z * X
  β = (X̂' * X) \ (X̂' * y)
  u = y .- X * β
  σ² = dot(u, u) / (n - p)
  V = σ² * inv(X̂' * X̂)
  se = sqrt.(diag(V))
  return (β=β, se=se)
end

"""
    run_nkpc(data; method, outdir, title_prefix)

Full NKPC pipeline for one estimator. Writes `theta_hat.csv`, `meta.txt`,
and per-coefficient PDFs plus a joint 2×2 panel under `outdir/figures/`.
"""
function run_nkpc(data; method::Symbol, outdir::AbstractString, title_prefix::AbstractString)
  mkpath(outdir)
  figdir = joinpath(outdir, "figures"); mkpath(figdir)
  T = length(data.y)
  τ = collect(range(1 / T, 1.0, length=T))
  # Tiny ridge relative to K_F/T entries (O(1/T)) — preserves smooth minimum distance weighting.
  K_F = build_fsmd_kernel(data.Z; ridge=1e-6)
  if method === :sobolev_m1
    sm = sobolev_m1_grams(T)
    G, G1 = sm.G, sm.G1
    K_ncs = nothing
    λ_grid = exp.(range(-15, 10, length=200))
  else
    G, G1 = nothing, nothing
    K_ncs = ncs_penalty(T)
    λ_grid = exp.(range(-18, 7, length=200))
  end

  out_cv = select_lambda_loocv(data.y, data.X, λ_grid;
                                method=method, K_F=K_F,
                                G=G, G1=G1, K_ncs=K_ncs)
  λ = out_cv.λ
  θ_raw = method === :sobolev_m1 ?
          sobolev_m1_estimate(data.y, data.X, K_F, G, G1, λ) :
          ncs_estimate(data.y, data.X, K_F, K_ncs, λ)
  θ_c = bias_correct(θ_raw, data.X, λ;
                     method=method, K_F=K_F, G=G, G1=G1, K_ncs=K_ncs)
  pw = pointwise_se(data.y, data.X, θ_c, λ;
                     method=method, K_F=K_F, G=G, G1=G1, K_ncs=K_ncs,
                     hac_lags=12)
  scb = bootstrap_scb(data.y, data.X, θ_c, pw.σ_V, pw.W, pw.u_c;
                       n_boot=1000, block_length=max(1, floor(Int, T^(1 / 3))),
                       rng=MersenneTwister(2026))

  u_c = pw.u_c
  σ̂_ε = std(u_c)
  R² = 1 - var(u_c) / var(data.y)

  int_ix = interior_indices(τ)
  open(joinpath(outdir, "theta_hat.csv"), "w") do io
    println(io, "date,tau,intercept,rho_forward,gamma_backward,alpha_du,",
                "scb_lo_1,scb_hi_1,scb_lo_2,scb_hi_2,scb_lo_3,scb_hi_3,scb_lo_4,scb_hi_4,",
                "pw_lo_1,pw_hi_1,pw_lo_2,pw_hi_2,pw_lo_3,pw_hi_3,pw_lo_4,pw_hi_4")
    d8(x) = round(x; digits=8)
    for t in 1:T
      row = [string(data.dates[t]), string(round(τ[t]; digits=6))]
      append!(row, string.(d8.(θ_c[t, :])))
      for j in 1:4
        push!(row, string(d8(scb.scb_lo[t, j])))
        push!(row, string(d8(scb.scb_hi[t, j])))
      end
      for j in 1:4
        push!(row, string(d8(θ_c[t, j] - 1.96 * pw.σ_V[t, j])))
        push!(row, string(d8(θ_c[t, j] + 1.96 * pw.σ_V[t, j])))
      end
      println(io, join(row, ','))
    end
  end
  open(joinpath(outdir, "meta.txt"), "w") do io
    println(io, "NKPC $method estimation")
    println(io, "  T      = $T")
    println(io, "  λ_CV   = $(round(λ; sigdigits=6))")
    println(io, "  σ̂_ε   = $(round(σ̂_ε; digits=4))")
    println(io, "  R²     = $(round(R²; digits=4))")
    println(io, "  Sample = $(data.dates[1]) to $(data.dates[end])")
    for (j, name) in enumerate(["c(τ)", "ρ(τ)", "γ(τ)", "α(τ)"])
      println(io, "  c_α[$name] = $(round(scb.c_α[j]; digits=3))")
    end
  end

  ylabels = [L"c(\tau)", L"\rho(\tau)", L"\gamma(\tau)", L"\alpha(\tau)"]
  names = ["intercept", "rho_forward", "gamma_backward", "alpha_du"]
  yr0 = year(data.dates[int_ix[1]]); yr1 = year(data.dates[int_ix[end]])
  tick_years = (yr0 + (4 - yr0 % 4) % 4):4:yr1
  tick_dates = [Date(y, 1, 1) for y in tick_years]
  tick_labels = string.(tick_years)
  panels = []
  for j in 1:4
    plt = plot(data.dates[int_ix], θ_c[int_ix, j]; lw=2, lc=:black,
               label="Estimate", ylabel=ylabels[j], grid=false, framestyle=:box,
               legend=:topright, legendfontsize=7, guidefontsize=12,
               tickfontsize=8, xticks=(tick_dates, tick_labels), xrotation=45)
    plot!(plt, data.dates[int_ix], scb.scb_lo[int_ix, j];
          lw=1.2, lc=:red, ls=:dot, label="95% SCB")
    plot!(plt, data.dates[int_ix], scb.scb_hi[int_ix, j];
          lw=1.2, lc=:red, ls=:dot, label="")
    plot!(plt, data.dates[int_ix],
          θ_c[int_ix, j] .- 1.96 .* pw.σ_V[int_ix, j];
          lw=0.7, lc=:black, ls=:dash, label="95% pw CI")
    plot!(plt, data.dates[int_ix],
          θ_c[int_ix, j] .+ 1.96 .* pw.σ_V[int_ix, j];
          lw=0.7, lc=:black, ls=:dash, label="")
    hline!(plt, [0.0]; lc=:gray, ls=:dot, lw=0.8, label="")
    push!(panels, plt)
    savefig(plt, joinpath(figdir, "nkpc_$(names[j]).pdf"))
  end
  big = plot(panels...; layout=(2, 2), size=(1400, 900),
             plot_title=title_prefix)
  savefig(big, joinpath(figdir, "nkpc_all_params.pdf"))
  savefig(panels[4], joinpath(figdir, "phillips_slope.pdf"))
  println("  Saved to $outdir")
  return (θ_c=θ_c, σ_V=pw.σ_V, scb_lo=scb.scb_lo, scb_hi=scb.scb_hi,
          dates=data.dates, int_ix=int_ix, figdir=figdir, names=["intercept",
          "rho_forward", "gamma_backward", "alpha_du"],
          ylabels=[L"c(\tau)", L"\rho(\tau)", L"\gamma(\tau)", L"\alpha(\tau)"])
end

"""
    plot_matched(results_m1, results_ncs, β_2sls; label)

Re-render both 2×2 coefficient panels using a common y-axis per coefficient
(computed jointly from the two estimators' bands), and overlay the 2SLS
constant-parameter benchmark as a horizontal dash-dotted line so the reader
can see whether the constant-parameter fit lies inside the simultaneous
confidence band.
"""
function plot_matched(m1, ncs, β_2sls::AbstractVector)
  int_ix = m1.int_ix
  yr0 = year(m1.dates[int_ix[1]]); yr1 = year(m1.dates[int_ix[end]])
  tick_years = (yr0 + (4 - yr0 % 4) % 4):4:yr1
  tick_dates = [Date(y, 1, 1) for y in tick_years]
  tick_labels = string.(tick_years)

  # Shared y-axis per coefficient so the two estimators plot on the same scale.
  function ylim(j)
    los = vcat(m1.scb_lo[int_ix, j], ncs.scb_lo[int_ix, j])
    his = vcat(m1.scb_hi[int_ix, j], ncs.scb_hi[int_ix, j])
    los = filter(!isnan, los); his = filter(!isnan, his)
    lo = minimum(los); hi = maximum(his)
    pad = 0.05 * (hi - lo)
    return (lo - pad, hi + pad)
  end

  for r in (m1, ncs)
    per_panel = Dict{Int,Any}()
    for j in 1:4
      plt = plot(r.dates[int_ix], r.θ_c[int_ix, j]; lw=2, lc=:black,
                 label="Estimate", ylabel=r.ylabels[j], grid=false, framestyle=:box,
                 legend=:topright, legendfontsize=7, guidefontsize=12,
                 tickfontsize=8, xticks=(tick_dates, tick_labels), xrotation=45,
                 ylims=ylim(j))
      plot!(plt, r.dates[int_ix], r.scb_lo[int_ix, j];
            lw=1.2, lc=:red, ls=:dot, label="95% SCB")
      plot!(plt, r.dates[int_ix], r.scb_hi[int_ix, j];
            lw=1.2, lc=:red, ls=:dot, label="")
      plot!(plt, r.dates[int_ix],
            r.θ_c[int_ix, j] .- 1.96 .* r.σ_V[int_ix, j];
            lw=0.7, lc=:black, ls=:dash, label="95% pw CI")
      plot!(plt, r.dates[int_ix],
            r.θ_c[int_ix, j] .+ 1.96 .* r.σ_V[int_ix, j];
            lw=0.7, lc=:black, ls=:dash, label="")
      hline!(plt, [0.0]; lc=:gray, ls=:dot, lw=0.8, label="")
      hline!(plt, [β_2sls[j]]; lc=:blue, ls=:dashdot, lw=1.5, label="2SLS (const)")
      per_panel[j] = plt
      savefig(plt, joinpath(r.figdir, "nkpc_$(r.names[j]).pdf"))
    end
    # Joint figure drops the intercept panel.
    big = plot(per_panel[2], per_panel[3], per_panel[4];
               layout=(3, 1), size=(1000, 1100))
    savefig(big, joinpath(r.figdir, "nkpc_all_params.pdf"))
    savefig(per_panel[4], joinpath(r.figdir, "phillips_slope.pdf"))
  end
end

const PROJECT = dirname(@__DIR__)
const DATA_DIR = joinpath(PROJECT, "data")
const RESULTS = joinpath(PROJECT, "results")

data = load_nkpc_core(joinpath(DATA_DIR, "data_NKPC_coreCPI.csv"))
bench = ols_2sls(data.y, data.X, data.Z)
open(joinpath(RESULTS, "nkpc_core_2sls.txt"), "w") do io
  println(io, "2SLS constant-parameter benchmark, core CPI")
  println(io, "  T = $(length(data.y)), p = $(size(data.X, 2)), q = $(size(data.Z, 2))")
  for (j, name) in enumerate(["c", "ρ (forward Δπ)", "γ (backward Δπ)", "α (Δu)"])
    β = round(bench.β[j]; digits=4)
    se = round(bench.se[j]; digits=4)
    sign = bench.β[j] ≥ 0 ? "+" : ""
    println(io, "  ", rpad(name, 20), "  $sign$β  (SE $se)")
  end
end

results = Dict{Symbol,Any}()
for m in [:sobolev_m1, :ncs]
  filter_method !== nothing && String(m) != filter_method && continue
  suffix = m === :sobolev_m1 ? "" : "_ncs"
  outdir = joinpath(RESULTS, "nkpc_core$suffix")
  println("\n═══ NKPC core CPI, $m ═══")
  results[m] = run_nkpc(data; method=m, outdir=outdir,
                         title_prefix="NKPC ($m) — Core CPI")
end

if haskey(results, :sobolev_m1) && haskey(results, :ncs)
  plot_matched(results[:sobolev_m1], results[:ncs], bench.β)
  println("  Re-rendered with matched y-axis + 2SLS overlay.")
end
