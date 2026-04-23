# Build the MC figures for the paper from results/mc_{sobolev_m1,ncs}/.
# Run scripts/run_mc.jl first. Pass --diagnostics to also produce the
# MAD, coverage, variance-ratio, and off-paper median-rep plots.

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using TVPSobolev
using DelimitedFiles, Statistics, Plots, LaTeXStrings
gr()

const PROJECT = dirname(@__DIR__)
const METHOD_NAME = Dict(:sobolev_m1 => "mc", :ncs => "mc_ncs")
const T_LIST = [200, 400, 800]
const P = 2
const TRIM = 0.05
const ls_map = Dict(200 => :solid, 400 => :dash, 800 => :dot)
const DIAGNOSTICS = "--diagnostics" in ARGS

"""
    load_pertau(method, design, T)

Reshape a per-τ per-rep CSV into `n_rep × T × P` arrays, or return
`nothing` if the file is missing.
"""
function load_pertau(method::Symbol, design::AbstractString, T::Integer)
  dir = joinpath(PROJECT, "results", "mc_$(method)")
  f = joinpath(dir, "$(design)_T$(T)_pertau.csv")
  isfile(f) || (return nothing)
  d, _ = readdlm(f, ','; header=true)
  n_rep = size(d, 1) ÷ T
  n_rep < 1 && return nothing
  τ = Float64.(d[1:T, 5])
  θ0 = hcat(Float64.(d[1:T, 6]), Float64.(d[1:T, 7]))
  θ_bc = zeros(n_rep, T, P); σ = zeros(n_rep, T, P)
  pw_lo = zeros(n_rep, T, P); pw_hi = zeros(n_rep, T, P)
  scb_lo = zeros(n_rep, T, P); scb_hi = zeros(n_rep, T, P)
  for r in 1:n_rep
    rows = ((r - 1) * T + 1):(r * T)
    θ_bc[r, :, 1]  = Float64.(d[rows, 10]); θ_bc[r, :, 2]  = Float64.(d[rows, 11])
    σ[r, :, 1] = Float64.(d[rows, 12]); σ[r, :, 2] = Float64.(d[rows, 13])
    pw_lo[r, :, 1] = Float64.(d[rows, 14]); pw_lo[r, :, 2] = Float64.(d[rows, 15])
    pw_hi[r, :, 1] = Float64.(d[rows, 16]); pw_hi[r, :, 2] = Float64.(d[rows, 17])
    scb_lo[r, :, 1] = Float64.(d[rows, 18]); scb_lo[r, :, 2] = Float64.(d[rows, 19])
    scb_hi[r, :, 1] = Float64.(d[rows, 20]); scb_hi[r, :, 2] = Float64.(d[rows, 21])
  end
  int = interior_indices(τ; trim=TRIM)
  return (τ=τ, θ0=θ0, int=int, θ_bc=θ_bc, σ=σ,
          pw_lo=pw_lo, pw_hi=pw_hi, scb_lo=scb_lo, scb_hi=scb_hi, n_rep=n_rep)
end

"""
    per_tau_metrics(mc)

Interior per-τ summaries across reps: MSE, MAD, PW/SCB coverage, and
(bootstrap SE)/(MC SD).
"""
function per_tau_metrics(mc)
  int = mc.int
  out = Dict{Symbol,Matrix{Float64}}()
  for k in [:mse, :mad, :pw_cov, :scb_cov, :vr]
    out[k] = zeros(length(int), P)
  end
  for (ii, i) in enumerate(int), j in 1:P
    diffs = mc.θ_bc[:, i, j] .- mc.θ0[i, j]
    out[:mse][ii, j] = mean(diffs .^ 2)
    out[:mad][ii, j] = mean(abs.(diffs))
    out[:pw_cov][ii, j]  = mean((mc.θ0[i, j] .≥ mc.pw_lo[:, i, j]) .&
                                 (mc.θ0[i, j] .≤ mc.pw_hi[:, i, j]))
    out[:scb_cov][ii, j] = mean((mc.θ0[i, j] .≥ mc.scb_lo[:, i, j]) .&
                                 (mc.θ0[i, j] .≤ mc.scb_hi[:, i, j]))
    out[:vr][ii, j] = mean(mc.σ[:, i, j]) / std(mc.θ_bc[:, i, j])
  end
  return out
end

"""
    plot_per_tau(method, design, data_dict, key, ylabel_str; refline=nothing)

Per-τ metric `key` as a 1×2 figure, one line per `T`. Saves PDF + PNG.
"""
function plot_per_tau(method::Symbol, design::AbstractString,
                      data_dict::Dict, key::Symbol, ylabel_str::AbstractString;
                      refline::Union{Nothing,Real}=nothing,
                      ylims_per_panel::Union{Nothing,Vector{Tuple{Float64,Float64}}}=nothing)
  panels = []
  for j in 1:P
    lab = j == 1 ? L"\theta_1(\tau)" : L"\theta_2(\tau)"
    ylim_kw = ylims_per_panel === nothing ? () : (ylims=ylims_per_panel[j],)
    plt = plot(; xlabel=L"\tau", ylabel=ylabel_str, title=lab,
               grid=false, framestyle=:box, legend=:best, legendfontsize=7,
               guidefontsize=10, tickfontsize=8, ylim_kw...)
    for T in T_LIST
      haskey(data_dict, T) || continue
      m = per_tau_metrics(data_dict[T])
      plot!(plt, data_dict[T].τ[data_dict[T].int], m[key][:, j];
            lw=1.3, lc=:black, ls=ls_map[T], label="T=$T")
    end
    refline !== nothing && hline!(plt, [refline]; lc=:gray, ls=:dot, lw=0.8, label="")
    push!(panels, plt)
  end
  outdir = joinpath(PROJECT, "figures", METHOD_NAME[method]); mkpath(outdir)
  fig = plot(panels...; layout=(1, 2), size=(1000, 420))
  savefig(fig, joinpath(outdir, "$(design)_per_tau_$(String(key)).pdf"))
  savefig(fig, joinpath(outdir, "$(design)_per_tau_$(String(key)).png"))
end

"""
    mse_ylims_global()

Common y-limits for every MSE panel so figures are visually comparable.
"""
function mse_ylims_global()
  hi = -Inf
  for method in [:sobolev_m1, :ncs]
    for design in ["lzh", "str"]
      for T in T_LIST
        d = load_pertau(method, design, T)
        d === nothing && continue
        m = per_tau_metrics(d)
        hi = max(hi, maximum(m[:mse]))
      end
    end
  end
  ylim = (0.0, hi * 1.05)
  return [ylim, ylim]
end

"""
    plot_median_rep(method, design, data, T)

Plot the rep closest in L² to the pointwise-median curve, overlaid with
its pointwise CI and SCB.
"""
function plot_median_rep(method::Symbol, design::AbstractString,
                          data, T::Integer)
  int = data.int
  med = zeros(length(int), P)
  for (ii, i) in enumerate(int), j in 1:P
    med[ii, j] = median(data.θ_bc[:, i, j])
  end
  dist = zeros(data.n_rep)
  for r in 1:data.n_rep
    s = 0.0
    for (ii, i) in enumerate(int), j in 1:P
      s += (data.θ_bc[r, i, j] - med[ii, j])^2
    end
    dist[r] = s
  end
  r_typ = argmin(dist)
  panels = []
  for j in 1:P
    lab = j == 1 ? L"\theta_1(\tau)" : L"\theta_2(\tau)"
    τi = data.τ[int]
    plt = plot(τi, data.θ0[int, j]; lw=2.5, lc=:black, label="True",
               xlabel=L"\tau", ylabel=lab, grid=false, framestyle=:box,
               legend=:topleft, legendfontsize=7)
    plot!(plt, τi, data.θ_bc[r_typ, int, j]; lw=1.5, lc=:blue, label="Estimate")
    plot!(plt, τi, data.pw_lo[r_typ, int, j]; lw=0.7, lc=:black, ls=:dash, label="95% CI")
    plot!(plt, τi, data.pw_hi[r_typ, int, j]; lw=0.7, lc=:black, ls=:dash, label="")
    plot!(plt, τi, data.scb_lo[r_typ, int, j]; lw=1.2, lc=:red, ls=:dot, label="95% SCB")
    plot!(plt, τi, data.scb_hi[r_typ, int, j]; lw=1.2, lc=:red, ls=:dot, label="")
    push!(panels, plt)
  end
  outdir = joinpath(PROJECT, "figures", METHOD_NAME[method]); mkpath(outdir)
  fig = plot(panels...; layout=(1, 2), size=(1000, 420))
  savefig(fig, joinpath(outdir, "$(design)_median_rep_T$(T).pdf"))
  savefig(fig, joinpath(outdir, "$(design)_median_rep_T$(T).png"))
end

const MSE_YLIMS = mse_ylims_global()

for method in [:sobolev_m1, :ncs]
  for design in ["lzh", "str"]
    data_dict = Dict{Int,Any}()
    for T in T_LIST
      d = load_pertau(method, design, T)
      d !== nothing && (data_dict[T] = d)
    end
    isempty(data_dict) && continue

    plot_per_tau(method, design, data_dict, :mse, "MSE";
                 ylims_per_panel=MSE_YLIMS)

    # LZH median-rep panels go into Figures 4 and 6 of the paper.
    if design == "lzh"
      for T in T_LIST
        haskey(data_dict, T) && plot_median_rep(method, design, data_dict[T], T)
      end
    end

    if DIAGNOSTICS
      plot_per_tau(method, design, data_dict, :mad,     "MAD")
      plot_per_tau(method, design, data_dict, :pw_cov,  "Pointwise coverage"; refline=0.95)
      plot_per_tau(method, design, data_dict, :scb_cov, "SCB containment"; refline=0.95)
      plot_per_tau(method, design, data_dict, :vr,      "Variance ratio"; refline=1.0)
      if design == "str"
        for T in T_LIST
          haskey(data_dict, T) && plot_median_rep(method, design, data_dict[T], T)
        end
      end
    end

    println("Done: $method $design")
  end
end
