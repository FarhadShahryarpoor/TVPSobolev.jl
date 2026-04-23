# Monte Carlo driver. All knobs live in scripts/config.yaml

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using TVPSobolev
using Statistics, Random

function load_config(path::AbstractString)
  lines = readlines(path)
  cfg = Dict{String,Any}()
  stack = Pair{Int,Dict{String,Any}}[0 => cfg]
  for raw in lines
    line = rstrip(raw)
    (isempty(line) || startswith(lstrip(line), "#")) && continue
    indent = length(line) - length(lstrip(line))
    while last(stack).first >= indent && length(stack) > 1
      pop!(stack)
    end
    parent = last(stack).second
    stripped = lstrip(line)
    if ':' in stripped
      key, rest = split(stripped, ':', limit=2)
      key = strip(key); rest = strip(rest)
      if isempty(rest)
        d = Dict{String,Any}()
        parent[key] = d
        push!(stack, indent => d)
      else
        parent[key] = _parse_yaml_value(rest)
      end
    end
  end
  return cfg
end

function _parse_yaml_value(s::AbstractString)
  s = strip(s)
  if startswith(s, "[") && endswith(s, "]")
    items = split(strip(s[2:end-1]), ',')
    return [_parse_yaml_value(strip(i)) for i in items if !isempty(strip(i))]
  end
  v = tryparse(Int, s);   v !== nothing && return v
  v = tryparse(Float64, s); v !== nothing && return v
  return strip(s, ['"', '\''])
end

filter_design = nothing
filter_method = nothing
filter_T = nothing
let i = 1
  while i ≤ length(ARGS)
    a = ARGS[i]
    if a == "--design"; global filter_design = ARGS[i + 1]; i += 2
    elseif a == "--method"; global filter_method = ARGS[i + 1]; i += 2
    elseif a == "--T"; global filter_T = parse(Int, ARGS[i + 1]); i += 2
    else; i += 1
    end
  end
end

"""
    run_mc(config; design, T, method)

One MC cell: `n_rep` reps of `design` at sample size `T` with `method`.
Writes a per-rep TSV and a per-(τ, rep) CSV under `results/mc_<method>/`.
"""
function run_mc(cfg::Dict; design::AbstractString, T::Int, method::Symbol)
  n_rep = cfg["mc"]["n_rep"]
  n_boot = cfg["mc"]["n_boot"]
  trim = cfg["mc"]["trim"]
  base_seed = cfg["mc"]["seed"]
  lcv_cfg = cfg["loocv"][String(method)]
  λ_grid = exp.(range(lcv_cfg["log_min"], lcv_cfg["log_max"];
                       length=lcv_cfg["n_grid"]))

  outdir = joinpath(cfg["paths"]["results_dir"], "mc_$(method)")
  mkpath(outdir)
  tsv_path = joinpath(outdir, "$(design)_T$(T).tsv")
  csv_path = joinpath(outdir, "$(design)_T$(T)_pertau.csv")

  τ = collect(range(1 / T, 1.0, length=T))
  if method === :sobolev_m1
    sm_ref = sobolev_m1_grams(T)
    G, G1 = sm_ref.G, sm_ref.G1
    K_ncs = nothing
  else
    G, G1 = nothing, nothing
    K_ncs = ncs_penalty(T)
  end
  θ0 = true_theta(τ)

  open(tsv_path, "w") do io_tsv
    println(io_tsv, join(["rep", "T", "lambda",
      "imse_raw_1", "imse_raw_2", "imse_bc_1", "imse_bc_2",
      "pw_025_1", "pw_025_2", "pw_050_1", "pw_050_2", "pw_075_1", "pw_075_2",
      "scb_1", "scb_2", "scb_w_1", "scb_w_2",
      "se_mid_1", "se_mid_2", "theta_mid_1", "theta_mid_2"], "\t"))
    open(csv_path, "w") do io_csv
      println(io_csv, "design,T,rep,tau_idx,tau,theta0_1,theta0_2,",
                      "theta_raw_1,theta_raw_2,theta_bc_1,theta_bc_2,",
                      "sigma_1,sigma_2,",
                      "pw_lo_1,pw_lo_2,pw_hi_1,pw_hi_2,",
                      "scb_lo_1,scb_lo_2,scb_hi_1,scb_hi_2")
      Random.seed!(base_seed)
      t0 = time()
      for rep in 1:n_rep
        data = design == "lzh" ? generate_lzh(T) : generate_str(T)
        K_F = build_fsmd_kernel(data.Z)
        out_cv = select_lambda_loocv(data.y, data.X, λ_grid;
                                      method=method, K_F=K_F,
                                      G=G, G1=G1, K_ncs=K_ncs)
        λ = out_cv.λ
        θ_raw = method === :sobolev_m1 ?
                sobolev_m1_estimate(data.y, data.X, K_F, G, G1, λ) :
                ncs_estimate(data.y, data.X, K_F, K_ncs, λ)
        θ_bc = bias_correct(θ_raw, data.X, λ;
                            method=method, K_F=K_F, G=G, G1=G1, K_ncs=K_ncs)
        pw = pointwise_se(data.y, data.X, θ_bc, λ;
                           method=method, K_F=K_F, G=G, G1=G1, K_ncs=K_ncs)
        scb = bootstrap_scb(data.y, data.X, θ_bc, pw.σ_V, pw.W, pw.u_c;
                             n_boot=n_boot, block_length=1,
                             trim=trim, nominal=0.95,
                             rng=Random.default_rng())

        imse_raw = vec(mean((θ_raw .- θ0) .^ 2; dims=1))
        imse_bc  = vec(mean((θ_bc  .- θ0) .^ 2; dims=1))
        τ_eval = [0.25, 0.5, 0.75]
        pw_cov = zeros(Bool, length(τ_eval), 2)
        for (ei, τ_e) in enumerate(τ_eval)
          i0 = argmin(abs.(τ .- τ_e))
          for j in 1:2
            pw_cov[ei, j] = abs(θ_bc[i0, j] - θ0[i0, j]) ≤ 1.96 * pw.σ_V[i0, j]
          end
        end
        scb_cov = zeros(Bool, 2); scb_w = zeros(2)
        int_ix = interior_indices(τ; trim=trim)
        for j in 1:2
          ok = true; wsum = 0.0
          for i in int_ix
            hw = scb.c_α[j] * pw.σ_V[i, j]
            abs(θ_bc[i, j] - θ0[i, j]) > hw && (ok = false)
            wsum += 2 * hw
          end
          scb_cov[j] = ok
          scb_w[j]   = wsum / length(int_ix)
        end
        τ_mid = argmin(abs.(τ .- 0.5))
        row = Any[rep, T, λ,
                  imse_raw[1], imse_raw[2], imse_bc[1], imse_bc[2],
                  pw_cov[1, 1], pw_cov[1, 2], pw_cov[2, 1], pw_cov[2, 2],
                  pw_cov[3, 1], pw_cov[3, 2],
                  scb_cov[1], scb_cov[2], scb_w[1], scb_w[2],
                  pw.σ_V[τ_mid, 1], pw.σ_V[τ_mid, 2],
                  θ_bc[τ_mid, 1], θ_bc[τ_mid, 2]]
        println(io_tsv, join((_gfmt(v) for v in row), "\t"))
        flush(io_tsv)
        d8 = x -> round(x; digits=8)
        for i in 1:T
          fields = [design, T, rep, i,
                    round(τ[i]; digits=6),
                    d8(θ0[i, 1]), d8(θ0[i, 2]),
                    d8(θ_raw[i, 1]), d8(θ_raw[i, 2]),
                    d8(θ_bc[i, 1]), d8(θ_bc[i, 2]),
                    d8(pw.σ_V[i, 1]), d8(pw.σ_V[i, 2]),
                    d8(θ_bc[i, 1] - 1.96 * pw.σ_V[i, 1]),
                    d8(θ_bc[i, 2] - 1.96 * pw.σ_V[i, 2]),
                    d8(θ_bc[i, 1] + 1.96 * pw.σ_V[i, 1]),
                    d8(θ_bc[i, 2] + 1.96 * pw.σ_V[i, 2]),
                    d8(scb.scb_lo[i, 1]), d8(scb.scb_lo[i, 2]),
                    d8(scb.scb_hi[i, 1]), d8(scb.scb_hi[i, 2])]
          println(io_csv, join(fields, ','))
        end
        flush(io_csv)
        if rep % 25 == 0
          elapsed = round(Int, time() - t0)
          println("  $design T=$T $method: rep $rep/$n_rep ($(elapsed)s)")
          flush(stdout)
        end
      end
      elapsed = round(Int, time() - t0)
      println("  $design T=$T $method: done in $(elapsed)s")
    end
  end
end

function _gfmt(v)
  v isa Bool && return string(Int(v))
  v isa Integer && return string(v)
  return string(round(v; sigdigits=8))
end

const CFG = load_config(joinpath(@__DIR__, "config.yaml"))

for design in CFG["mc"]["designs"]
  filter_design !== nothing && design != filter_design && continue
  for T in CFG["mc"]["T_values"]
    filter_T !== nothing && T != filter_T && continue
    for m in CFG["mc"]["methods"]
      filter_method !== nothing && m != filter_method && continue
      println("\n═══ $(uppercase(design))  T=$T  $m ═══")
      run_mc(CFG; design=design, T=T, method=Symbol(m))
    end
  end
end
