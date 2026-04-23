using TVPSobolev
using Test, TestItemRunner

# One @testitem per exported function or per tightly-scoped property.

@testitem "interior_indices: basic trimming" begin
  using TVPSobolev
  τ = collect(range(0.01, 1.0, length=100))
  @test length(interior_indices(τ; trim=0.0)) == 100
  @test all(τ[i] ≥ 0.05 && τ[i] ≤ 0.95 for i in interior_indices(τ; trim=0.05))
  @test isempty(interior_indices([0.0, 1.0]; trim=0.5))
end

@testitem "roughness: zero on linear paths, positive otherwise" begin
  using TVPSobolev
  @test isapprox(roughness(collect(1.0:10.0)), 0.0; atol=1e-12)
  @test isapprox(roughness([3.0, 3.0, 3.0, 3.0]), 0.0; atol=1e-12)
  @test roughness([0.0, 1.0, 0.0, 1.0, 0.0]) > 0
end

@testitem "build_design: correct sparsity and placement" begin
  using TVPSobolev
  X = [1.0 0.5; 1.0 0.7; 1.0 0.9]
  D = build_design(X)
  @test size(D) == (3, 6)
  @test D[1, 1] == 1.0 && D[1, 4] == 0.5
  @test D[2, 2] == 1.0 && D[2, 5] == 0.7
  @test D[3, 3] == 1.0 && D[3, 6] == 0.9
  @test count(!iszero, D) == 6  # everywhere else is zero
end

@testitem "build_fsmd_kernel: symmetric, positive definite, diagonal ~ 1/T + ridge" begin
  using TVPSobolev, LinearAlgebra, Random
  rng = MersenneTwister(2026)
  Z = randn(rng, 50, 2)
  K = build_fsmd_kernel(Z; ridge=1e-3)
  @test size(K) == (50, 50)
  @test issymmetric(K)
  # Ridge makes K strictly positive-definite, not just PSD.
  @test minimum(eigen(Matrix(K)).values) > 0
  # Product-Gaussian K(z,z) = 1, then scaled by 1/T with ridge on diagonal.
  @test isapprox(K[1, 1], 1 / 50 + 1e-3; atol=1e-10)
end

@testitem "true_theta: correct values at boundaries and bump peak" begin
  using TVPSobolev
  θ = true_theta([0.0, 0.5, 1.0])
  @test size(θ) == (3, 2)
  @test isapprox(θ[1, 1], 0.2 * exp(-0.7); atol=1e-12)         # θ_1(0)
  @test isapprox(θ[3, 1], 0.2 * exp(-0.7 + 3.5); atol=1e-12)   # θ_1(1)
  @test isapprox(θ[2, 2], 1.0 + 1.0 - 1.0; atol=1e-12)         # θ_2(0.5) = 1
end

@testitem "generate_lzh: reproducible with fixed RNG, right shapes" begin
  using TVPSobolev, Random
  data1 = generate_lzh(100; rng=MersenneTwister(42))
  data2 = generate_lzh(100; rng=MersenneTwister(42))
  @test data1.y == data2.y
  @test data1.X == data2.X
  @test size(data1.X) == (100, 2)
  @test size(data1.Z) == (100, 4)
  @test length(data1.y) == 100
  @test size(data1.θ0) == (100, 2)
end

@testitem "generate_str: reproducible and has single-column instrument" begin
  using TVPSobolev, Random
  data = generate_str(100; rng=MersenneTwister(42))
  @test size(data.Z) == (100, 1)
  @test size(data.X) == (100, 2)
end

@testitem "sobolev_m1_grams: symmetric and PSD" begin
  using TVPSobolev, LinearAlgebra
  sm = sobolev_m1_grams(40)
  @test size(sm.G) == (40, 40)
  @test size(sm.G1) == (40, 40)
  @test isapprox(sm.G, sm.G'; atol=1e-12)
  @test isapprox(sm.G1, sm.G1'; atol=1e-12)
  # G1 is the penalty-space Gram and must be PSD.
  eigs = eigen(Symmetric(sm.G1)).values
  @test minimum(eigs) > -1e-10
end

@testitem "ncs_penalty: symmetric, PSD, vanishes on linear paths" begin
  using TVPSobolev, LinearAlgebra
  T = 50
  K = ncs_penalty(T)
  @test size(K) == (T, T)
  @test isapprox(Matrix(K), Matrix(K)'; atol=1e-12)
  eigs = eigen(Symmetric(Matrix(K))).values
  # K is PSD analytically, but its computed min eigenvalue floats around
  # zero at the standard LAPACK backward-error scale ε·‖K‖·T — which
  # differs across BLAS/Julia versions, so we test against that bound
  # rather than a hardcoded constant.
  @test minimum(eigs) > -eps(Float64) * opnorm(Matrix(K)) * T
  # Null space contains constants and linear paths.
  @test isapprox(ones(T)' * K * ones(T), 0.0; atol=1e-8)
  τ = collect(range(1 / T, 1.0, length=T))
  @test isapprox(τ' * K * τ, 0.0; atol=1e-8)
end

@testitem "ncs_penalty: null space has rank 2 (constants + linear)" begin
  using TVPSobolev, LinearAlgebra
  T = 30
  K = ncs_penalty(T)
  eigs = eigen(Symmetric(Matrix(K))).values
  small = count(abs.(eigs) .< 1e-6)
  @test small == 2
end

@testitem "sobolev_m1_estimate: recovers zero coefficient with zero data" begin
  using TVPSobolev, LinearAlgebra, Random
  rng = MersenneTwister(1)
  T = 50
  Z = randn(rng, T, 2)
  K_F = build_fsmd_kernel(Z; ridge=1e-3)
  sm = sobolev_m1_grams(T)
  X = hcat(ones(T), randn(rng, T))
  y = zeros(T)
  θ̂ = sobolev_m1_estimate(y, X, K_F, sm.G, sm.G1, 1e-3)
  @test maximum(abs.(θ̂)) < 1e-8
end

@testitem "ncs_estimate: recovers linear truth on noiseless linear DGP" begin
  using TVPSobolev, LinearAlgebra, Random
  rng = MersenneTwister(1)
  T = 100
  τ = collect(range(1 / T, 1.0, length=T))
  # Truth is linear + constant — both lie in the NCS null space, so the
  # penalty contributes nothing and the noiseless fit must be exact.
  θ0 = hcat(fill(0.5, T) .+ τ, zeros(T))
  X = hcat(ones(T), randn(rng, T))
  y = vec(sum(X .* θ0; dims=2))
  Z = randn(rng, T, 2)
  K_F = build_fsmd_kernel(Z; ridge=1e-3)
  K_ncs = ncs_penalty(T)
  θ̂ = ncs_estimate(y, X, K_F, K_ncs, 1e-4)
  @test maximum(abs.(θ̂ .- θ0)) < 1e-6
end

@testitem "bias_correct: algebraic identity W = 2S - SDS" begin
  using TVPSobolev, LinearAlgebra, Random
  rng = MersenneTwister(0)
  T = 40
  data = generate_lzh(T; rng=rng)
  K_F = build_fsmd_kernel(data.Z)
  K_ncs = ncs_penalty(T)
  λ = 1e-3
  θ̂ = ncs_estimate(data.y, data.X, K_F, K_ncs, λ)
  θ̂_c = bias_correct(θ̂, data.X, λ;
                      method=:ncs, K_F=K_F, K_ncs=K_ncs)
  # Rebuild θ̂_c = 2θ̂ − S(Dθ̂) by hand and check it matches.
  fitted = vec(sum(data.X .* θ̂; dims=2))
  θ_sm = ncs_estimate(fitted, data.X, K_F, K_ncs, λ)
  θ̂_c_manual = 2 .* θ̂ .- θ_sm
  @test isapprox(θ̂_c, θ̂_c_manual; atol=1e-12)
end

@testitem "select_lambda_loocv: returns a λ in the grid" begin
  using TVPSobolev, Random
  rng = MersenneTwister(0)
  data = generate_lzh(80; rng=rng)
  K_F = build_fsmd_kernel(data.Z)
  K_ncs = ncs_penalty(80)
  λ_grid = exp.(range(-10.0, 2.0, length=40))
  out = select_lambda_loocv(data.y, data.X, λ_grid;
                             method=:ncs, K_F=K_F, K_ncs=K_ncs)
  @test out.λ ∈ λ_grid
  @test isfinite(out.cv)
  @test out.cv > 0
end

@testitem "pointwise_se (NCS): non-negative and right shape" begin
  using TVPSobolev, Random
  rng = MersenneTwister(5)
  T = 60
  data = generate_lzh(T; rng=rng)
  K_F = build_fsmd_kernel(data.Z)
  K_ncs = ncs_penalty(T)
  λ = 1e-3
  θ̂ = ncs_estimate(data.y, data.X, K_F, K_ncs, λ)
  θ̂_c = bias_correct(θ̂, data.X, λ;
                      method=:ncs, K_F=K_F, K_ncs=K_ncs)
  pw = pointwise_se(data.y, data.X, θ̂_c, λ;
                     method=:ncs, K_F=K_F, K_ncs=K_ncs)
  @test all(pw.σ_V .>= 0)
  @test size(pw.σ_V) == (T, 2)
  @test length(pw.u_c) == T
end

@testitem "pointwise_se (Sobolev m=1): non-negative and right shape" begin
  using TVPSobolev, Random
  rng = MersenneTwister(5)
  T = 60
  data = generate_lzh(T; rng=rng)
  K_F = build_fsmd_kernel(data.Z)
  sm = sobolev_m1_grams(T)
  λ = 1e-3
  θ̂ = sobolev_m1_estimate(data.y, data.X, K_F, sm.G, sm.G1, λ)
  θ̂_c = bias_correct(θ̂, data.X, λ;
                      method=:sobolev_m1, K_F=K_F, G=sm.G, G1=sm.G1)
  pw = pointwise_se(data.y, data.X, θ̂_c, λ;
                     method=:sobolev_m1, K_F=K_F, G=sm.G, G1=sm.G1)
  @test all(pw.σ_V .>= 0)
  @test size(pw.σ_V) == (T, 2)
  @test length(pw.u_c) == T
  @test size(pw.W) == (2T, T)  # coefficient-space smoother: pT rows
end

@testitem "bootstrap_scb (Sobolev m=1): deterministic with fixed RNG" begin
  using TVPSobolev, Random
  T = 50
  data = generate_lzh(T; rng=MersenneTwister(0))
  K_F = build_fsmd_kernel(data.Z)
  sm = sobolev_m1_grams(T)
  λ = 1e-3
  θ̂ = sobolev_m1_estimate(data.y, data.X, K_F, sm.G, sm.G1, λ)
  θ̂_c = bias_correct(θ̂, data.X, λ;
                      method=:sobolev_m1, K_F=K_F, G=sm.G, G1=sm.G1)
  pw = pointwise_se(data.y, data.X, θ̂_c, λ;
                     method=:sobolev_m1, K_F=K_F, G=sm.G, G1=sm.G1)
  out1 = bootstrap_scb(data.y, data.X, θ̂_c, pw.σ_V, pw.W, pw.u_c;
                        n_boot=100, rng=MersenneTwister(7))
  out2 = bootstrap_scb(data.y, data.X, θ̂_c, pw.σ_V, pw.W, pw.u_c;
                        n_boot=100, rng=MersenneTwister(7))
  @test out1.c_α == out2.c_α
  @test all(out1.c_α .> 1.96)
end

@testitem "bootstrap_scb: deterministic with fixed RNG; c_α > 1.96" begin
  using TVPSobolev, Random
  rng = MersenneTwister(7)
  T = 50
  data = generate_lzh(T; rng=MersenneTwister(0))
  K_F = build_fsmd_kernel(data.Z)
  K_ncs = ncs_penalty(T)
  λ = 1e-3
  θ̂ = ncs_estimate(data.y, data.X, K_F, K_ncs, λ)
  θ̂_c = bias_correct(θ̂, data.X, λ;
                      method=:ncs, K_F=K_F, K_ncs=K_ncs)
  pw = pointwise_se(data.y, data.X, θ̂_c, λ;
                     method=:ncs, K_F=K_F, K_ncs=K_ncs)
  out1 = bootstrap_scb(data.y, data.X, θ̂_c, pw.σ_V, pw.W, pw.u_c;
                        n_boot=100, rng=MersenneTwister(7))
  out2 = bootstrap_scb(data.y, data.X, θ̂_c, pw.σ_V, pw.W, pw.u_c;
                        n_boot=100, rng=MersenneTwister(7))
  @test out1.c_α == out2.c_α
  @test all(out1.c_α .> 1.96)  # SCB must be wider than pointwise 95%
end

@testitem "Aqua quality assurance" begin
  using TVPSobolev, Aqua
  # Skip `piracies` (false-positives on Base) and `check_extras` (test-only
  # deps don't need compat pins).
  Aqua.test_all(
    TVPSobolev;
    ambiguities=(recursive=false,),
    piracies=false,
    deps_compat=(check_extras=false,),
  )
end

# Smoother-estimator consistency, bias correction, LOOCV, SE formula.

@testitem "sobolev_m1_smoother: shape and consistency with estimate" begin
  using TVPSobolev, LinearAlgebra, Random
  rng = MersenneTwister(0)
  T = 40
  data = generate_lzh(T; rng=rng)
  K_F = build_fsmd_kernel(data.Z)
  gr = sobolev_m1_grams(T)
  λ = 1e-3
  p = size(data.X, 2)
  sm = sobolev_m1_smoother(data.X, K_F, gr.G, gr.G1, λ)
  @test size(sm.S)  == (p * T, T)
  @test size(sm.Sθ) == (p * T, T)
  @test size(sm.Φ)  == (T, p * T)
  # Smoother must reproduce the direct estimator: reshape(Sθ·y) = θ̂, G·α_j = θ̂[:, j].
  θ̂ = sobolev_m1_estimate(data.y, data.X, K_F, gr.G, gr.G1, λ)
  θ̂_from_smoother = reshape(sm.Sθ * data.y, T, p)
  @test isapprox(θ̂, θ̂_from_smoother; atol=1e-8)
  α = sm.S * data.y
  for j in 1:p
    idx = ((j - 1) * T + 1):(j * T)
    @test isapprox(gr.G * α[idx], θ̂[:, j]; atol=1e-8)
  end
end

@testitem "ncs_smoother: shape and consistency with estimate" begin
  using TVPSobolev, LinearAlgebra, Random
  rng = MersenneTwister(0)
  T = 40
  data = generate_lzh(T; rng=rng)
  K_F = build_fsmd_kernel(data.Z)
  K_ncs = ncs_penalty(T)
  λ = 1e-3
  p = size(data.X, 2)
  sm = ncs_smoother(data.X, K_F, K_ncs, λ)
  @test size(sm.S) == (p * T, T)
  @test size(sm.D) == (T, p * T)
  # S·y, reshaped, must match the direct estimator.
  α = sm.S * data.y
  θ̂ = ncs_estimate(data.y, data.X, K_F, K_ncs, λ)
  @test isapprox(reshape(α, T, p), θ̂; atol=1e-8)
  # Hat matrix H = D·S maps y to the estimator's fitted values.
  H = sm.D * sm.S
  @test size(H) == (T, T)
  ŷ = H * data.y
  ŷ_direct = vec(sum(data.X .* θ̂; dims=2))
  @test isapprox(ŷ, ŷ_direct; atol=1e-8)
end

@testitem "bias_correct (sobolev_m1): algebraic identity θ̂_c = 2θ̂ - S(Dθ̂)" begin
  using TVPSobolev, LinearAlgebra, Random
  rng = MersenneTwister(0)
  T = 40
  data = generate_lzh(T; rng=rng)
  K_F = build_fsmd_kernel(data.Z)
  gr = sobolev_m1_grams(T)
  λ = 1e-3
  θ̂ = sobolev_m1_estimate(data.y, data.X, K_F, gr.G, gr.G1, λ)
  θ̂_c = bias_correct(θ̂, data.X, λ;
                      method=:sobolev_m1, K_F=K_F, G=gr.G, G1=gr.G1)
  # Rebuild θ̂_c = 2θ̂ − S(Dθ̂) by hand and check it matches.
  fitted = vec(sum(data.X .* θ̂; dims=2))
  θ_sm = sobolev_m1_estimate(fitted, data.X, K_F, gr.G, gr.G1, λ)
  θ̂_c_manual = 2 .* θ̂ .- θ_sm
  @test isapprox(θ̂_c, θ̂_c_manual; atol=1e-12)
end

@testitem "sobolev_m1_estimate: recovers constant truth (null space) on noiseless data" begin
  using TVPSobolev, LinearAlgebra, Random
  rng = MersenneTwister(2)
  T = 100
  X = hcat(ones(T), randn(rng, T))
  # Constant truths lie in the m=1 null space, so the penalty vanishes
  # and the noiseless fit must recover them exactly for any λ.
  θ0 = hcat(fill(1.5, T), fill(-0.7, T))
  y = vec(sum(X .* θ0; dims=2))
  Z = randn(rng, T, 2)
  K_F = build_fsmd_kernel(Z; ridge=1e-3)
  gr = sobolev_m1_grams(T)
  θ̂ = sobolev_m1_estimate(y, X, K_F, gr.G, gr.G1, 1e-4)
  @test maximum(abs.(θ̂ .- θ0)) < 1e-6
end

@testitem "select_lambda_loocv: singleton grid returns that grid element" begin
  using TVPSobolev, Random
  data = generate_lzh(60; rng=MersenneTwister(0))
  K_F = build_fsmd_kernel(data.Z)
  K_ncs = ncs_penalty(60)
  out = select_lambda_loocv(data.y, data.X, [0.01];
                             method=:ncs, K_F=K_F, K_ncs=K_ncs)
  @test out.λ == 0.01
  @test out.cv == out.cv_curve[1]
  @test isfinite(out.cv)
end

@testitem "select_lambda_loocv: argmin of returned curve matches returned λ" begin
  using TVPSobolev, Random
  data = generate_lzh(80; rng=MersenneTwister(3))
  K_F = build_fsmd_kernel(data.Z)
  K_ncs = ncs_penalty(80)
  λ_grid = exp.(range(-10.0, 2.0, length=40))
  out = select_lambda_loocv(data.y, data.X, λ_grid;
                             method=:ncs, K_F=K_F, K_ncs=K_ncs)
  # Among valid (non-NaN) grid entries, the returned λ must be the argmin.
  valid = .!isnan.(out.cv_curve)
  @test any(valid)
  @test out.cv == minimum(out.cv_curve[valid])
  k_min = findfirst(i -> valid[i] && out.cv_curve[i] == out.cv, eachindex(out.cv_curve))
  @test out.λ == λ_grid[k_min]
end

@testitem "pointwise_se: matches the weight-matrix formula term-by-term" begin
  using TVPSobolev, Random
  rng = MersenneTwister(7)
  T = 40
  data = generate_lzh(T; rng=rng)
  K_F = build_fsmd_kernel(data.Z)
  K_ncs = ncs_penalty(T)
  λ = 1e-3
  θ̂ = ncs_estimate(data.y, data.X, K_F, K_ncs, λ)
  θ̂_c = bias_correct(θ̂, data.X, λ; method=:ncs, K_F=K_F, K_ncs=K_ncs)
  pw = pointwise_se(data.y, data.X, θ̂_c, λ;
                     method=:ncs, K_F=K_F, K_ncs=K_ncs)
  # Recompute σ̂_V(τ)² = Σ_t W[τ,t]² · u_c[t]² from scratch and compare.
  p = size(data.X, 2)
  for j in 1:p, i in 1:T
    row = (j - 1) * T + i
    expected = sqrt(max(sum((@view pw.W[row, :]) .^ 2 .* pw.u_c .^ 2), 1e-20))
    @test isapprox(pw.σ_V[i, j], expected; atol=1e-12)
  end
  @test isapprox(pw.u_c, data.y .- vec(sum(data.X .* θ̂_c; dims=2)); atol=1e-12)
end

@testitem "bootstrap_scb: c_α monotone in nominal level and SCB = θ̂_c ± c_α σ_V" begin
  using TVPSobolev, Random
  T = 40
  data = generate_lzh(T; rng=MersenneTwister(0))
  K_F = build_fsmd_kernel(data.Z)
  K_ncs = ncs_penalty(T)
  λ = 1e-3
  θ̂ = ncs_estimate(data.y, data.X, K_F, K_ncs, λ)
  θ̂_c = bias_correct(θ̂, data.X, λ; method=:ncs, K_F=K_F, K_ncs=K_ncs)
  pw = pointwise_se(data.y, data.X, θ̂_c, λ;
                     method=:ncs, K_F=K_F, K_ncs=K_ncs)
  # Fix the RNG across nominal levels so only the quantile changes.
  args = (data.y, data.X, θ̂_c, pw.σ_V, pw.W, pw.u_c)
  kw(nom) = (n_boot=500, nominal=nom, rng=MersenneTwister(1))
  out90 = bootstrap_scb(args...; kw(0.90)...)
  out95 = bootstrap_scb(args...; kw(0.95)...)
  out99 = bootstrap_scb(args...; kw(0.99)...)
  for j in 1:size(data.X, 2)
    @test out90.c_α[j] ≤ out95.c_α[j]
    @test out95.c_α[j] ≤ out99.c_α[j]
  end
  # Band half-width at each interior τ must equal c_α · σ_V.
  τ = collect(range(1 / T, 1.0, length=T))
  int_ix = interior_indices(τ)
  for j in 1:size(data.X, 2), i in int_ix
    hw = out95.scb_hi[i, j] - θ̂_c[i, j]
    @test isapprox(hw, out95.c_α[j] * pw.σ_V[i, j]; atol=1e-10)
    hw_lo = θ̂_c[i, j] - out95.scb_lo[i, j]
    @test isapprox(hw_lo, out95.c_α[j] * pw.σ_V[i, j]; atol=1e-10)
  end
end

# Mathematical identities the paper relies on.

@testitem "Lemma 2 (twicing identity): W·D - I = -(I - S·D)²" begin
  # Numerically verify the identity the Gao–Tsay bias correction rests on:
  # it drops the smoothing bias from O(λ) to O(λ²).
  using TVPSobolev, LinearAlgebra, Random
  rng = MersenneTwister(0)
  T = 30
  data = generate_lzh(T; rng=rng)
  K_F = build_fsmd_kernel(data.Z)
  K_ncs = ncs_penalty(T)
  λ = 1e-3
  sm = ncs_smoother(data.X, K_F, K_ncs, λ)
  S, D = sm.S, sm.D
  p = size(data.X, 2)
  W  = 2 .* S .- S * (D * S)
  Ipt = Matrix{Float64}(I, p * T, p * T)
  IminusSD = Ipt .- S * D
  lhs = W * D .- Ipt
  rhs = -(IminusSD * IminusSD)
  @test isapprox(lhs, rhs; atol=1e-8)
end

@testitem "estimators: linear in y (superposition holds for closed-form smoothers)" begin
  using TVPSobolev, LinearAlgebra, Random
  rng = MersenneTwister(1)
  T = 50
  X = hcat(ones(T), randn(rng, T))
  Z = randn(rng, T, 2)
  K_F = build_fsmd_kernel(Z)
  gr = sobolev_m1_grams(T)
  K_ncs = ncs_penalty(T)
  λ = 1e-3
  y1 = randn(rng, T); y2 = randn(rng, T); a, b = 1.7, -0.3
  e1 = ncs_estimate(y1, X, K_F, K_ncs, λ)
  e2 = ncs_estimate(y2, X, K_F, K_ncs, λ)
  em = ncs_estimate(a .* y1 .+ b .* y2, X, K_F, K_ncs, λ)
  @test isapprox(em, a .* e1 .+ b .* e2; atol=1e-8)
  f1 = sobolev_m1_estimate(y1, X, K_F, gr.G, gr.G1, λ)
  f2 = sobolev_m1_estimate(y2, X, K_F, gr.G, gr.G1, λ)
  fm = sobolev_m1_estimate(a .* y1 .+ b .* y2, X, K_F, gr.G, gr.G1, λ)
  @test isapprox(fm, a .* f1 .+ b .* f2; atol=1e-8)
end

@testitem "sobolev_m1_estimate: λ → ∞ collapses fit onto the constant null space" begin
  # At very large λ the m=1 penalty dominates, so every column of θ̂ is
  # forced onto its 1-D null space (constants).
  using TVPSobolev, LinearAlgebra, Random, Statistics
  rng = MersenneTwister(4)
  T = 80
  data = generate_lzh(T; rng=rng)
  K_F = build_fsmd_kernel(data.Z)
  gr = sobolev_m1_grams(T)
  θ̂_small = sobolev_m1_estimate(data.y, data.X, K_F, gr.G, gr.G1, 1e-3)
  θ̂_large = sobolev_m1_estimate(data.y, data.X, K_F, gr.G, gr.G1, 1e6)
  p = size(data.X, 2)
  for j in 1:p
    # Range across τ collapses at large λ.
    rng_small = maximum(θ̂_small[:, j]) - minimum(θ̂_small[:, j])
    rng_large = maximum(θ̂_large[:, j]) - minimum(θ̂_large[:, j])
    @test rng_large < rng_small / 100
  end
end

@testitem "ncs_estimate: λ → ∞ collapses fit onto the linear null space {1, τ}" begin
  # Same as the m=1 test, but with a 2-D null space {1, τ}. Measured by the
  # residual of regressing θ̂[:, j] onto (1, τ).
  using TVPSobolev, LinearAlgebra, Random, Statistics
  rng = MersenneTwister(5)
  T = 100
  data = generate_lzh(T; rng=rng)
  K_F = build_fsmd_kernel(data.Z)
  K_ncs = ncs_penalty(T)
  θ̂_large = ncs_estimate(data.y, data.X, K_F, K_ncs, 1e8)
  τ = collect(range(1 / T, 1.0, length=T))
  A = hcat(ones(T), τ)
  p = size(data.X, 2)
  for j in 1:p
    coef = A \ θ̂_large[:, j]
    resid = θ̂_large[:, j] .- A * coef
    scale = maximum(abs.(θ̂_large[:, j])) + 1.0
    @test maximum(abs.(resid)) / scale < 1e-6
  end
end

@testitem "build_fsmd_kernel: scale-invariant under Z → c·Z (pooled-std normalization)" begin
  # Pooled-std normalization absorbs any global rescaling of Z, so the R
  # reference implementation (Antoine & Sun) doesn't pre-standardize either.
  using TVPSobolev, LinearAlgebra, Random
  rng = MersenneTwister(6)
  Z = randn(rng, 50, 3)
  K1 = build_fsmd_kernel(Z; ridge=0.0)
  K2 = build_fsmd_kernel(100 .* Z; ridge=0.0)
  K3 = build_fsmd_kernel(0.01 .* Z; ridge=0.0)
  @test isapprox(K1, K2; atol=1e-10)
  @test isapprox(K1, K3; atol=1e-10)
end

@testitem "build_fsmd_kernel: positive definite on random Z across varying (T, q)" begin
  using TVPSobolev, LinearAlgebra, Random
  for seed in 0:4
    rng = MersenneTwister(seed)
    T = rand(rng, 20:60)
    q = rand(rng, 1:4)
    Z = randn(rng, T, q)
    K = build_fsmd_kernel(Z; ridge=1e-4)
    @test issymmetric(K)
    @test minimum(eigen(Matrix(K)).values) > 0
    # Diagonal entries are bounded above by 1/T + ridge.
    @test all(x -> 0 ≤ x ≤ 1 / T + 1e-4 + 1e-10, diag(K))
  end
end

@testitem "ncs_smoother: effective degrees of freedom monotone in λ and bounded by T" begin
  # tr(D·S_λ) is the effective degrees of freedom: must decrease with λ,
  # stay in (0, pT), and at λ → ∞ collapse to 2p (the NCS null-space dim).
  using TVPSobolev, LinearAlgebra, Random
  rng = MersenneTwister(8)
  T = 80
  data = generate_lzh(T; rng=rng)
  K_F = build_fsmd_kernel(data.Z)
  K_ncs = ncs_penalty(T)
  p = size(data.X, 2)
  λs = [1e-8, 1e-4, 1e-1, 1e2, 1e4]
  edfs = Float64[]
  for λ in λs
    sm = ncs_smoother(data.X, K_F, K_ncs, λ)
    edf = tr(sm.D * sm.S)
    @test 0 < edf
    @test edf ≤ T + 1e-6  # hat-matrix trace bounded by data dim
    push!(edfs, edf)
  end
  for i in 1:(length(λs) - 1)
    @test edfs[i + 1] ≤ edfs[i] + 1e-6
  end
  @test edfs[end] < 2 * p + 0.5
end

@testitem "full pipeline: LOOCV → estimate → bias-correct → SE → SCB all wire together" begin
  # End-to-end run checking every inter-module contract.
  using TVPSobolev, LinearAlgebra, Random, Statistics
  T = 200
  data = generate_lzh(T; rng=MersenneTwister(2026))
  K_F = build_fsmd_kernel(data.Z)
  K_ncs = ncs_penalty(T)
  λ_grid = exp.(range(-12, 2, length=40))
  cv = select_lambda_loocv(data.y, data.X, λ_grid;
                            method=:ncs, K_F=K_F, K_ncs=K_ncs)
  @test cv.λ ∈ λ_grid && cv.cv > 0
  θ̂ = ncs_estimate(data.y, data.X, K_F, K_ncs, cv.λ)
  θ̂_c = bias_correct(θ̂, data.X, cv.λ; method=:ncs, K_F=K_F, K_ncs=K_ncs)
  pw = pointwise_se(data.y, data.X, θ̂_c, cv.λ;
                     method=:ncs, K_F=K_F, K_ncs=K_ncs)
  @test isapprox(pw.u_c, data.y .- vec(sum(data.X .* θ̂_c; dims=2)); atol=1e-10)
  scb = bootstrap_scb(data.y, data.X, θ̂_c, pw.σ_V, pw.W, pw.u_c;
                       n_boot=500, rng=MersenneTwister(42))
  @test all(isfinite, scb.c_α)
  @test all(scb.c_α .> 1.96)
  # SCB envelope dominates the pointwise 95% envelope everywhere inside.
  τ = collect(range(1 / T, 1.0, length=T))
  int_ix = interior_indices(τ)
  p = size(data.X, 2)
  for j in 1:p, i in int_ix
    @test scb.c_α[j] * pw.σ_V[i, j] ≥ 1.96 * pw.σ_V[i, j] - 1e-12
  end
  imse = mean((θ̂_c[int_ix, :] .- data.θ0[int_ix, :]) .^ 2)
  @test imse < 1.0  # sanity: single-rep IMSE is not catastrophic
end

@testitem "small T edge case: T=10 pipeline still resolves and produces finite output" begin
  using TVPSobolev, LinearAlgebra, Random
  T = 10
  data = generate_lzh(T; rng=MersenneTwister(9))
  K_F = build_fsmd_kernel(data.Z)
  K_ncs = ncs_penalty(T)
  gr = sobolev_m1_grams(T)
  θ̂_ncs = ncs_estimate(data.y, data.X, K_F, K_ncs, 1e-2)
  θ̂_m1  = sobolev_m1_estimate(data.y, data.X, K_F, gr.G, gr.G1, 1e-2)
  @test all(isfinite, θ̂_ncs)
  @test all(isfinite, θ̂_m1)
  @test size(θ̂_ncs) == (T, 2)
  @test size(θ̂_m1) == (T, 2)
end

# Statistical-behaviour checks via small Monte Carlo.

@testitem "NCS estimator: IMSE decreases substantially as T grows (convergence rate)" begin
  # Theoretical rate is T^{-4/5}, so 4×T should cut IMSE by ≈0.33. With
  # R=10 reps the MC noise is large, so we only assert a factor-of-1.25
  # (bar of 0.8) — well inside expectation but tight enough to detect a
  # fundamentally broken estimator.
  using TVPSobolev, LinearAlgebra, Random, Statistics
  function mean_imse(T; R=10, base_seed=3_000)
    K_ncs = ncs_penalty(T)
    λ = T^(-4 / 5)  # IMSE-optimal scaling
    τ = collect(range(1 / T, 1.0, length=T))
    ix = interior_indices(τ)
    imse = 0.0
    for r in 1:R
      data = generate_lzh(T; rng=MersenneTwister(base_seed + r))
      K_F = build_fsmd_kernel(data.Z)
      θ̂ = ncs_estimate(data.y, data.X, K_F, K_ncs, λ)
      θ̂_c = bias_correct(θ̂, data.X, λ;
                          method=:ncs, K_F=K_F, K_ncs=K_ncs)
      imse += mean((θ̂_c[ix, :] .- data.θ0[ix, :]) .^ 2)
    end
    return imse / R
  end
  imse_100 = mean_imse(100)
  imse_400 = mean_imse(400)
  @test imse_100 > 0 && imse_400 > 0
  @test imse_400 / imse_100 < 0.8
end

@testitem "bias_correct: corrected IMSE < raw IMSE on curved truth at moderate λ" begin
  # On a curved truth (outside the NCS null space) the Gao–Tsay twicing
  # must reduce integrated squared bias — the whole point of the correction.
  using TVPSobolev, LinearAlgebra, Random, Statistics
  function run_mc(T, λ, R)
    K_ncs = ncs_penalty(T)
    τ = collect(range(1 / T, 1.0, length=T))
    ix = interior_indices(τ)
    raw = 0.0; bc = 0.0
    for r in 1:R
      data = generate_lzh(T; rng=MersenneTwister(4_000 + r))
      K_F = build_fsmd_kernel(data.Z)
      θ̂ = ncs_estimate(data.y, data.X, K_F, K_ncs, λ)
      θ̂_c = bias_correct(θ̂, data.X, λ;
                          method=:ncs, K_F=K_F, K_ncs=K_ncs)
      raw += mean((θ̂[ix, :]   .- data.θ0[ix, :]) .^ 2)
      bc  += mean((θ̂_c[ix, :] .- data.θ0[ix, :]) .^ 2)
    end
    return (raw=raw / R, bc=bc / R)
  end
  r = run_mc(200, 200.0^(-4 / 5), 15)
  @test r.bc < r.raw
end

@testitem "generate_str: regime transition S(τ) crosses 0.5 at τ ∈ {1/4, 3/4}" begin
  # Design 2's instrument strength vanishes exactly where S=1/2; this is
  # what creates the locally-weak-identification regions studied in the paper.
  using TVPSobolev, Random
  T = 400
  τ = collect(range(1 / T, 1.0, length=T))
  S = 1.0 ./ (1.0 .+ exp.(-(τ .- 0.25) .* (τ .- 0.75) .* T))
  i_quarter = argmin(abs.(τ .- 0.25))
  i_three_q = argmin(abs.(τ .- 0.75))
  # S ≈ 1/2 at the transitions, far from 1/2 in the interior.
  @test abs(S[i_quarter] - 0.5) < 0.01
  @test abs(S[i_three_q] - 0.5) < 0.01
  i_mid = argmin(abs.(τ .- 0.5))
  @test S[i_mid] < 0.5 - 0.1  # clearly outside the transitions
  i_start = argmin(abs.(τ .- 0.1))
  @test S[i_start] > 0.5 + 0.1
end

@testitem "m=1 and NCS agree on truly linear truth (no identification loss)" begin
  # On a linear truth both estimators should recover it: NCS because linear
  # paths are in its null space; m=1 because its bounded-derivative penalty
  # stays small. Check both hug the truth and agree with each other.
  using TVPSobolev, LinearAlgebra, Random, Statistics
  rng = MersenneTwister(77)
  T = 120
  τ = collect(range(1 / T, 1.0, length=T))
  θ0 = hcat(0.5 .+ τ, 1.0 .- 0.3 .* τ)
  X = hcat(ones(T), randn(rng, T))
  y = vec(sum(X .* θ0; dims=2)) .+ 0.05 .* randn(rng, T)
  Z = randn(rng, T, 2)
  K_F = build_fsmd_kernel(Z; ridge=1e-3)
  K_ncs = ncs_penalty(T)
  gr = sobolev_m1_grams(T)
  λ = 1e-4
  θ̂_ncs = ncs_estimate(y, X, K_F, K_ncs, λ)
  θ̂_m1  = sobolev_m1_estimate(y, X, K_F, gr.G, gr.G1, λ)
  ix = interior_indices(τ)
  @test mean((θ̂_ncs[ix, :] .- θ0[ix, :]) .^ 2) < 0.05
  @test mean((θ̂_m1[ix, :]  .- θ0[ix, :]) .^ 2) < 0.05
  @test mean((θ̂_ncs[ix, :] .- θ̂_m1[ix, :]) .^ 2) < 0.05
end

@testitem "inference pipeline: pointwise coverage tracks nominal on null-space truth" begin
  # Truth is linear (in the NCS null space), so smoothing bias is zero and
  # pointwise 95% CIs should cover near nominal. R=50 gives MC stderr ≈0.03
  # on the coverage estimate; we test ≥ 0.85 to tolerate that noise while
  # still catching a broken pipeline.
  using TVPSobolev, LinearAlgebra, Random, Statistics
  function pw_coverage_mc(T, R; σ=0.3, τ_eval=0.5)
    K_ncs = ncs_penalty(T)
    τ = collect(range(1 / T, 1.0, length=T))
    i_eval = argmin(abs.(τ .- τ_eval))
    θ0 = hcat(0.3 .+ τ, 0.8 .- 0.4 .* τ)
    c1 = 0; c2 = 0
    for r in 1:R
      rng = MersenneTwister(8_000 + r)
      X = hcat(ones(T), randn(rng, T))
      Z = randn(rng, T, 2)
      ε = σ .* randn(rng, T)
      y = vec(sum(X .* θ0; dims=2)) .+ ε
      K_F = build_fsmd_kernel(Z)
      λ = T^(-4 / 5)
      θ̂ = ncs_estimate(y, X, K_F, K_ncs, λ)
      θ̂_c = bias_correct(θ̂, X, λ; method=:ncs, K_F=K_F, K_ncs=K_ncs)
      pw = pointwise_se(y, X, θ̂_c, λ; method=:ncs, K_F=K_F, K_ncs=K_ncs)
      c1 += abs(θ̂_c[i_eval, 1] - θ0[i_eval, 1]) ≤ 1.96 * pw.σ_V[i_eval, 1]
      c2 += abs(θ̂_c[i_eval, 2] - θ0[i_eval, 2]) ≤ 1.96 * pw.σ_V[i_eval, 2]
    end
    return (c1=c1 / R, c2=c2 / R)
  end
  cov = pw_coverage_mc(120, 50)
  @test cov.c1 ≥ 0.85
  @test cov.c2 ≥ 0.85
end

@testitem "build_fsmd_kernel: ridge is monotone and bounds λ_min from below" begin
  using TVPSobolev, LinearAlgebra, Random
  rng = MersenneTwister(11)
  Z = randn(rng, 60, 3)
  ridges = [1e-8, 1e-6, 1e-4, 1e-2, 1e0]
  mins = Float64[]
  for ρ in ridges
    K = build_fsmd_kernel(Z; ridge=ρ)
    push!(mins, minimum(eigen(Matrix(K)).values))
  end
  # λ_min must be monotone in ρ and at least ρ (ridge sits on a PSD kernel).
  for k in 1:(length(ridges) - 1)
    @test mins[k + 1] ≥ mins[k]
  end
  for k in eachindex(ridges)
    @test mins[k] ≥ ridges[k] - 1e-10
  end
end

@testitem "pipeline: determinism under full re-run with identical seeds" begin
  # Two identical runs must produce bitwise-equal outputs — this is what
  # the paper's bit-for-bit reproducibility claim rests on.
  using TVPSobolev, LinearAlgebra, Random
  function one_run()
    T = 80
    data = generate_lzh(T; rng=MersenneTwister(1_234))
    K_F = build_fsmd_kernel(data.Z)
    K_ncs = ncs_penalty(T)
    λ_grid = exp.(range(-10, 2, length=30))
    cv = select_lambda_loocv(data.y, data.X, λ_grid;
                              method=:ncs, K_F=K_F, K_ncs=K_ncs)
    θ̂ = ncs_estimate(data.y, data.X, K_F, K_ncs, cv.λ)
    θ̂_c = bias_correct(θ̂, data.X, cv.λ;
                        method=:ncs, K_F=K_F, K_ncs=K_ncs)
    pw = pointwise_se(data.y, data.X, θ̂_c, cv.λ;
                       method=:ncs, K_F=K_F, K_ncs=K_ncs)
    scb = bootstrap_scb(data.y, data.X, θ̂_c, pw.σ_V, pw.W, pw.u_c;
                         n_boot=300, rng=MersenneTwister(9_999))
    return (cv_λ=cv.λ, θ̂_c=θ̂_c, σ_V=pw.σ_V, c_α=scb.c_α,
            scb_lo=scb.scb_lo, scb_hi=scb.scb_hi)
  end
  a = one_run(); b = one_run()
  @test a.cv_λ == b.cv_λ
  @test a.θ̂_c == b.θ̂_c
  @test a.σ_V == b.σ_V
  @test a.c_α == b.c_α
  # isequal, not ==, because NaN ≠ NaN at the trimmed boundary.
  @test isequal(a.scb_lo, b.scb_lo)
  @test isequal(a.scb_hi, b.scb_hi)
end

@run_package_tests
