"""
    true_theta(τ)

True coefficient paths for both MC designs: an exponential intercept
`θ₁(τ) = 0.2·exp(-0.7 + 3.5τ)` and a linear trend plus Gaussian bump
`θ₂(τ) = 2τ + exp(-16(τ - 1/2)²) - 1`. Returns a `|τ| × 2` matrix.
"""
function true_theta(τ::AbstractVector)
  θ1 = 0.2 .* exp.(-0.7 .+ 3.5 .* τ)
  θ2 = 2.0 .* τ .+ exp.(-16.0 .* (τ .- 0.5) .^ 2) .- 1.0
  return hcat(θ1, θ2)
end

"""
    generate_lzh(T; ρ=0.8, rng=Random.default_rng())

Simulate Design 1: four time-varying-AR instruments, linear first stage,
`corr(ε,v) = ρ`. Returns `(y, X, Z, u, θ0, τ)` with `X` already
intercept-augmented (`T × 2`) and `Z` of shape `T × 4`.
"""
function generate_lzh(T::Integer; ρ::Real=0.8, rng::AbstractRNG=Random.default_rng())
  τ = collect(range(1 / T, 1.0, length=T))
  θ0 = true_theta(τ)
  Z = zeros(T, 4)
  @inbounds for t in 2:T
    ar = 0.9 * sin(4π * τ[t])
    Z[t, 1] = ar * Z[t - 1, 1] + randn(rng)
    Z[t, 2] = 0.5 + 0.8 * sin(4π * τ[t]) * Z[t - 1, 2] + randn(rng)
    Z[t, 3] = ar * Z[t - 1, 3] + randn(rng)
    Z[t, 4] = ar * Z[t - 1, 4] + randn(rng)
  end
  ε = randn(rng, T); v = randn(rng, T)
  u = ρ .* v .+ sqrt(1 - ρ^2) .* ε
  X = 0.8 .* Z[:, 1] .+ 1.3 .* Z[:, 2] .+ Z[:, 3] .+ Z[:, 4] .+ v
  X_mat = hcat(ones(T), X)
  y = vec(sum(X_mat .* θ0; dims=2)) .+ u
  return (y=y, X=X_mat, Z=Z, u=u, θ0=θ0, τ=τ)
end

"""
    generate_str(T; ρ=0.8, rng=Random.default_rng())

Simulate Design 2: one `Uniform(-2, 2)` instrument, regime-switching
cubic first stage whose relevance vanishes at `τ = 1/4, 3/4`. Same return
tuple as [`generate_lzh`](@ref); `Z` is `T × 1`.
"""
function generate_str(T::Integer; ρ::Real=0.8, rng::AbstractRNG=Random.default_rng())
  τ = collect(range(1 / T, 1.0, length=T))
  θ0 = true_theta(τ)
  Z_unif = rand(rng, T) .* 4 .- 2
  Z = reshape(Z_unif, T, 1)
  S = 1.0 ./ (1.0 .+ exp.(-(τ .- 0.25) .* (τ .- 0.75) .* T))
  ε = randn(rng, T); v = randn(rng, T)
  u = ρ .* v .+ sqrt(1 - ρ^2) .* ε
  X = 10.0 .* (2.0 .* S .- 1.0) .* (Z_unif .- (2.0 / 5.0) .* Z_unif .^ 3) .+ v
  X_mat = hcat(ones(T), X)
  y = vec(sum(X_mat .* θ0; dims=2)) .+ u
  return (y=y, X=X_mat, Z=Z, u=u, θ0=θ0, τ=τ)
end
