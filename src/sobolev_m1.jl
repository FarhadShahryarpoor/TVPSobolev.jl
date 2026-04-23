"""
    sobolev_m1_grams(T)

Wahba's RKHS Gram matrices on the grid `τ = 1/T, 2/T, …, 1`. `G1`
implements the penalty `∫(θ')²` and has the constants as its null space;
`G = 1·1ᵀ + G1` is the full RKHS Gram. Returns `(G, G1, τ)`.
"""
function sobolev_m1_grams(T::Integer)
  τ = collect(range(1 / T, 1.0, length=T))
  k1 = τ .- 0.5
  U = abs.(τ .- τ')
  k1_U = U .- 0.5
  k2_U = (k1_U .^ 2 .- 1 / 12) ./ 2
  G1 = k1 * k1' .+ k2_U
  G = ones(T, T) .+ G1
  return (G=G, G1=G1, τ=τ)
end

"""
    sobolev_m1_estimate(y, X, K_F, G, G1, λ)

Closed-form solution of the m=1 Sobolev smooth minimum distance problem. Coefficients are
expanded in the representer basis `θⱼ(τ) = Σᵢ αⱼᵢ G(τ, τᵢ)` and `α` solves
one linear system. Returns `θ̂` on the grid as a `T × p` matrix.
"""
function sobolev_m1_estimate(y::AbstractVector, X::AbstractMatrix,
                             K_F::AbstractMatrix, G::AbstractMatrix,
                             G1::AbstractMatrix, λ::Real)
  T, p = size(X)
  Φ = _build_sobolev_phi(X, G)
  G1_tilde = kron(I(p), G1)
  M = Symmetric(Φ' * K_F * Φ .+ λ .* G1_tilde)
  α̃ = M \ (Φ' * (K_F * y))
  θ̂ = zeros(T, p)
  @inbounds for j in 1:p
    idx = ((j - 1) * T + 1):(j * T)
    θ̂[:, j] = G * α̃[idx]
  end
  return θ̂
end

"""
    sobolev_m1_smoother(X, K_F, G, G1, λ)

Linear-smoother matrices for the m=1 estimator. `S` maps `y` to `α̂`,
and `Sθ` maps `y` to coefficient-space `θ̂` (row `(j-1)T+i` gives
`θ̂ⱼ(τᵢ)`). Consumed by [`bias_correct`](@ref) and [`pointwise_se`](@ref).
Returns `(S, Sθ, Φ)`.
"""
function sobolev_m1_smoother(X::AbstractMatrix, K_F::AbstractMatrix,
                             G::AbstractMatrix, G1::AbstractMatrix, λ::Real)
  T, p = size(X)
  Φ = _build_sobolev_phi(X, G)
  G1_tilde = kron(I(p), G1)
  M = Symmetric(Φ' * K_F * Φ .+ λ .* G1_tilde)
  F = cholesky(M)
  S = F \ Matrix(Φ' * K_F)
  G_block = kron(I(p), G)
  Sθ = G_block * S
  return (S=S, Sθ=Sθ, Φ=Φ)
end

function _build_sobolev_phi(X::AbstractMatrix, G::AbstractMatrix)
  T, p = size(X)
  Φ = zeros(T, p * T)
  @inbounds for t in 1:T
    Φ[t, :] .= kron(X[t, :], G[t, :])
  end
  return Φ
end
