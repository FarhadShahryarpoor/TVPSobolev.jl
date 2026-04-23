"""
    ncs_penalty(T)

Natural cubic spline roughness matrix `K` on the grid `τ = 1/T, …, 1`.
For any vector `g` of knot values, `gᵀ K g = ∫(s'')²` where `s` is the
unique natural cubic spline through `g`. Null space is `{1, τ}`. Built via
the Green–Silverman `Q·R⁻¹·Qᵀ` construction.
"""
function ncs_penalty(T::Integer)
  h = 1.0 / T
  Q = spzeros(T, T - 2)
  @inbounds for j in 1:(T - 2)
    Q[j,     j] =  1 / h
    Q[j + 1, j] = -2 / h
    Q[j + 2, j] =  1 / h
  end
  R = spzeros(T - 2, T - 2)
  @inbounds for j in 1:(T - 2)
    R[j, j] = 2h / 3
    if j < T - 2
      R[j,     j + 1] = h / 6
      R[j + 1, j]     = h / 6
    end
  end
  Rdense = Matrix(R)
  Qdense = Matrix(Q)
  RinvQᵀ = Rdense \ Qdense'
  K = Qdense * RinvQᵀ
  return Symmetric((K .+ K') ./ 2)
end

"""
    ncs_estimate(y, X, K_F, K_ncs, λ)

Closed-form NCS F-SMD estimate. Parametrises `θ̂ⱼ(τᵢ)` directly as the
knot value `α̂_{(j-1)T + i}` and solves one linear system. Returns `θ̂`
as a `T × p` matrix.
"""
function ncs_estimate(y::AbstractVector, X::AbstractMatrix,
                      K_F::AbstractMatrix, K_ncs::AbstractMatrix, λ::Real)
  T, p = size(X)
  D = build_design(X)
  DtKF = D' * K_F
  M = Symmetric(DtKF * D .+ λ .* kron(I(p), Matrix(K_ncs)))
  α = M \ (DtKF * y)
  return reshape(α, T, p)
end

"""
    ncs_smoother(X, K_F, K_ncs, λ)

Linear-smoother `(S, D)` for the NCS estimator: `α̂ = S·y` with design
`D` from [`build_design`](@ref). Consumed by the inference pipeline.
"""
function ncs_smoother(X::AbstractMatrix, K_F::AbstractMatrix,
                      K_ncs::AbstractMatrix, λ::Real)
  p = size(X, 2)
  D = build_design(X)
  DtKF = D' * K_F
  M = Symmetric(DtKF * D .+ λ .* kron(I(p), Matrix(K_ncs)))
  F = cholesky(M)
  S = F \ Matrix(DtKF)
  return (S=S, D=D)
end
