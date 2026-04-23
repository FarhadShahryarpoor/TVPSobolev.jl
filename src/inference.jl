"""
    select_lambda_loocv(y, X, λ_grid; method, kwargs...)

Pick `λ` by the Craven–Wahba LOOCV shortcut over the grid. Grid points
where the normal matrix's Cholesky fails are skipped (`NaN` in the
returned curve). Note that under smooth minimum distance weighting the shortcut is an
approximation to true LOO, but a standard one. Returns
`(λ, cv, grid, cv_curve)`.
"""
function select_lambda_loocv(y::AbstractVector, X::AbstractMatrix,
                             λ_grid::AbstractVector;
                             method::Symbol,
                             K_F::AbstractMatrix,
                             G::Union{AbstractMatrix,Nothing}=nothing,
                             G1::Union{AbstractMatrix,Nothing}=nothing,
                             K_ncs::Union{AbstractMatrix,Nothing}=nothing)
  T, p = size(X)
  cv_curve = fill(NaN, length(λ_grid))
  best_cv = Inf; best_λ = λ_grid[1]
  for (k, λ) in enumerate(λ_grid)
    Q = try
      _fit_hat(X, λ; method, K_F, G, G1, K_ncs)
    catch
      continue
    end
    ŷ = Q * y
    r = y .- ŷ
    s = 0.0; valid = true
    @inbounds for t in 1:T
      d = 1.0 - Q[t, t]
      if abs(d) < 1e-6; valid = false; break; end
      s += (r[t] / d)^2
    end
    valid || continue
    s /= T
    cv_curve[k] = s
    if s < best_cv
      best_cv = s; best_λ = λ
    end
  end
  return (λ=best_λ, cv=best_cv, grid=λ_grid, cv_curve=cv_curve)
end

function _fit_hat(X::AbstractMatrix, λ::Real;
                  method::Symbol, K_F::AbstractMatrix,
                  G, G1, K_ncs)
  p = size(X, 2)
  if method === :sobolev_m1
    Φ = _build_sobolev_phi(X, G)
    M = Symmetric(Φ' * K_F * Φ .+ λ .* kron(I(p), G1))
    F = cholesky(M, check=true)
    return Φ * (F \ Matrix(Φ' * K_F))
  elseif method === :ncs
    D = build_design(X)
    DtKF = D' * K_F
    M = Symmetric(DtKF * D .+ λ .* kron(I(p), Matrix(K_ncs)))
    F = cholesky(M, check=true)
    return D * (F \ Matrix(DtKF))
  else
    error("unknown method $method")
  end
end

"""
    pointwise_se(y, X, θ̂_c, λ; method, hac_lags=0, kwargs...)

Pointwise SE of the bias-corrected estimator at every grid point.

With `hac_lags = 0` (default) this is the HC/iid formula
``σ̂_V(τ)² = Σ_t W_{τ,t}² û_{c,t}²``. With `hac_lags = L > 0` this is
Newey–West (Bartlett kernel) with `L` lags, which is what the NKPC
application uses to absorb serial correlation in the residuals.

Returns `(σ_V, W, u_c)`.
"""
function pointwise_se(y::AbstractVector, X::AbstractMatrix,
                      θ̂_c::AbstractMatrix, λ::Real;
                      method::Symbol,
                      K_F::AbstractMatrix,
                      G::Union{AbstractMatrix,Nothing}=nothing,
                      G1::Union{AbstractMatrix,Nothing}=nothing,
                      K_ncs::Union{AbstractMatrix,Nothing}=nothing,
                      hac_lags::Integer=0)
  T, p = size(X)
  W = _build_W(X, λ; method, K_F, G, G1, K_ncs)
  u_c = y .- vec(sum(X .* θ̂_c; dims=2))
  σ_V = zeros(T, p)
  if hac_lags ≤ 0
    @inbounds for j in 1:p, i in 1:T
      row = (j - 1) * T + i
      σ_V[i, j] = sqrt(max(sum((@view W[row, :]) .^ 2 .* u_c .^ 2), 1e-20))
    end
  else
    L = hac_lags
    @inbounds for j in 1:p, i in 1:T
      row = (j - 1) * T + i
      w = @view W[row, :]
      s = 0.0
      for t in 1:T
        s += (w[t] * u_c[t])^2
      end
      for ℓ in 1:L
        kw = 1.0 - ℓ / (L + 1)
        cross = 0.0
        for t in 1:(T - ℓ)
          cross += w[t] * w[t + ℓ] * u_c[t] * u_c[t + ℓ]
        end
        s += 2 * kw * cross
      end
      σ_V[i, j] = sqrt(max(s, 1e-20))
    end
  end
  return (σ_V=σ_V, W=W, u_c=u_c)
end

# W = 2S - SDS: the Gao–Tsay weight matrix for the bias-corrected fit.
function _build_W(X::AbstractMatrix, λ::Real;
                  method::Symbol, K_F::AbstractMatrix,
                  G, G1, K_ncs)
  if method === :sobolev_m1
    sm = sobolev_m1_smoother(X, K_F, G, G1, λ)
    D_θ = _build_dtheta(X)
    return 2 .* sm.Sθ .- sm.Sθ * D_θ * sm.Sθ
  elseif method === :ncs
    sm = ncs_smoother(X, K_F, K_ncs, λ)
    return 2 .* sm.S .- sm.S * (sm.D * sm.S)
  else
    error("unknown method $method")
  end
end

# D_θ (m=1): T × pT with [D_θ]_{t, (j-1)T+t} = X[t,j], shaped so that
# W = 2Sθ - Sθ D_θ Sθ composes cleanly.
function _build_dtheta(X::AbstractMatrix)
  T, p = size(X)
  Dθ = zeros(T, p * T)
  @inbounds for j in 1:p, t in 1:T
    Dθ[t, (j - 1) * T + t] = X[t, j]
  end
  return Dθ
end

"""
    bootstrap_scb(y, X, θ̂_c, σ_V, W, u_c; n_boot=1000, block_length=nothing,
                  trim=0.05, nominal=0.95, rng=Random.default_rng())

SCB critical values by block-multiplier bootstrap. For each draw,
simulate `V*(τ) = Σ W_{τ,t} û_{c,t} R_t` with block-Gaussian `R_t`
(default block length `⌊T^{1/3}⌋`) and record `sup|V*|/σ̂_V`. The band is
`θ̂_c(τ) ± c_α · σ̂_V(τ)`, with NaNs outside the `[trim, 1-trim]`
interior. Returns `(c_α, scb_lo, scb_hi)`.
"""
function bootstrap_scb(y::AbstractVector, X::AbstractMatrix,
                       θ̂_c::AbstractMatrix, σ_V::AbstractMatrix,
                       W::AbstractMatrix, u_c::AbstractVector;
                       n_boot::Integer=1000,
                       block_length::Union{Nothing,Integer}=nothing,
                       trim::Real=0.05,
                       nominal::Real=0.95,
                       rng::AbstractRNG=Random.default_rng())
  T, p = size(X)
  mb = isnothing(block_length) ? max(1, floor(Int, T^(1 / 3))) : block_length
  τ_grid = collect(range(1 / T, 1.0, length=T))
  interior = interior_indices(τ_grid; trim=trim)
  c_α = zeros(p)
  for j in 1:p
    T_star = zeros(n_boot)
    for b in 1:n_boot
      R = _block_multipliers(rng, T, mb)
      sv = 0.0
      @inbounds for i in interior
        row = (j - 1) * T + i
        V = sum((@view W[row, :]) .* u_c .* R)
        sv = max(sv, abs(V) / max(σ_V[i, j], 1e-10))
      end
      T_star[b] = sv
    end
    sort!(T_star)
    c_α[j] = T_star[ceil(Int, nominal * n_boot)]
  end
  scb_lo = fill(NaN, T, p); scb_hi = fill(NaN, T, p)
  @inbounds for j in 1:p, i in interior
    scb_lo[i, j] = θ̂_c[i, j] - c_α[j] * σ_V[i, j]
    scb_hi[i, j] = θ̂_c[i, j] + c_α[j] * σ_V[i, j]
  end
  return (c_α=c_α, scb_lo=scb_lo, scb_hi=scb_hi)
end

# One N(0,1) draw per block, repeated within the block.
function _block_multipliers(rng::AbstractRNG, T::Integer, mb::Integer)
  R = zeros(T)
  n_blocks = ceil(Int, T / mb)
  R_block = randn(rng, n_blocks)
  @inbounds for k in 1:n_blocks
    a = (k - 1) * mb + 1
    b = min(k * mb, T)
    R[a:b] .= R_block[k]
  end
  return R
end
