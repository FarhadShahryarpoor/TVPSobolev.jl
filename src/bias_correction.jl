"""
    bias_correct(θ̂, X, λ; method, kwargs...)

One-step Gao–Tsay twicing: `θ̂_c = 2·θ̂ - S(D·θ̂)`. This drops the
first-order smoothing bias from `O(λ)` to `O(λ²)` via the identity
`W·D - I = -(I - S·D)²` (Lemma 2 in the paper).

Pass `method = :sobolev_m1` or `:ncs`; the same `K_F`, `G`/`G1` or
`K_ncs` you used for the raw estimate. The outcome `y` is not needed.
"""
function bias_correct(θ̂::AbstractMatrix, X::AbstractMatrix, λ::Real;
                      method::Symbol,
                      K_F::AbstractMatrix,
                      G::Union{AbstractMatrix,Nothing}=nothing,
                      G1::Union{AbstractMatrix,Nothing}=nothing,
                      K_ncs::Union{AbstractMatrix,Nothing}=nothing)
  fitted = vec(sum(X .* θ̂; dims=2))
  θ_sm = _apply_smoother(fitted, X, λ; method, K_F, G, G1, K_ncs)
  return 2 .* θ̂ .- θ_sm
end

function _apply_smoother(y::AbstractVector, X::AbstractMatrix, λ::Real;
                         method::Symbol,
                         K_F::AbstractMatrix,
                         G, G1, K_ncs)
  if method === :sobolev_m1
    G === nothing && error("sobolev_m1 requires G")
    G1 === nothing && error("sobolev_m1 requires G1")
    return sobolev_m1_estimate(y, X, K_F, G, G1, λ)
  elseif method === :ncs
    K_ncs === nothing && error("ncs requires K_ncs")
    return ncs_estimate(y, X, K_F, K_ncs, λ)
  else
    error("unknown method $method; expected :sobolev_m1 or :ncs")
  end
end
