"""
    interior_indices(τ; trim=0.05)

Indices where `τ[i] ∈ [trim, 1 - trim]`. Used to exclude boundary
points from coverage summaries, since the smoother leaks mass there.
"""
function interior_indices(τ::AbstractVector; trim::Real=0.05)
  findall(t -> trim ≤ t ≤ 1 - trim, τ)
end

"""
    roughness(x)

Mean squared second difference of `x` — a discrete proxy for `∫(f'')²`.
Rescale by `T⁴` for comparisons across grid sizes.
"""
function roughness(x::AbstractVector)
  d² = diff(diff(x))
  return sum(d² .^ 2) / length(d²)
end

"""
    build_design(X)

Design matrix `D` of size `T × pT` with `D[t, (j-1)T + t] = X[t, j]`,
used by the NCS parametrisation that places unknowns at the grid knots.
Sparse by construction but returned dense for BLAS.

"""
function build_design(X::AbstractMatrix)
  T, p = size(X)
  D = zeros(T, p * T)
  @inbounds for t in 1:T, j in 1:p
    D[t, (j - 1) * T + t] = X[t, j]
  end
  return D
end
