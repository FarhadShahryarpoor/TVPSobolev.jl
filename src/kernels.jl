"""
    build_fsmd_kernel(Z; ridge=1e-4)

Build the F-SMD instrument kernel from `Z` (size `T × q`).

Product Gaussian with bandwidth equal to the pooled std of `Z`, scaled by
`1/T`, plus `ridge·I` on the diagonal. The normalisation and ridge match
the R reference implementation from Antoine & Sun (2024) byte-for-byte.
"""
function build_fsmd_kernel(Z::AbstractMatrix; ridge::Real=1e-4)
  T, q = size(Z)
  σ_pooled = max(std(vec(Z)), 1e-12)
  Z_std = Z ./ σ_pooled
  K = ones(T, T)
  @inbounds for ℓ in 1:q
    Δ = Z_std[:, ℓ] .- Z_std[:, ℓ]'
    K .*= exp.(-0.5 .* Δ .^ 2)
  end
  K ./= T
  K .+= ridge .* I(T)
  return Symmetric((K .+ K') ./ 2)
end
