"""
    TVPSobolev

Sobolev-penalised SMD estimators for time-varying coefficient regression
under endogeneity. Two estimators (a natural linear spline and a natural
cubic spline) share the same instrument kernel, bias correction, and
multiplier-bootstrap SCB.
"""
module TVPSobolev

using LinearAlgebra, SparseArrays, Statistics, Random

# Instrument kernel
export build_fsmd_kernel

# DGPs
export true_theta, generate_lzh, generate_str

# Natural linear spline (m=1 penalty)
export sobolev_m1_grams, sobolev_m1_estimate, sobolev_m1_smoother

# Natural cubic spline (m=2 penalty)
export ncs_penalty, ncs_estimate, ncs_smoother

# Bias correction
export bias_correct

# Inference: LOOCV, pointwise SE, bootstrap SCB
export select_lambda_loocv, pointwise_se, bootstrap_scb

# Utilities
export interior_indices, roughness, build_design

include("utils.jl")
include("kernels.jl")
include("dgp.jl")
include("sobolev_m1.jl")
include("ncs.jl")
include("bias_correction.jl")
include("inference.jl")

end # module TVPSobolev
