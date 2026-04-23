# TVPSobolev.jl

Penalised SMD estimators of time-varying coefficients under
endogeneity. Two variants (a natural linear spline and a natural cubic
spline) share the same instrument kernel, the same Gao–Tsay bias
correction, and the same multiplier-bootstrap simultaneous confidence
band.

## Model

Outcome ``y_t``, possibly endogenous regressor ``X_t \in \mathbb{R}^p``,
instruments ``Z_t \in \mathbb{R}^q``, ``t = 1, \ldots, T``. The package
estimates the coefficient path ``\theta : [0,1] \to \mathbb{R}^p`` in

```math
y_t = X_t^\top\, \theta(t/T) + u_t,
\qquad
\mathbb{E}[u_t \mid Z_t] = 0 \;\; \text{a.s.}
```

The first-stage relation between ``X_t`` and ``Z_t`` is left
unspecified; the instrument kernel handles identification without one.

## Two variants

Both minimise

```math
\sum_{t,s} (y_t - X_t^\top \theta(\tau_t))(y_s - X_s^\top \theta(\tau_s))\, K(Z_t - Z_s)
\;+\; \lambda \int_0^1 \|\theta^{(m)}(\tau)\|^2\, d\tau,
```

and differ only in the penalty order ``m``:

| Estimator | ``m`` | Penalty | Null space | Optimal ``\lambda`` |
|-----------|:-:|---------|------------|---------------------|
| [`sobolev_m1_estimate`](@ref) | 1 | ``\int \theta'^2`` | constants | ``T^{-2/3}`` |
| [`ncs_estimate`](@ref)        | 2 | ``\int \theta''^2`` | ``\{1, \tau\}`` | ``T^{-4/5}`` |

Both have closed-form solutions and share [`build_fsmd_kernel`](@ref)
for the instrument kernel, [`bias_correct`](@ref) for Gao–Tsay twicing,
and [`bootstrap_scb`](@ref) for the simultaneous band.

## Quick start

```julia
using TVPSobolev, Random

data  = generate_lzh(500; rng = MersenneTwister(0))
K_F   = build_fsmd_kernel(data.Z)
K_ncs = ncs_penalty(500)

cv = select_lambda_loocv(data.y, data.X,
                         exp.(range(-12, 2, length = 60));
                         method = :ncs, K_F = K_F, K_ncs = K_ncs)

θ̂   = ncs_estimate(data.y, data.X, K_F, K_ncs, cv.λ)
θ̂_c = bias_correct(θ̂, data.X, cv.λ;
                    method = :ncs, K_F = K_F, K_ncs = K_ncs)
pw  = pointwise_se(data.y, data.X, θ̂_c, cv.λ;
                    method = :ncs, K_F = K_F, K_ncs = K_ncs)
scb = bootstrap_scb(data.y, data.X, θ̂_c, pw.σ_V, pw.W, pw.u_c;
                     n_boot = 1000, rng = MersenneTwister(7))
```

## Where to go next

- [Methodology](methodology.md) for the estimator construction.
- [Reproducibility](reproducibility.md) for regenerating the paper's
  tables and figures.
- [API Reference](api.md) for the exported function docstrings.
