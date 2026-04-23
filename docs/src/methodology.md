# Methodology

## Identification via the instrument kernel

The conditional-moment restriction
``\mathbb{E}[u_t \mid Z_t] = 0``
is equivalent, by Bierens (1982)'s Fourier characterisation, to a
continuum of unconditional moments indexed by frequency ``\xi``:

```math
\mathbb{E}[u_t \mid Z_t] = 0 \;\; \text{a.s.}
\quad\Longleftrightarrow\quad
\mathbb{E}[u_t\, e^{i\xi^\top Z_t}] = 0 \;\; \forall \xi \in \mathbb{R}^q.
```

Integrating the squared moment against a positive measure ``\mu`` on
frequencies collapses the continuum to a single scalar criterion:

```math
Q_T(\theta) = \sum_{t,s=1}^T u_t\, u_s\, K(Z_t - Z_s),
\qquad K(z) = \int e^{i\xi^\top z}\, d\mu(\xi),
```

with ``K`` positive definite by Bochner. [`build_fsmd_kernel`](@ref)
builds the empirical Gram matrix using a Gaussian ``\mu``, normalised by
``1/T`` with a small ridge on the diagonal. No first-stage
parametrisation is needed: the instrument kernel handles the
``Z``-direction, the penalty handles the ``\tau``-direction.

## Two choices of smoothness penalty

### ``m = 1``: natural linear spline

The minimiser of the ``m = 1`` problem is a natural linear spline with
knots at the observed ``\tau_i``: piecewise linear, constant outside
the knot range. We parametrise it via Wahba (1990)'s
Bernoulli-polynomial RKHS. The penalty matrix is

```math
\int_0^1 \theta'(\tau)^2\, d\tau = \boldsymbol\alpha^\top \mathbf{G}_1\, \boldsymbol\alpha,
```

with ``\mathbf{G}_1 = k_1 k_1^\top + k_2(|\Delta|)`` built from
``k_1(\tau) = \tau - 1/2`` and ``k_2(u) = ((u - 1/2)^2 - 1/12)/2``. The
null space is the constants; the IMSE-optimal rate is
``\lambda \asymp T^{-2/3}``; effective degrees of freedom scale as
``T^{1/3}``. See [`sobolev_m1_grams`](@ref) and
[`sobolev_m1_estimate`](@ref).

### ``m = 2``: natural cubic spline

The minimiser of the ``m = 2`` problem is a natural cubic spline with
knots at the ``\tau_i``: piecewise cubic, continuous second derivative,
linear outside the knot range (Green & Silverman 1994). We parametrise
it directly by its knot values, giving the quadratic form

```math
\int_0^1 s''(\tau)^2\, d\tau = \mathbf{g}^\top \mathbf{K}_{\mathrm{NCS}}\, \mathbf{g},
\qquad \mathbf{K}_{\mathrm{NCS}} = \mathbf{Q}\, \mathbf{R}^{-1}\, \mathbf{Q}^\top,
```

with banded ``\mathbf{Q}`` and tridiagonal ``\mathbf{R}`` (see
[`ncs_penalty`](@ref)). The null space is ``\{1, \tau\}``; the
IMSE-optimal rate is ``\lambda \asymp T^{-4/5}``; effective degrees of
freedom scale as ``T^{1/5}``, so the integrated squared second
derivative of the fitted path stays ``O(1)`` as ``T`` grows.

## Inference pipeline

Both estimators are linear smoothers:
``\hat{\boldsymbol\alpha} = \mathbf{S}_\lambda\, y``.

[`bias_correct`](@ref) applies one round of Gao–Tsay twicing,
``\hat\theta_c = 2\hat\theta - \mathbf{S}_\lambda(\mathbf{D}\hat\theta)``.
Using the identity
``\mathbf{W}\mathbf{D} - \mathbf{I} = -(\mathbf{I} - \mathbf{S}_\lambda\mathbf{D})^2``
with ``\mathbf{W} = 2\mathbf{S}_\lambda - \mathbf{S}_\lambda\mathbf{D}\mathbf{S}_\lambda``,
this drops the first-order smoothing bias from ``O(\lambda)`` to
``O(\lambda^2)``.

[`pointwise_se`](@ref) returns
``\hat\sigma_V(\tau)^2 = \sum_t \mathbf{W}_{\tau, t}^2\, \hat u_{c,t}^2``,
the variance of the bias-corrected estimator at a point. Pass
`hac_lags = L` for a Newey–West Bartlett correction with ``L`` lags;
the NKPC application uses ``L = 12``.

[`bootstrap_scb`](@ref) draws block-Gaussian multipliers ``R_t`` (one
``\mathcal{N}(0, 1)`` per block of length ``\lfloor T^{1/3} \rfloor``,
repeated within the block), forms
``V^*(\tau) = \sum_t \mathbf{W}_{\tau, t}\, \hat u_{c,t}\, R_t``,
and uses the empirical ``(1-\alpha)``-quantile of
``\sup_\tau |V^*(\tau)| / \hat\sigma_V(\tau)`` as the simultaneous-band
critical value.

## Trade-offs between the two variants

Both estimators are reliable. The Monte Carlo in the paper documents
complementary finite-sample trade-offs:

- [`ncs_estimate`](@ref) reaches nominal pointwise and simultaneous
  coverage on the strongly identified design, benefits from the faster
  ``T^{-4/5}`` rate, and recovers linear trends without shrinkage via
  the null space ``\{1, \tau\}``.
- [`sobolev_m1_estimate`](@ref) has a one-dimensional (constant) null
  space, which gives it more freedom to track curvature under locally
  weak identification at the cost of paths whose integrated second
  derivative grows with ``T``.
