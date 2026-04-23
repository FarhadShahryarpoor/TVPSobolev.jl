# API Reference

## Instrument kernel

```@docs
build_fsmd_kernel
```

## Data-generating processes

```@docs
true_theta
generate_lzh
generate_str
```

## Natural linear spline estimator (``m = 1``)

```@docs
sobolev_m1_grams
sobolev_m1_estimate
sobolev_m1_smoother
```

## Natural cubic spline estimator (``m = 2``)

```@docs
ncs_penalty
ncs_estimate
ncs_smoother
```

## Bias correction

```@docs
bias_correct
```

## Inference

```@docs
select_lambda_loocv
pointwise_se
bootstrap_scb
```

## Utilities

```@docs
interior_indices
roughness
build_design
```
