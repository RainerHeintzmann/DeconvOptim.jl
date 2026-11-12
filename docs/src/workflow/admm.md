# ADMM solver (no automatic differentiation)

For the most common deconvolution problems, `deconvolution` can use an
alternating direction method of multipliers (ADMM) instead of L-BFGS:

```julia
res, info = deconvolution(img, psf; loss=Poisson(), regularizer=TV(), λ=0.05,
                          opt=ADMM(iterations=100))
```

ADMM splits the problem into simple sub-steps, each of which is solved exactly
(M. Figueiredo, J. Bioucas-Dias, IEEE TIP 19(12), 3133, 2010):

* a linear step `(HᵀH + ΣKᵀK + I) x = …`, which is a single division in Fourier space,
* the proximal operator of the loss (closed form for `Poisson()` and `Gauss()`; for
  `Anscombe(b)` a few per-pixel Newton steps on a cubic, no Hessian matrix is formed),
* the proximal operator of the regularizer (isotropic (Huber) shrinkage for `TV()`/`TH()`,
  eigenvalue soft-thresholding of the 2×2 Hessian for `HS()`
  (S. Lefkimmiatis, M. Unser, IEEE TIP 22(11), 4314, 2013); quadratic
  `Tikhonov()` terms are absorbed into the linear step),
* a projection onto `x ≥ 0` (replacing the `mapping`), combined with the soft thresholding of
  `L1()` when that regularizer is used.

## Differences to the L-BFGS path

* **No automatic differentiation**: neither Zygote nor the `rrule`s are used.
* **Exact non-smooth regularizers**: the ℓ1-type norms are minimized exactly. The smoothing
  constant `ϵ` of `TV`/`TH` becomes the threshold `δ = √ϵ` of a Huber norm (quadratic for
  `‖∇x‖ < δ`, linear above; it has the same curvature at 0 and the same asymptote as the
  `√(‖∇x‖² + ϵ)` of the L-BFGS path). Larger `ϵ` (e.g. `1e-2` in units of the normalized data)
  reduces staircasing; the default `1f-8` is practically plain TV. `HS` is always exact.
* **Convex formulation**: positivity is a projection, not the `x = v²` mapping.
  The result therefore reflects `λ` and depends much less on the number of
  iterations (L-BFGS results are often implicitly regularized by stopping early).
  Expect to choose a somewhat different `λ`.
* **Periodic boundaries** of the reconstruction grid. With `padding > 0` the
  padded border is not constrained by the data
  (M. Almeida, M. Figueiredo, IEEE TIP 22(8), 3074, 2013), which
  removes wrap-around artifacts for objects extending beyond the field of view.
* **Convergence**: the penalty `ρ` has a tuned default and is adapted by residual
  balancing. Use `ADMM(tol=1e-3)` to stop on the relative primal/dual residuals.
  The second return value is an `ADMMResult` with the residual `history`.

## Supported combinations

| `loss` | `regularizer` |
|---|---|
| `Poisson()`, `Gauss()`, `Anscombe(b)` | `nothing`, `TV()`, `TH()`, `HS()` (p=1), `Tikhonov()` (all modes), `L1()` |

`L1()` (sparsity, `λ Σ|x|`) only makes sense with `Gauss()` or `Anscombe()`: with `Poisson()`, a
normalized PSF and no background the convolution conserves the total intensity, and `λ Σx` then
merely scales the result by `1/(1 + λ)` (a warning is issued).

Other losses or regularizers (e.g. `GR()`, `ScaledGauss()`) and custom forward
models still require the L-BFGS path or [`invert`](@ref).

```@docs
ADMM
DeconvOptim.ADMMResult
```
