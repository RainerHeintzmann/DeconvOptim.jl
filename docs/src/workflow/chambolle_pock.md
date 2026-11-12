# Chambolle–Pock solver (primal–dual, no automatic differentiation)

The same problems as the [ADMM solver](admm.md) can be solved by the primal–dual algorithm of
Chambolle and Pock (also called PDHG):

```julia
res, info = deconvolution(img, psf; loss=Poisson(), regularizer=TV(), λ=0.05,
                          opt=ChambollePock(iterations=300))
```

The problem is written as a saddle-point problem
`min_x max_p ⟨Kx, p⟩ + G(x) − F*(p)` with `K = [H; K₁; …]` (the forward model and the regularizer
operators, e.g. the gradient for `TV()`), `F` the loss and the regularizer norms, and `G` the
constraint `x ≥ 0` (A. Chambolle, T. Pock, J. Math. Imaging Vis. 40, 120, 2011). Each iteration

* applies `K` once (a dual step: the proximal maps of the conjugates, obtained by the Moreau
  identity from the same data-term proxes as ADMM, a projection onto the ball of radius λ for
  `TV()`/`TH()` (after a scaling for the Huber norm, see below), eigenvalue clipping for `HS()`,
  a scaling for `Tikhonov()`),
* applies `Kᵀ` once (a primal step followed by the projection onto `x ≥ 0`, with `L1()` a
  one-sided soft thresholding),
* extrapolates `x̄ = 2x⁺ − x`.

## Compared with ADMM

* **No linear solve**: every term enters through `K` and `Kᵀ` only. ADMM needs a division in
  Fourier space, which requires all operators to be periodic convolutions.
* **Same model**: exact non-smooth regularizers (the `ϵ` of `TV()`/`TH()` is the Huber threshold
  `√ϵ`, as in ADMM), exact projection onto `x ≥ 0`, periodic boundaries with an unconstrained
  `padding` border; the same supported combinations, plus `TGV()`.
* **Convergence**: in our tests it reaches the same optimum as ADMM and L-BFGS, but needs about
  2–3 times as many iterations as ADMM for the same objective (on a 96×96×48 Poisson+TV problem:
  objective within 2·10⁻⁴ of L-BFGS after 200 iterations, ADMM after 25–75), at a similar cost
  per iteration.
* **Step sizes**: `τσ‖K‖² = step_factor²` with `‖K‖` computed exactly from the Fourier symbols;
  the ratio of τ and σ is balanced adaptively by the primal and dual residuals (T. Goldstein,
  M. Li, X. Yuan, NeurIPS 2015). Use `ChambollePock(tol=1e-3)` to stop on the relative residuals;
  the second return value is a `ChambollePockResult` with the residual `history`.

## Total generalized variation

`regularizer=TGV()` (second order, K. Bredies, K. Kunisch, T. Pock, SIAM J. Imaging Sci. 3, 492,
2010) is only available with this solver:

```julia
res, info = deconvolution(img, psf; loss=Gauss(), regularizer=TGV(ratio=2), λ=0.05,
                          opt=ChambollePock(iterations=1000))
```

`TGV(x) = min_w ‖∇x − w‖₁ + ratio·‖ε(w)‖₁` with the symmetrized gradient `ε(w)`. The vector
field `w` is an additional primal variable of the solver (it is not an input); the operator
becomes `K = [H 0; ∇ −I; 0 ε]` acting on `(x, w)`, with one dual ball of radius `λ` and one of
radius `ratio·λ`. On piecewise-constant regions `w = 0` and TGV equals TV (with the same `λ`); on
smooth ramps `w` follows the gradient, so TGV avoids the staircasing of TV while keeping edges
sharp, where the Total Hessian (`TH()`) would blur them. In a test with a ramp and a step edge
(Gauss noise, `λ = 0.05`), TV produced 6 flat steps along the ramp, TGV none, and with `ratio` 2–4
the edge kept a 10–90 % width of 0–1 pixel (TH: 4 pixels). `ratio` between 1 and 4 is typical;
for very large `ratio` only a constant `w` remains, i.e. TV up to one global slope. Each iteration
costs about `2 + N(N+1)/2` additional stencil applications and `N + N(N+1)/2` additional arrays
(N dimensions).

```@docs
ChambollePock
DeconvOptim.ChambollePockResult
```
