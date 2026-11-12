# Regularizers mainly meant for the splitting solvers (ADMM, Chambolle–Pock): the ℓ1 norm of the
# object (sparsity) and second-order total generalized variation (TGV).

export L1, TGV

"""
    L1()

ℓ1 norm of the object, `Σᵢ |xᵢ|` (sparsity prior). Use with `λ` as any other regularizer.

Supported by the gradient-based optimizers (the value is differentiable for `x ≠ 0`; with the
default mapping `x ≥ 0` it is the linear term `Σᵢ xᵢ`) and by [`ADMM`](@ref) and
[`ChambollePock`](@ref), which use its exact proximal map (soft thresholding, combined with the
projection onto `x ≥ 0`).

!!! note "Poisson noise"
    With the `Poisson()` loss, a normalized PSF and no `background`, the forward model conserves
    the total intensity, so `λ Σᵢ xᵢ` only rescales the solution (by `1/(1 + λ)`) and does not
    promote sparsity. Use it with `Gauss()` or `Anscombe()`, where it soft-thresholds the object.
"""
L1() = TaggedRegularizer(:L1, _l1_norm, NamedTuple())
_l1_norm(x) = sum(abs, x)

"""
    TGV(; num_dims=nothing, sum_dims=nothing, weights=nothing, ratio=2)

Second-order total generalized variation (K. Bredies, K. Kunisch, T. Pock, SIAM J. Imaging Sci.
3, 492 (2010)),

    TGV(x) = min_w  Σᵢ ‖(∇x − w)ᵢ‖₂ + ratio · Σᵢ ‖(ε(w))ᵢ‖_F ,

with an auxiliary vector field `w` (one component per dimension in `sum_dims`) and its symmetrized
gradient `ε(w) = (∇w + ∇wᵀ)/2`. Where the object is piecewise constant, `w = 0` and TGV equals
[`TV`](@ref) (same `λ`); on smooth ramps `w` follows the gradient and only its change is
penalized, which avoids the staircasing of TV while keeping edges sharp (unlike the Total Hessian
[`TH`](@ref), which smooths the edges themselves).

`w` is an internal variable of the solver and is not an input. TGV is only supported by
[`ChambollePock`](@ref).

# Arguments
- `num_dims`, `sum_dims`, `weights`: as for [`TV`](@ref) (differences along `sum_dims`, a dimension
  `d` is weighted like `weights[k]` in TV, i.e. its differences are scaled by `√weights[k]`).
- `ratio=2`: weight α₀/α₁ of the second-order term relative to the first-order term (typical 1–3;
  larger values approach TV).
"""
function TGV(; num_dims=nothing, sum_dims=nothing, weights=nothing, ratio=2.0)
    ratio > 0 || throw(ArgumentError("TGV: ratio must be positive"))
    return TaggedRegularizer(:TGV, _tgv_unsupported, (; num_dims, sum_dims, weights, ratio=Float64(ratio)))
end
_tgv_unsupported(x) = throw(ArgumentError(
    "TGV() has an internal auxiliary field and is only supported by opt=ChambollePock()."))

_reg_kind(reg) = reg isa TaggedRegularizer ? reg.kind : :other
