# Smooth positivity penalty, an alternative to the domain constraint by a
# mapping (e.g. `Non_negative()`, x = v²). Mappings can stall near zero (the
# gradient of v² vanishes at v = 0), whereas a penalty on negative values keeps
# the problem in the original variables. The penalty weight is ramped up over a
# few continuation stages: early on the solver may cross into negative values to
# find a better basin, positivity is enforced strongly only at the end.

export Positivity

"""
    Positivity(; weight=20, σ=0.1, mode="huber", stages=3, ramp=√10, clip=true)

Smooth penalty on negative values for the gradient-based (Optim) path of
[`deconvolution`](@ref) and [`invert`](@ref), passed as `positivity=Positivity()`
together with `mapping=nothing`. It is added to the objective as
`w_s * Σᵢ φ(xᵢ)` next to the (shape) `regularizer`.

# Penalty shapes `φ` (`mode`)
- `"softplus"` ("log-positive"): `φ(x) = σ log(1 + exp(-x/σ))`, smooth everywhere,
  grows linearly (slope 1) for `x ≪ -σ` and decays exponentially for `x ≫ σ`.
- `"huber"`: `0` for `x ≥ 0`, `x²/(2σ)` for `-σ < x < 0`, `-x - σ/2` below;
  continuously differentiable, no influence at all on positive values.
- `"quadratic"`: `min(x, 0)²` (`σ` unused); only approximately enforces `x ≥ 0`.

The slope-1 shapes (`softplus`, `huber`) are *exact* penalties: once the weight
exceeds the magnitude of the remaining gradient pushing a value below zero, the
minimizer is non-negative (up to `σ`). For the mean-normalized data used by
`deconvolution`, the Poisson data term has a gradient ≤ 1 (= 1 for pixels with
zero counts), so the weight of *every* stage must exceed 1: with a smaller
slope the objective is unbounded below for zero-count pixels and the
reconstruction diverges. The `"quadratic"` shape is bounded for any weight but
needs large weights (≈ 100) to enforce positivity well.

# Ramp (continuation)
The iterations are split evenly over `stages` runs of the optimizer with the
weights `weight * ramp^(s - stages)`, `s = 1 … stages` (default ≈ 2, 6.3, 20),
each started from the previous result. The optimizer (e.g. L-BFGS) restarts at
each stage, since its curvature memory is invalid after the objective changed.
`stages=1` uses the final weight from the start.

# Loss continuation
Negative object values can produce a negative forward model `μ`, where the
`Poisson()` and `Anscombe(b)` losses are undefined. With a positivity penalty
these two losses are therefore continued by their second order Taylor
polynomial below `μ = 0.01·mean(measured)` (resp. `μ + b` below that value),
which keeps the objective smooth and well defined. Other losses are used as given.

- `σ=0.1`: transition width in units of the (mean-normalized) data. The default `"huber"`
  shape has no influence on positive values; `"softplus"` also pushes values in `0 … few σ`
  upwards (gradient `-weight/2` at zero), so use it with a small `σ` (e.g. 0.01).
- `clip=true`: set remaining small negative values of the result to zero.
"""
struct Positivity
    weight::Float64
    σ::Float64
    mode::Symbol
    stages::Int
    ramp::Float64
    clip::Bool
end

function Positivity(; weight=20.0, σ=0.1, mode="huber", stages=3, ramp=sqrt(10.0), clip=true)
    m = Symbol(mode)
    m in (:softplus, :huber, :quadratic) ||
        throw(ArgumentError("Positivity mode must be \"softplus\", \"huber\" or \"quadratic\", got \"$mode\""))
    stages >= 1 || throw(ArgumentError("Positivity needs stages ≥ 1"))
    σ > 0 || throw(ArgumentError("Positivity needs σ > 0"))
    return Positivity(Float64(weight), Float64(σ), m, Int(stages), Float64(ramp), Bool(clip))
end

"""
    (p::Positivity)(x)

The un-weighted penalty `Σᵢ φ(xᵢ)`.
"""
(p::Positivity)(x) = positivity_penalty(x, eltype(x)(p.σ), p.mode)

"""
    stage_weights(p::Positivity)

Penalty weights of the continuation stages.
"""
stage_weights(p::Positivity) = [p.weight * p.ramp^(s - p.stages) for s in 1:p.stages]

# split `iterations` evenly over `n` stages (later stages get the remainder)
function stage_iterations(iterations, n)
    n = max(1, min(n, iterations))
    its = fill(iterations ÷ n, n)
    its[end-(iterations % n)+1:end] .+= 1
    return its
end

@inline function _pos_val(::Val{:softplus}, x, σ)
    z = -x / σ
    return σ * (max(z, zero(z)) + log1p(exp(-abs(z))))
end
@inline _pos_grad(::Val{:softplus}, x, σ) = -one(x) / (one(x) + exp(x / σ))
@inline _pos_val(::Val{:huber}, x, σ) = x >= 0 ? zero(x) : (x > -σ ? x * x / (2 * σ) : -x - σ / 2)
@inline _pos_grad(::Val{:huber}, x, σ) = x >= 0 ? zero(x) : (x > -σ ? x / σ : -one(x))
@inline _pos_val(::Val{:quadratic}, x, σ) = abs2(min(x, zero(x)))
@inline _pos_grad(::Val{:quadratic}, x, σ) = 2 * min(x, zero(x))

function positivity_penalty(x, σ, mode::Symbol)
    v = Val(mode)
    return sum(xi -> _pos_val(v, xi, σ), x)
end

function ChainRulesCore.rrule(::typeof(positivity_penalty), x, σ, mode::Symbol)
    y = positivity_penalty(x, σ, mode)
    function positivity_pullback(Δ)
        v = Val(mode)
        return NoTangent(), eltype(x)(unthunk(Δ)) .* (xi -> _pos_grad(v, xi, σ)).(x), NoTangent(), NoTangent()
    end
    return y, positivity_pullback
end

# ---------------------------------------------------------------------------
# smooth continuation of the Poisson / Anscombe losses below a threshold ε
# (second order Taylor polynomial at ε), used together with `Positivity`
# ---------------------------------------------------------------------------
struct PoissonContinued{T}
    ε::T
end
struct AnscombeContinued{T}
    b::T
    ε::T
end

@inline function _pc_val(μ, y, ε)
    μ > ε && return μ - y * log(μ)
    d = μ - ε
    return (ε - y * log(ε)) + (1 - y / ε) * d + y / (2 * ε^2) * d^2
end
@inline _pc_grad(μ, y, ε) = μ > ε ? 1 - y / μ : (1 - y / ε) + y / ε^2 * (μ - ε)

# f(μ) = (√(y+b) - √(μ+b))², continued below μ + b = ε
@inline function _ac_val(μ, y, b, ε)
    s = sqrt(max(y + b, zero(y)))
    u = μ + b
    u > ε && return abs2(s - sqrt(u))
    t = sqrt(ε); d = u - ε
    return abs2(s - t) + (1 - s / t) * d + s / (4 * t^3) * d^2
end
@inline function _ac_grad(μ, y, b, ε)
    s = sqrt(max(y + b, zero(y)))
    u = μ + b
    u > ε && return 1 - s / sqrt(u)
    t = sqrt(ε)
    return (1 - s / t) + s / (2 * t^3) * (u - ε)
end

function (l::PoissonContinued)(μ, meas, storage=similar(μ))
    ε = eltype(μ)(l.ε)
    storage .= _pc_val.(μ, meas, ε)
    return sum(storage)
end
function ChainRulesCore.rrule(l::PoissonContinued, μ, meas, storage)
    y = l(μ, meas, storage)
    ε = eltype(μ)(l.ε)
    pb(Δ) = (NoTangent(), eltype(μ)(unthunk(Δ)) .* _pc_grad.(μ, meas, ε), NoTangent(), NoTangent())
    return y, pb
end

function (l::AnscombeContinued)(μ, meas, storage=similar(μ))
    T = eltype(μ)
    storage .= _ac_val.(μ, meas, T(l.b), T(l.ε))
    return sum(storage)
end
function ChainRulesCore.rrule(l::AnscombeContinued, μ, meas, storage)
    y = l(μ, meas, storage)
    T = eltype(μ)
    pb(Δ) = (NoTangent(), T(unthunk(Δ)) .* _ac_grad.(μ, meas, T(l.b), T(l.ε)), NoTangent(), NoTangent())
    return y, pb
end

continued_loss(::typeof(poisson_aux), ε) = PoissonContinued(Float64(ε))
continued_loss(l::AnscombeLoss, ε) = AnscombeContinued(Float64(l.b), Float64(ε))
continued_loss(l, ε) = l
