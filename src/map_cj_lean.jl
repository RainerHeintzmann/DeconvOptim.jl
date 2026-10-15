export map_cj_lean

"""
    map_cj_lean(measured, psf; <keyword arguments>)

Memory-lean variant of [`map_cj`](@ref): maximum-a-posteriori deconvolution
with a pointwise positivity mapping `object = f(x)`, Fletcher--Reeves
conjugate gradients and a single analytic Newton step per iteration.

Instead of caching polynomial line-search coefficients, the step length is
`α = -φ'(0)/φ''(0)` with the exact first and second directional derivatives
at `α = 0`, followed by Armijo backtracking. The regularizer terms are
recomputed on the fly by fused stencil kernels instead of being stored. Apart
from the OTF and the FFT buffer, only six full-size arrays are used (data,
`x`, blurred object, gradient, search direction and one work array), plus a
mask when padding. Each iteration costs three convolutions (`H(f'(x) d)`,
`H f(x_new)` and the adjoint for the gradient) and one more per backtracking
step.

# Keyword arguments
- `penalty=:good`: `:good` or `:hessian_frobenius`. `:good` is Good's
  roughness `4 Σ_d weights[d] Σ (u_{n+e_d} - u_n)^2` of the amplitude `u`,
  where `u = x` for `mapping=:square` (MAPPR's `-4 xᵀLx`) and `u = sqrt(f)`
  otherwise. `:hessian_frobenius` is `Σ sqrt(epsilon^2 + ||Hessian(f)||_F^2)`
  with the same weighted stencils as `map_cj`.
- `loss=:poisson`: data term, `:poisson` or `:anscombe`
  (`Σ (sqrt(μ + β) - sqrt(measured + β))^2`).
- `anscombe_offset=3/8`: `β` of the `:anscombe` loss, in measured counts.
- `mapping=:square`: `:square` (`f = x^2`) or `:continuous_positive`
  (`f = x + sqrt(x^2 + 4 mapping_scale^2)`, strictly positive with `f' > 0`,
  so no voxel can get stuck, but `f' ≈ 2 f^2 / (4 mapping_scale^2)` for
  small `f`).
- `mapping_scale=1.0`: scale of `:continuous_positive`; `f(0) = 2 mapping_scale`.
- `gamma=0.0`: nonnegative regularization strength.
- `epsilon=nothing`: Frobenius smoothing; defaults to `sqrt(eps(T))`.
- `weights=nothing`: per-dimension weights as in `map_cj` (`1/S_d^2`).
- `background=0`: nonnegative background.
- `iterations=60`: number of conjugate-gradient iterations.
- `saddle_escape=1.0`, `escape_threshold=1e-4`: saddle escape for
  `mapping=:square`, see `map_cj`. Ignored for other mappings.
- `padding=0.0`, `initial=nothing`, `conv_dims`, `normalize_psf=true`,
  `callback=nothing`: as in `map_cj`.

The returned array is the reconstructed object `f(x)`.
"""
function map_cj_lean(measured::AbstractArray, psf::AbstractArray;
                     penalty=:good,
                     loss=:poisson,
                     anscombe_offset=3 / 8,
                     mapping=:square,
                     mapping_scale=1.0,
                     gamma=0.0,
                     epsilon=nothing,
                     weights=nothing,
                     background=0.0,
                     iterations=60,
                     saddle_escape=1.0,
                     escape_threshold=1e-4,
                     initial=nothing,
                     conv_dims=1:ndims(measured),
                     normalize_psf=true,
                     padding=0.0,
                     callback=nothing)
    iterations isa Integer && iterations >= 0 ||
        throw(ArgumentError("`iterations` must be a nonnegative integer"))
    isfinite(saddle_escape) && saddle_escape >= 0 ||
        throw(ArgumentError("`saddle_escape` must be nonnegative and finite"))
    isfinite(escape_threshold) && escape_threshold >= 0 ||
        throw(ArgumentError("`escape_threshold` must be nonnegative and finite"))
    c, x = _lean_setup(measured, psf; penalty, loss, anscombe_offset, mapping,
                       mapping_scale, gamma, epsilon, weights, background,
                       initial, conv_dims, normalize_psf, padding)
    T = eltype(x)
    m = c.mapping
    mu = similar(x)
    gradient = similar(x)
    direction = similar(x)
    tmp = similar(x)
    fill!(direction, zero(T))

    _lean_model!(c, mu, x)
    psi = _lean_objective(c, x, mu, tmp)
    gnorm2_prev = zero(T)
    alpha_prev = one(T)
    restart = true
    steepest = true
    escape = m isa _LeanSquare && saddle_escape > 0

    for iteration in 1:iterations
        _lean_gradient!(c, gradient, tmp, x, mu)
        gnorm2 = sum(abs2, gradient)
        beta = restart || iszero(gnorm2_prev) ? zero(T) : T(gnorm2 / gnorm2_prev)
        @. direction = beta * direction - gradient
        steepest = iszero(beta)
        if sum(_lazy(*, gradient, direction)) >= 0
            @. direction = -gradient
            steepest = true
        end
        gnorm2_prev = gnorm2
        restart = false
        escape && _lean_saddle_escape!(c, direction, x, tmp, T(saddle_escape),
                                       T(escape_threshold))

        dphi = sum(_lazy(*, gradient, direction))
        d2phi = _lean_curvature!(c, tmp, x, direction, mu)
        alpha = isfinite(d2phi) && d2phi > 0 ? T(-dphi / d2phi) : 2 * alpha_prev
        if !(isfinite(alpha) && alpha > 0)
            alpha = alpha_prev
        end

        # Armijo backtracking; every trial costs one convolution.
        @. x += alpha * direction
        _lean_model!(c, mu, x)
        psi_new = _lean_objective(c, x, mu, tmp)
        backtracks = 0
        while !(isfinite(psi_new) && psi_new <= psi + T(1e-4) * alpha * dphi) &&
              backtracks < 20
            alpha /= 2
            @. x -= alpha * direction
            _lean_model!(c, mu, x)
            psi_new = _lean_objective(c, x, mu, tmp)
            backtracks += 1
        end
        if isfinite(psi_new) && psi_new < psi
            psi = psi_new
            alpha_prev = alpha
        else
            # no decrease: undo the step and restart with steepest descent;
            # if steepest descent itself fails, the iteration has converged
            @. x -= alpha * direction
            _lean_model!(c, mu, x)
            steepest && break
            restart = true
        end

        if callback !== nothing
            @. tmp = _f(m, x)
            object = c.padded && c.crop ? center_extract(tmp, c.measured_size) : tmp
            model = c.padded && c.crop ? center_extract(mu, c.measured_size) : mu
            callback(iteration, object, model) === true && break
        end
    end

    @. tmp = _f(m, x)
    return c.padded && c.crop ? center_extract(tmp, c.measured_size) : tmp
end

# --- mappings ---------------------------------------------------------------
# f is the object, u the amplitude used by Good's roughness.

struct _LeanSquare end
struct _LeanContinuousPositive{T}
    c::T            # 4 * mapping_scale^2
end

Base.broadcastable(m::Union{_LeanSquare,_LeanContinuousPositive}) = Ref(m)

@inline _f(::_LeanSquare, x) = x * x
@inline _df(::_LeanSquare, x) = 2 * x
@inline _d2f(::_LeanSquare, x) = 2 * one(x)
@inline _u(::_LeanSquare, x) = x
@inline _du(::_LeanSquare, x) = one(x)
@inline _d2u(::_LeanSquare, x) = zero(x)
@inline _finv(::_LeanSquare, f) = sqrt(max(f, zero(f)))

@inline _f(m::_LeanContinuousPositive, x) = x + sqrt(x * x + m.c)
@inline _df(m::_LeanContinuousPositive, x) = 1 + x / sqrt(x * x + m.c)
@inline _d2f(m::_LeanContinuousPositive, x) = m.c / sqrt(x * x + m.c)^3
@inline _u(m::_LeanContinuousPositive, x) = sqrt(_f(m, x))
@inline _du(m::_LeanContinuousPositive, x) = _df(m, x) / (2 * _u(m, x))
@inline function _d2u(m::_LeanContinuousPositive, x)
    u = _u(m, x)
    return _d2f(m, x) / (2 * u) - _df(m, x)^2 / (4 * u^3)
end
@inline function _finv(m::_LeanContinuousPositive, f)
    f = max(f, sqrt(m.c) * eps(typeof(f)))
    return (f * f - m.c) / (2 * f)
end

# lazy broadcast for allocation-free reductions on CPU and GPU arrays
_lazy(f, args...) = Broadcast.instantiate(Broadcast.broadcasted(f, args...))

# --- setup and data term ----------------------------------------------------

function _lean_setup(measured, psf; penalty, loss, anscombe_offset, mapping,
                     mapping_scale, gamma, epsilon, weights, background,
                     initial, conv_dims, normalize_psf, padding)
    N = ndims(measured)
    eltype(measured) <: Real && eltype(psf) <: Real ||
        throw(ArgumentError("`measured` and `psf` must contain real-valued data"))
    N == ndims(psf) ||
        throw(ArgumentError("`measured` and `psf` must have the same number of dimensions"))
    all(size(psf, d) <= size(measured, d) for d in 1:N) ||
        throw(ArgumentError("`psf` dimensions must not exceed `measured` dimensions"))
    padding isa Real && isfinite(padding) ||
        throw(ArgumentError("`padding` must be finite"))
    penalty in (:good, :hessian_frobenius) ||
        throw(ArgumentError("`penalty` must be :good or :hessian_frobenius"))
    loss in (:poisson, :anscombe) ||
        throw(ArgumentError("`loss` must be :poisson or :anscombe"))
    mapping in (:square, :continuous_positive) ||
        throw(ArgumentError("`mapping` must be :square or :continuous_positive"))
    isfinite(mapping_scale) && mapping_scale > 0 ||
        throw(ArgumentError("`mapping_scale` must be positive and finite"))
    isfinite(anscombe_offset) && anscombe_offset > 0 ||
        throw(ArgumentError("`anscombe_offset` must be positive and finite"))
    isfinite(gamma) && gamma >= 0 ||
        throw(ArgumentError("`gamma` must be nonnegative and finite"))
    isfinite(background) && background >= 0 ||
        throw(ArgumentError("`background` must be nonnegative and finite"))

    T = promote_type(float(eltype(measured)), float(eltype(psf)))
    measured_data = T.(measured)
    any((.!isfinite.(measured_data)) .| (measured_data .< zero(T))) &&
        throw(ArgumentError("measurements must be nonnegative and finite"))
    psf_data = T.(psf)
    any(.!isfinite.(psf_data)) &&
        throw(ArgumentError("`psf` must contain only finite values"))
    conv_dims = Tuple(Int.(collect(conv_dims)))
    all(d -> 1 <= d <= N, conv_dims) ||
        throw(ArgumentError("`conv_dims` contains a dimension outside 1:$N"))
    if normalize_psf
        psf_sum = sum(psf_data)
        isfinite(psf_sum) && psf_sum > 0 ||
            throw(ArgumentError("`psf` must have a positive finite sum when normalized"))
        psf_data ./= psf_sum
    end

    size_padded = ntuple(N) do d
        if size(measured, d) == 1 || iszero(padding)
            size(measured, d)
        else
            size(measured, d) + next_fast_fft_size(max(4, 2 * round(Int,
                size(measured, d) * abs(padding))))
        end
    end
    padded = any(size_padded[d] != size(measured, d) for d in 1:N)
    data = similar(measured_data, T, size_padded)
    fill!(data, zero(T))
    mask = nothing
    if padded
        center_set!(data, measured_data)
        mask = similar(data)
        fill!(mask, zero(T))
        measured_mask = similar(measured_data)
        fill!(measured_mask, one(T))
        center_set!(mask, measured_mask)
    else
        copyto!(data, measured_data)
    end

    # OTF and an in-place convolution sharing one FFT buffer
    work = similar(data)
    fill!(work, zero(T))
    center_set!(work, fftshift(psf_data, conv_dims))
    work = ifftshift(work, conv_dims)
    P = get_plan(T)(work, conv_dims)
    ft = P * work
    otf = copy(ft)
    P_inv = inv(P)
    work = nothing
    conv! = (dst, src, adjoint) -> begin
        mul!(ft, P, src)
        if adjoint
            ft .*= conj.(otf)
        else
            ft .*= otf
        end
        mul!(dst, P_inv.p, ft)
        dst .*= P_inv.scale
        return dst
    end

    m = mapping === :square ? _LeanSquare() :
        _LeanContinuousPositive(T(4 * mapping_scale^2))
    eps_hessian = epsilon === nothing ? sqrt(eps(T)) : T(epsilon)
    isfinite(eps_hessian) && eps_hessian > 0 ||
        throw(ArgumentError("`epsilon` must be positive and finite"))
    dim_weights = weights === nothing ? ntuple(_ -> one(T), N) : Tuple(T.(weights))
    length(dim_weights) == N && all(w -> isfinite(w) && w >= 0, dim_weights) ||
        throw(ArgumentError("`weights` must contain one nonnegative finite value per dimension"))
    pair_list, pair_weights = _mappr_hessian_pairs(N)

    x = similar(data)
    mean_level = max(mean(measured_data) - T(background), eps(T))
    if initial === nothing
        fill!(x, _finv(m, mean_level))
    elseif initial isa Number
        isfinite(initial) && initial >= 0 ||
            throw(ArgumentError("`initial` must be nonnegative and finite"))
        fill!(x, _finv(m, T(initial)))
    else
        size(initial) == size(measured_data) || size(initial) == size(data) ||
            throw(ArgumentError("array-valued `initial` must have the measured or padded size"))
        any((.!isfinite.(initial)) .| (initial .< 0)) &&
            throw(ArgumentError("`initial` must be nonnegative and finite"))
        initial_data = similar(data, T, size(initial))
        copyto!(initial_data, initial)
        @. initial_data = _finv(m, initial_data)
        if size(initial_data) == size(data)
            copyto!(x, initial_data)
        else
            fill!(x, _finv(m, mean_level))
            center_set!(x, initial_data)
        end
    end

    c = (data=data, mask=mask, background=T(background), loss=loss,
         offset=T(anscombe_offset), mapping=m, penalty=penalty, gamma=T(gamma),
         eps2=eps_hessian^2, weights=dim_weights, pairs=Tuple(pair_list),
         pair_weights=Tuple(T.(pair_weights)), conv! = conv!, padded=padded,
         crop=padding > 0, measured_size=size(measured_data))
    return c, x
end

_lean_mask(c) = c.mask === nothing ? one(eltype(c.data)) : c.mask

function _lean_model!(c, mu, x)
    m = c.mapping
    @. mu = _f(m, x)
    return c.conv!(mu, mu, false)
end

# per-voxel data loss, its μ-derivative (the residual) and second derivative
@inline function _lean_loss(::Val{:poisson}, mu, g, b, offset)
    mt = max(mu + b, eps(typeof(mu)))
    return mt - g * log(mt)
end
@inline _lean_residual(::Val{:poisson}, mu, g, b, offset) =
    1 - g / max(mu + b, eps(typeof(mu)))
@inline _lean_loss2(::Val{:poisson}, mu, g, b, offset) =
    g / max(mu + b, eps(typeof(mu)))^2

@inline function _lean_loss(::Val{:anscombe}, mu, g, b, offset)
    return abs2(sqrt(max(mu + b, zero(mu)) + offset) - sqrt(g + offset))
end
@inline _lean_residual(::Val{:anscombe}, mu, g, b, offset) =
    1 - sqrt(g + offset) / sqrt(max(mu + b, zero(mu)) + offset)
@inline function _lean_loss2(::Val{:anscombe}, mu, g, b, offset)
    M = max(mu + b, zero(mu)) + offset
    return sqrt(g + offset) / (2 * M * sqrt(M))
end

function _lean_objective(c, x, mu, tmp)
    L = Val(c.loss)
    b, o = c.background, c.offset
    data_term = sum(_lazy((mk, u, g) -> mk * _lean_loss(L, u, g, b, o),
                          _lean_mask(c), mu, c.data))
    iszero(c.gamma) && return data_term
    _lean_penalty_value!(c, tmp, x)
    return data_term + c.gamma * sum(tmp)
end

# gradient = ∇_x Ψ. On return `tmp` holds the object-space gradient g_f,
# including γ ∇_f R for the Hessian penalty (data part only for Good's).
function _lean_gradient!(c, gradient, tmp, x, mu)
    L = Val(c.loss)
    m = c.mapping
    b, o, mk, g = c.background, c.offset, _lean_mask(c), c.data
    @. tmp = mk * _lean_residual(L, mu, g, b, o)
    c.conv!(tmp, tmp, true)
    if c.penalty === :hessian_frobenius && !iszero(c.gamma)
        # `gradient` is free here: it holds r = sqrt(ε² + Σ w h²) for the adjoint
        _lean_penalty_value!(c, gradient, x)
        _lean_launch!(_lean_hf_grad_kernel!, tmp, x, gradient, m, c.pairs,
                      c.pair_weights, c.weights, c.gamma)
    end
    @. gradient = _df(m, x) * tmp
    if c.penalty === :good && !iszero(c.gamma)
        _lean_launch!(_lean_good_grad_kernel!, gradient, x, m, c.weights, c.gamma)
    end
    return gradient
end

# φ''(0) along `d`, given `tmp` = g_f from `_lean_gradient!`. Uses
#   φ''(0) = <g_f, f''(x) d²> + Σ ℓ''(μ) (H(f'(x) d))² + regularizer terms,
# where <g_f, f'' d²> replaces the convolution H(f'' d²). Overwrites `tmp`.
function _lean_curvature!(c, tmp, x, d, mu)
    L = Val(c.loss)
    m = c.mapping
    curvature = sum(_lazy((g, xi, di) -> g * _d2f(m, xi) * di * di, tmp, x, d))
    @. tmp = _df(m, x) * d
    c.conv!(tmp, tmp, false)
    b, o = c.background, c.offset
    curvature += sum(_lazy((mk, h, u, g) -> mk * _lean_loss2(L, u, g, b, o) * h * h,
                           _lean_mask(c), tmp, mu, c.data))
    if !iszero(c.gamma)
        if c.penalty === :hessian_frobenius
            _lean_launch!(_lean_hf_curv_kernel!, tmp, x, d, m, c.pairs,
                          c.pair_weights, c.weights, c.eps2)
        else
            _lean_launch!(_lean_good_curv_kernel!, tmp, x, d, m, c.weights)
        end
        curvature += c.gamma * sum(tmp)
    end
    return curvature
end

function _lean_penalty_value!(c, out, x)
    if c.penalty === :hessian_frobenius
        _lean_launch!(_lean_hf_value_kernel!, out, x, c.mapping, c.pairs,
                      c.pair_weights, c.weights, c.eps2)
    else
        _lean_launch!(_lean_good_value_kernel!, out, x, c.mapping, c.weights)
    end
    return out
end

# Saddle escape for f = x^2, as in `map_cj`; `g_f` is the object-space
# gradient (for Good's penalty its curvature at x = 0 is added).
function _lean_saddle_escape!(c, direction, x, g_f, strength, threshold)
    T = eltype(direction)
    m = c.mapping
    shift = c.penalty === :good ? 8 * c.gamma * sum(c.weights) : zero(T)
    level = threshold * maximum(_lazy(xi -> _f(m, xi), x))
    pull = (xi, g) -> ifelse((_f(m, xi) <= level) & (g + shift < 0), -(g + shift),
                             zero(level))
    max_pull = maximum(_lazy(pull, x, g_f))
    max_pull > 0 || return direction
    rms = sqrt(sum(abs2, direction) / length(direction))
    rms > 0 || return direction
    scale = strength * rms
    @. direction += ifelse(x < 0, -scale, scale) * sqrt(pull(x, g_f) / max_pull)
    return direction
end

function _lean_launch!(kernel, out, args...)
    backend = KernelAbstractions.get_backend(out)
    kernel(backend)(out, args...; ndrange=size(out))
    return out
end

# --- fused stencil kernels --------------------------------------------------

@inline _lean_step(I::CartesianIndex{N}, a, s) where {N} =
    CartesianIndex(ntuple(k -> k == a ? I[k] + s : I[k], Val(N)))

# field value at J: Val(1) -> f(x), Val(2) -> f'(x) d
@inline _lean_field(m, ::Val{1}, x, d, J) = @inbounds _f(m, x[J])
@inline _lean_field(m, ::Val{2}, x, d, J) = @inbounds _df(m, x[J]) * d[J]

# weighted Hessian entry (a, b) of the field at I, matching
# `_mappr_hessian_kernel!`
@inline function _lean_hessian_entry(m, F, x, d, I, a, b, weights)
    sz = size(x)
    if a == b
        ip = _lean_step(I, a, I[a] < sz[a] ? 1 : 0)
        im = _lean_step(I, a, I[a] > 1 ? -1 : 0)
        return weights[a] * (_lean_field(m, F, x, d, ip) -
                             2 * _lean_field(m, F, x, d, I) +
                             _lean_field(m, F, x, d, im))
    elseif I[a] < sz[a] && I[b] < sz[b]
        ia = _lean_step(I, a, 1)
        ib = _lean_step(I, b, 1)
        iab = _lean_step(ia, b, 1)
        return sqrt(weights[a] * weights[b]) * (_lean_field(m, F, x, d, iab) -
            _lean_field(m, F, x, d, ia) - _lean_field(m, F, x, d, ib) +
            _lean_field(m, F, x, d, I))
    else
        return zero(eltype(x))
    end
end

@inline _lean_hessian(m, F, x, d, I, pairs, weights) =
    ntuple(k -> _lean_hessian_entry(m, F, x, d, I, pairs[k]..., weights),
           Val(length(pairs)))

# q_k(J) = w_k h_k(J) / r(J), only the k-th entry, with r precomputed
@inline _lean_hf_q(m, x, r, J, a, b, pw, weights) =
    pw * _lean_hessian_entry(m, Val(1), x, x, J, a, b, weights) / @inbounds(r[J])

@kernel function _lean_hf_value_kernel!(out, x, m, pairs, pair_weights, weights, eps2)
    I = @index(Global, Cartesian)
    h = _lean_hessian(m, Val(1), x, x, I, pairs, weights)
    s = eps2
    for k in 1:length(h)
        s += pair_weights[k] * h[k]^2
    end
    @inbounds out[I] = sqrt(s)
end

# out += gamma * Σ_k D_kᵀ (w_k h_k / r), the adjoint pattern of
# `_mappr_hessian_adjoint_kernel!`, with r = sqrt(ε² + Σ w h²) precomputed and
# only the k-th Hessian entry recomputed at each neighbour
@kernel function _lean_hf_grad_kernel!(out, x, r, m, pairs, pair_weights, weights,
                                       gamma)
    I = @index(Global, Cartesian)
    sz = size(x)
    T = eltype(out)
    v = zero(T)
    for k in 1:length(pairs)
        a, b = pairs[k]
        pw = pair_weights[k]
        if a == b
            n = sz[a]
            if n > 1
                sc = weights[a]
                v += (I[a] == 1 || I[a] == n ? -one(T) : -2 * one(T)) *
                     _lean_hf_q(m, x, r, I, a, b, pw, weights) * sc
                if I[a] > 1
                    v += _lean_hf_q(m, x, r, _lean_step(I, a, -1), a, b, pw, weights) * sc
                end
                if I[a] < n
                    v += _lean_hf_q(m, x, r, _lean_step(I, a, 1), a, b, pw, weights) * sc
                end
            end
        else
            sc = sqrt(weights[a] * weights[b])
            if I[a] < sz[a] && I[b] < sz[b]
                v += _lean_hf_q(m, x, r, I, a, b, pw, weights) * sc
            end
            if I[a] > 1 && I[b] < sz[b]
                v -= _lean_hf_q(m, x, r, _lean_step(I, a, -1), a, b, pw, weights) * sc
            end
            if I[a] < sz[a] && I[b] > 1
                v -= _lean_hf_q(m, x, r, _lean_step(I, b, -1), a, b, pw, weights) * sc
            end
            if I[a] > 1 && I[b] > 1
                v += _lean_hf_q(m, x, r, _lean_step(_lean_step(I, a, -1), b, -1),
                                a, b, pw, weights) * sc
            end
        end
    end
    @inbounds out[I] += gamma * v
end

# Regularizer part of φ''(0) without the h·h'' term, which enters through
# <g_f, f'' d²>:  Σ_k w h'_k² / r - (Σ_k w h_k h'_k)² / r³,  h' = D_k(f' d)
@kernel function _lean_hf_curv_kernel!(out, x, d, m, pairs, pair_weights,
                                       weights, eps2)
    I = @index(Global, Cartesian)
    h = _lean_hessian(m, Val(1), x, d, I, pairs, weights)
    hp = _lean_hessian(m, Val(2), x, d, I, pairs, weights)
    s = eps2
    cross = zero(eltype(out))
    slope = zero(eltype(out))
    for k in 1:length(h)
        s += pair_weights[k] * h[k]^2
        cross += pair_weights[k] * h[k] * hp[k]
        slope += pair_weights[k] * hp[k]^2
    end
    r = sqrt(s)
    @inbounds out[I] = slope / r - cross^2 / (s * r)
end

# weighted Neumann Laplacian of u at I
@inline function _lean_laplacian_u(m, x, I, weights)
    sz = size(x)
    uI = @inbounds _u(m, x[I])
    v = zero(uI)
    for a in 1:length(weights)
        if I[a] < sz[a]
            v += weights[a] * (@inbounds(_u(m, x[_lean_step(I, a, 1)])) - uI)
        end
        if I[a] > 1
            v += weights[a] * (@inbounds(_u(m, x[_lean_step(I, a, -1)])) - uI)
        end
    end
    return v
end

@kernel function _lean_good_value_kernel!(out, x, m, weights)
    I = @index(Global, Cartesian)
    sz = size(x)
    uI = @inbounds _u(m, x[I])
    v = zero(eltype(out))
    for a in 1:length(weights)
        if I[a] < sz[a]
            v += weights[a] * (@inbounds(_u(m, x[_lean_step(I, a, 1)])) - uI)^2
        end
    end
    @inbounds out[I] = 4 * v
end

# gradient += gamma * ∂/∂x 4 Σ w (Δu)² = -8 gamma u'(x) (L_w u)
@kernel function _lean_good_grad_kernel!(out, x, m, weights, gamma)
    I = @index(Global, Cartesian)
    lap = _lean_laplacian_u(m, x, I, weights)
    @inbounds out[I] -= 8 * gamma * _du(m, x[I]) * lap
end

# 8 Σ w (Δ(u' d))² - 8 (L_w u) u'' d²
@kernel function _lean_good_curv_kernel!(out, x, d, m, weights)
    I = @index(Global, Cartesian)
    sz = size(x)
    vI = @inbounds _du(m, x[I]) * d[I]
    v = zero(eltype(out))
    for a in 1:length(weights)
        if I[a] < sz[a]
            J = _lean_step(I, a, 1)
            v += weights[a] * (@inbounds(_du(m, x[J]) * d[J]) - vI)^2
        end
    end
    lap = _lean_laplacian_u(m, x, I, weights)
    @inbounds out[I] = 8 * v - 8 * lap * _d2u(m, x[I]) * d[I]^2
end
