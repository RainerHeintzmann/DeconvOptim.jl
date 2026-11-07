# ADMM (alternating direction method of multipliers) solver for the common
# deconvolution problems
#
#     min_x  L(C(h ∗ x) + b, y) + λ R(x)      s.t. x ≥ 0
#
# without automatic differentiation. Variable splitting following
#
# * M. Figueiredo, J. Bioucas-Dias, "Restoration of Poissonian images using
#   alternating direction optimization", IEEE TIP 19(12), 3133 (2010)  [PIDAL]
# * M. Almeida, M. Figueiredo, "Deconvolving images with unknown boundaries
#   using the alternating direction method of multipliers", IEEE TIP 22(8),
#   3074 (2013)  [cropping / unknown boundaries]
# * S. Lefkimmiatis, M. Unser, "Poisson image reconstruction with Hessian
#   Schatten-norm regularization", IEEE TIP 22(11), 4314 (2013)  [HS prox]
#
# Splits:  z₁ = h ∗ x        (data term, separable prox; pixels outside the
#                             measured region (padding) are unconstrained,
#                             which is the Almeida-Figueiredo trick: the
#                             x-update stays diagonal in Fourier space)
#          z₂ₖ = Kₖ x        (regularizer, e.g. ∇x for TV; group shrinkage)
#          z₃ = x            (projection onto x ≥ 0)
# Quadratic regularizers (Tikhonov) need no split, they enter the x-update.
# The x-update solves (HᵀH + Σ KₖᵀKₖ + I + 2λ/ρ QᵀQ) x = rhs by one division
# in Fourier space (periodic boundaries of the reconstruction grid).

export ADMM

"""
    ADMM(; ρ=nothing, adaptive_ρ=true, iterations=100, tol=0, newton_steps=6,
           check_every=10, ρ_balance=10, ρ_factor=2)

Select the ADMM solver in [`deconvolution`](@ref) via `opt=ADMM()`.
No automatic differentiation is used. ADMM minimizes the exact (non-smoothed)
objective and enforces non-negativity by projection instead of a mapping.

Supported combinations:
- `loss`: `Poisson()`, `Gauss()`, `Anscombe(b)`
- `regularizer`: `nothing`, `TV()`, `TH()`, `HS()` (only `p=1`), `Tikhonov()` (all modes)
- `mapping`: any non-`nothing` mapping is interpreted as `x ≥ 0`, `nothing` as unconstrained.

Boundaries of the reconstruction grid are periodic. Use `padding>0` in
`deconvolution` to avoid wrap-around: the padded border is then not constrained
by the data (Almeida & Figueiredo 2013).

# Arguments
- `ρ=nothing`: initial penalty parameter. `nothing` selects a loss-dependent default.
- `adaptive_ρ=true`: adapt `ρ` by residual balancing (Boyd et al. 2011, §3.4.1).
- `iterations=100`: maximum number of iterations (overridden by the `iterations`
   keyword of `deconvolution`, if given).
- `tol=0`: stop when relative primal and dual residuals are both below `tol`.
   `0` always runs all iterations.
- `newton_steps=6`: number of per-pixel Newton steps for the `Anscombe` prox.
- `check_every=10`: residuals (and `ρ`) are evaluated every `check_every` iterations.
- `progress_every=1`: `progress`/`debug_f` are called every `progress_every`
   iterations (each report costs an extra forward model evaluation).
- `callback=nothing`: `callback(iteration, x)` is called after every iteration with
   the current (non-negative) iterate; returning `true` stops the solver.
"""
Base.@kwdef struct ADMM
    ρ::Union{Nothing, Float64} = nothing
    adaptive_ρ::Bool = true
    iterations::Int = 100
    tol::Float64 = 0.0
    newton_steps::Int = 6
    check_every::Int = 10
    progress_every::Int = 1
    callback::Any = nothing
    ρ_balance::Float64 = 10.0
    ρ_factor::Float64 = 2.0
end

"""
    ADMMResult

Second return value of `deconvolution(...; opt=ADMM())`.
Fields: `minimizer` (full, padded reconstruction in normalized units),
`iterations`, `converged`, `ρ` (final penalty) and `history`
(a vector of `(iteration, r_rel, s_rel, ρ)` named tuples).
"""
struct ADMMResult{A}
    minimizer::A
    iterations::Int
    converged::Bool
    ρ::Float64
    history::Vector{NamedTuple{(:iteration, :r_rel, :s_rel, :ρ), Tuple{Int, Float64, Float64, Float64}}}
end

# ---------------------------------------------------------------------------
# periodic stencils:  (S x)[i] = Σⱼ cⱼ x[i + oⱼ]   (indices wrap around)
# ---------------------------------------------------------------------------
struct Stencil{N, T}
    offsets::Vector{NTuple{N, Int}}
    coeffs::Vector{T}
end

Base.adjoint(s::Stencil) = Stencil([map(-, o) for o in s.offsets], s.coeffs)

_unit(N, d, s) = ntuple(i -> i == d ? s : 0, N)

# out[i] = f(out[i], xs[1][i + o], xs[2][i + o], …), periodic; the shift is split
# into contiguous blocks, so only broadcasts over views are used (device-generic)
function _shift_apply!(f, out, o, xs...)
    parts = ntuple(ndims(out)) do d
        n = size(out, d)
        s = mod(o[d], n)
        s == 0 ? ((1:n, 1:n),) : ((1:n-s, 1+s:n), (n-s+1:n, 1:s))
    end
    for p in Iterators.product(parts...)
        dst = view(out, map(first, p)...)
        _block_op!(f, dst, map(x -> view(x, map(last, p)...), xs)...)
    end
    return out
end

# CPU: columns along dimension 1 are contiguous; the shift along dimension 1
# splits each column into two SIMD loops, the other dimensions wrap per column
function _shift_apply!(f, out::Array{<:Any, N}, o, xs::Array...) where N
    sz = size(out)
    n1 = sz[1]
    s1 = mod(o[1], n1)
    cols = CartesianIndices(ntuple(d -> 1:sz[d+1], N - 1))
    lcols = LinearIndices(cols)
    ncol = length(cols)
    body = (lo, hi) -> begin
        for c in lo:hi
            J = cols[c]
            Js = CartesianIndex(ntuple(d -> mod1(J[d] + o[d+1], sz[d+1]), N - 1))
            db = (c - 1) * n1
            sb = (lcols[Js] - 1) * n1
            @inbounds @simd for i in 1:n1-s1
                out[db+i] = f(out[db+i], map(x -> x[sb+i+s1], xs)...)
            end
            @inbounds @simd for i in n1-s1+1:n1
                out[db+i] = f(out[db+i], map(x -> x[sb+i+s1-n1], xs)...)
            end
        end
    end
    if Threads.nthreads() == 1 || length(out) < 1 << 15 || ncol < 2
        body(1, ncol)
    else
        nt = min(Threads.nthreads(), ncol)
        Threads.@threads :static for t in 1:nt
            body(div((t - 1) * ncol, nt) + 1, div(t * ncol, nt))
        end
    end
    return out
end

_block_op!(f, dst, srcs...) = (dst .= f.(dst, srcs...))
_shifted_add!(out, x, o, c) = _shift_apply!((d, v) -> d + c * v, out, o, x)

function apply!(out, s::Stencil, x)
    fill!(out, zero(eltype(out)))
    return apply_add!(out, s, x)
end

# out .+= S x
function apply_add!(out, s::Stencil, x)
    for (o, c) in zip(s.offsets, s.coeffs)
        _shifted_add!(out, x, o, c)
    end
    return out
end

# out .+= S (a - b)   (fused, no temporary)
function apply_add_diff!(out, s::Stencil, a, b)
    for (o, c) in zip(s.offsets, s.coeffs)
        _shift_apply!((d, u, v) -> d + c * (u - v), out, o, a, b)
    end
    return out
end

# elementwise dst[i] = f(srcs[1][i], …): multithreaded on CPU arrays (Julia
# broadcasts are single-threaded, ADMM is memory-bound), plain broadcast otherwise
function _tb!(f, dst::Array, srcs::Array...)
    L = length(dst)
    nt = Threads.nthreads()
    if nt == 1 || L < 1 << 15
        @inbounds @simd for i in eachindex(dst)
            dst[i] = f(map(a -> a[i], srcs)...)
        end
        return dst
    end
    Threads.@threads :static for c in 1:nt
        lo = div((c - 1) * L, nt) + 1
        hi = div(c * L, nt)
        @inbounds @simd for i in lo:hi
            dst[i] = f(map(a -> a[i], srcs)...)
        end
    end
    return dst
end
_tb!(f, dst, srcs...) = (dst .= f.(srcs...); dst)

# |FFT symbol|² of a stencil on the rfft grid, computed on the CPU
function _abs2_symbol(s::Stencil, sz, ::Type{T}) where T
    δ = zeros(T, sz); δ[1] = one(T)
    k = apply!(similar(δ), s, δ)
    return abs2.(rfft(k))
end

# ---------------------------------------------------------------------------
# translation of the regularizers into stencils
# ---------------------------------------------------------------------------
# kind = :group    → prox of λ Σᵢ ‖(z₁ᵢ, …, zₖᵢ)‖₂ (isotropic shrinkage; TV, TH)
# kind = :schatten → prox of λ Σᵢ ‖[[a, c/√2], [c/√2, d]]ᵢ‖_S1, components (a, d, c)
struct SplitBlock{S}
    kind::Symbol
    stencils::Vector{S}
end

_sum_dims(sum_dims, N) = isnothing(sum_dims) ? collect(1:N) : [d for d in sum_dims if d <= N]
_weights(weights, n, T) = isnothing(weights) ? ones(T, n) : T.(collect(weights))

_diff2(N, d, w, T) = Stencil([_unit(N, d, 1), _unit(N, d, -1), _unit(N, d, 0)], T[w, w, -2w])

function admm_reg(reg::Nothing, N, T)
    return SplitBlock{Stencil{N, T}}[], Stencil{N, T}[]
end

function admm_reg(reg::TaggedRegularizer, N, T)
    p = reg.params
    nd = isnothing(p.num_dims) ? N : min(p.num_dims, N)
    sd = _sum_dims(p.sum_dims, nd)
    w = _weights(p.weights, length(sd), T)
    S = Stencil{N, T}
    if reg.kind == :TV
        st = p.step
        o2 = p.mode == "central" ? -st : 0
        sts = S[Stencil([_unit(N, d, st), _unit(N, d, o2)], T[sqrt(wk), -sqrt(wk)]) for (d, wk) in zip(sd, w)]
        return [SplitBlock(:group, sts)], S[]
    elseif reg.kind == :TH
        sts = S[_diff2(N, d, wk, T) for (d, wk) in zip(sd, w)]
        for k in eachindex(sd), l in k+1:length(sd)
            d, e = sd[k], sd[l]
            c = sqrt(2 * w[k] * w[l])
            push!(sts, Stencil([_unit(N, d, 1) .+ _unit(N, e, 1), _unit(N, d, 1), _unit(N, e, 1), _unit(N, d, 0)],
                               T[c, -c, -c, c]))
        end
        return [SplitBlock(:group, sts)], S[]
    elseif reg.kind == :Tikhonov
        if p.mode == "identity"
            return SplitBlock{S}[], S[Stencil([_unit(N, 1, 0)], T[1])]
        elseif p.mode == "laplace"
            offs = NTuple{N, Int}[]; cs = T[]
            for (d, wk) in zip(sd, w)
                push!(offs, _unit(N, d, 1), _unit(N, d, -1)); push!(cs, wk, wk)
            end
            push!(offs, _unit(N, 1, 0)); push!(cs, -2 * sum(w))
            return SplitBlock{S}[], S[Stencil(offs, cs)]
        else # spatial_grad_square: Σ w (x[i+1] - x[i-1])²
            return SplitBlock{S}[], S[Stencil([_unit(N, d, 1), _unit(N, d, -1)], T[sqrt(wk), -sqrt(wk)]) for (d, wk) in zip(sd, w)]
        end
    end
    throw(ArgumentError("regularizer $(reg.kind) is not supported by ADMM"))
end

function admm_reg(reg::HessianSchattenNorm, N, T)
    isone(reg.p) || throw(ArgumentError("ADMM supports the Hessian Schatten norm only for p=1"))
    sd = isnothing(reg.sum_dims) ? [1, 2] : collect(reg.sum_dims)
    length(sd) == 2 || throw(ArgumentError("HS needs exactly two sum_dims"))
    w = _weights(reg.weights, 2, T)
    d1, d2 = sd
    # mixed central difference as in `Δr1r2`, scaled by √2·√(w₁w₂) (Frobenius ↔ Euclidean)
    c = T(0.25) * sqrt(2 * w[1] * w[2])
    mixed = Stencil([_unit(N, d1, 1) .+ _unit(N, d2, 1), _unit(N, d1, -1) .+ _unit(N, d2, 1),
                     _unit(N, d1, 1) .+ _unit(N, d2, -1), _unit(N, d1, -1) .+ _unit(N, d2, -1)],
                    T[c, -c, -c, c])
    sts = [_diff2(N, d1, w[1], T), _diff2(N, d2, w[2], T), mixed]
    return [SplitBlock(:schatten, sts)], Stencil{N, T}[]
end

admm_reg(reg, N, T) = throw(ArgumentError(
    "This regularizer is not supported by ADMM. Use `nothing`, `TV()`, `TH()`, `HS()` or `Tikhonov()`."))

# ---------------------------------------------------------------------------
# proximal operators of the data terms:  argmin_u  f(u + bg) + ρ/2 (u - v)²
# ---------------------------------------------------------------------------
abstract type ADMMLoss end
struct PoissonProx <: ADMMLoss end
struct GaussProx <: ADMMLoss end
struct AnscombeProx{T} <: ADMMLoss
    b::T
end

admm_loss(::typeof(poisson_aux), T) = PoissonProx()
admm_loss(::typeof(gauss_aux), T) = GaussProx()
admm_loss(l::AnscombeLoss, T) = AnscombeProx(T(l.b))
admm_loss(l, T) = throw(ArgumentError("This loss is not supported by ADMM. Use `Poisson()`, `Gauss()` or `Anscombe(b)`."))

# default initial penalties, tuned on mean-normalized data (best ρ was nearly
# independent of λ); Poisson/Anscombe curvature scales with 1/mean(y)
default_ρ(::PoissonProx, ymean) = 0.1 / ymean
default_ρ(::AnscombeProx, ymean) = 0.04 / ymean
default_ρ(::GaussProx, ymean) = 0.05

# f(μ) = μ - y log μ
@inline function prox(::PoissonProx, v, y, ρ, bg)
    a = ρ * (v + bg) - one(v)
    return (a + sqrt(a * a + 4 * ρ * max(y, zero(y)))) / (2 * ρ) - bg
end

# f(μ) = (μ - y)²
@inline prox(::GaussProx, v, y, ρ, bg) = (2 * y + ρ * (v + bg)) / (2 + ρ) - bg

# f(μ) = (√(y + b) - √(μ + b))². With t = √(μ + b) the optimality condition is the
# cubic  p(t) = ρt³ + (1 - ρ(v + bg + b)) t - √(y + b) = 0  with a unique root t ≥ 0.
# p is convex for t > 0 and p(t₀) ≥ 0 for t₀ = max(√(y+b), √(v+bg+b)), so Newton
# started at t₀ decreases monotonically to the root. Only scalar (per-pixel)
# derivatives are used; no Hessian matrix is formed.
@inline function prox(l::AnscombeProx, v, y, ρ, bg, nsteps)
    b = l.b
    vb = v + bg + b
    s = sqrt(max(y + b, zero(y)))
    c = one(v) - ρ * vb
    t = max(s, sqrt(max(vb, zero(vb))))
    for _ in 1:nsteps
        p = (ρ * t * t + c) * t - s
        dp = 3 * ρ * t * t + c
        t = max(t - p / max(dp, floatmin(dp)), zero(t))
    end
    return t * t - b - bg
end

function prox_data!(z, l::ADMMLoss, v, y, ρ, bg, mask, nsteps)
    ρT = eltype(z)(ρ)
    if l isa AnscombeProx
        _prox_apply!((vv, yy) -> prox(l, vv, yy, ρT, bg, nsteps), z, v, y, mask)
    else
        _prox_apply!((vv, yy) -> prox(l, vv, yy, ρT, bg), z, v, y, mask)
    end
    return z
end

_prox_apply!(pr, z, v, y, ::Nothing) = _tb!(pr, z, v, y)
_prox_apply!(pr, z, v, y, mask) = _tb!((vv, yy, mm) -> mm > 0 ? pr(vv, yy) : vv, z, v, y, mask)

# ---------------------------------------------------------------------------
# proximal operators of the split regularizer blocks (in place, v → z)
# ---------------------------------------------------------------------------
function prox_block!(zs, kind, vs, κ, s1, s2)
    T = eltype(s1)
    if kind == :group
        # shrinkage factor max(1 - κ/‖v‖, 0) in one pass over all components
        o, z0, tiny = one(T), zero(T), floatmin(T)   # values, not the type T (GPU kernels need isbits closures)
        _tb!((v...) -> max(o - κ / sqrt(sum(abs2, v) + tiny), z0), s1, vs...)
        for k in eachindex(vs)
            _tb!(*, zs[k], s1, vs[k])
        end
    else # :schatten, components (a, d, c) with c = √2·b, eigenvalue soft-thresholding
        a, d, c = vs
        # s1: half difference of the eigenvalues r → ratio r'/r,  s2: new mean m'
        _tb!((a, d, c) -> sqrt(abs2((a - d) / 2) + abs2(c) / 2), s1, a, d, c)
        _tb!((a, d, r) -> _hs_newmean((a + d) / 2, r, κ), s2, a, d, s1)
        _tb!((a, d, r) -> _hs_newratio((a + d) / 2, r, κ), s1, a, d, s1)
        _tb!((m, f, a, d) -> m + f * (a - d) / 2, zs[1], s2, s1, a, d)
        _tb!((m, f, a, d) -> m - f * (a - d) / 2, zs[2], s2, s1, a, d)
        _tb!(*, zs[3], s1, c)
    end
    return zs
end

@inline _soft(x, κ) = sign(x) * max(abs(x) - κ, zero(x))
@inline function _hs_newmean(m, r, κ)
    return (_soft(m + r, κ) + _soft(m - r, κ)) / 2
end
@inline function _hs_newratio(m, r, κ)
    rn = (_soft(m + r, κ) - _soft(m - r, κ)) / 2
    return r > floatmin(r) ? rn / r : zero(r)
end

# ---------------------------------------------------------------------------
# fused elementwise prox update (CPU): one pass reads v (held in w), writes
# z = prox(v) and w = v - z for all components of a block
# ---------------------------------------------------------------------------
@inline function _pix_prox(::Val{:group}, v::NTuple{K, T}, κ) where {K, T}
    f = max(one(T) - κ / sqrt(sum(abs2, v) + floatmin(T)), zero(T))
    return map(vi -> f * vi, v)
end
@inline function _pix_prox(::Val{:schatten}, v::NTuple{3, T}, κ) where T
    a, d, c = v
    r = sqrt(abs2((a - d) / 2) + abs2(c) / 2)
    m = (a + d) / 2
    mn = _hs_newmean(m, r, κ)
    f = _hs_newratio(m, r, κ)
    return (mn + f * (a - d) / 2, mn - f * (a - d) / 2, f * c)
end

# threaded loop over static chunks of 1:L, f(lo, hi)
function _tchunks(f, L)
    nt = Threads.nthreads()
    if nt == 1 || L < 1 << 15
        f(1, L)
    else
        Threads.@threads :static for c in 1:nt
            f(div((c - 1) * L, nt) + 1, div(c * L, nt))
        end
    end
    return nothing
end

function _prox_update!(::Val{K}, kind::Val, zs, ws, κ) where K
    zt = ntuple(k -> zs[k], Val(K))
    wt = ntuple(k -> ws[k], Val(K))
    _tchunks(length(wt[1])) do lo, hi
        @inbounds @simd for i in lo:hi
            v = map(w -> w[i], wt)
            z = _pix_prox(kind, v, κ)
            map((za, zi) -> (za[i] = zi), zt, z)
            map((wa, vi, zi) -> (wa[i] = vi - zi), wt, v, z)
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# FFT helpers
# ---------------------------------------------------------------------------
_plan_rfft(x::Array) = plan_rfft(similar(x); flags=FFTW.MEASURE)
_plan_rfft(x) = plan_rfft(x)
_plan_irfft(X::Array, n) = plan_irfft(similar(X), n; flags=FFTW.MEASURE)
_plan_irfft(X, n) = plan_irfft(X, n)

function _to_device(like, a)
    d = similar(like, eltype(a), size(a))
    copyto!(d, a)
    return d
end

# ---------------------------------------------------------------------------
# main solver
# ---------------------------------------------------------------------------
"""
    admm_deconvolution(measured, psf, x0; loss, regularizer, λ, background, opt::ADMM, ...)

Internal driver used by `deconvolution(...; opt=ADMM())`.
`measured` is the (normalized) data, `psf` the (normalized) PSF already padded
to the reconstruction size and `ifftshift`ed in its dimensions, `x0` the initial
object estimate on the (padded) reconstruction grid.
"""
function admm_deconvolution(measured::AbstractArray{T}, psf, x0;
        loss=Poisson(), regularizer=nothing, λ=T(0.05), background=zero(T),
        opt::ADMM=ADMM(), iterations=nothing, nonneg=true,
        debug_f=nothing, progress=nothing) where {T<:Real}
    N = ndims(x0)
    sz = size(x0)
    niter = isnothing(iterations) ? opt.iterations : iterations
    λT = T(λ); bg = T(background)

    lossp = admm_loss(loss, T)
    blocks, tik = admm_reg(regularizer, N, T)

    # --- kernel on the reconstruction grid (identity along non-PSF dimensions)
    h = similar(x0); fill!(h, zero(T))
    view(h, ntuple(d -> d <= ndims(psf) ? (1:size(psf, d)) : (1:1), N)...) .= psf
    P = _plan_rfft(x0)
    Hhat = P * h
    Pi = _plan_irfft(Hhat, sz[1])
    Hc = conj.(Hhat)

    # --- data embedded into the reconstruction grid + mask (Almeida & Figueiredo)
    if size(measured) == sz
        y = measured; mask = nothing
    else
        y = similar(x0); fill!(y, zero(T)); center_set!(y, measured)
        mask = similar(x0); fill!(mask, zero(T))
        o = similar(measured); fill!(o, one(T)); center_set!(mask, o)
    end

    # --- Fourier symbols for the x-update
    A = abs2.(Hhat)
    rsz = size(Hhat)
    Bc = zeros(T, rsz); Qc = zeros(T, rsz)
    for b in blocks, s in b.stencils
        Bc .+= _abs2_symbol(s, sz, T)
    end
    for s in tik
        Qc .+= _abs2_symbol(s, sz, T)
    end
    B = _to_device(A, Bc); Q = _to_device(A, Qc)
    has_tik = !isempty(tik)
    ρ = isnothing(opt.ρ) ? default_ρ(lossp, mean(measured)) : opt.ρ
    denom = similar(A)
    set_denom!(ρ) = (denom .= A .+ B .+ one(T) .+ (has_tik ? T(2λT / ρ) .* Q : zero(T)))
    set_denom!(ρ)

    # --- variables
    x = copy(x0)
    Hx = similar(x0); X = P * x; Xt = similar(X); R1 = similar(X); R2 = similar(X)
    Xt .= Hhat .* X; mul!(Hx, Pi, Xt)
    z1 = copy(Hx); w1 = zero(x0)
    zs = [[apply!(similar(x0), s, x) for s in b.stencils] for b in blocks]
    ws = [[zero(x0) for _ in b.stencils] for b in blocks]
    adj = [[s' for s in b.stencils] for b in blocks]
    z3 = nonneg ? max.(x, zero(T)) : copy(x); w3 = zero(x0)
    r = similar(x0); t = similar(x0); s1 = similar(x0); s2 = similar(x0)
    # fused elementwise prox update on the CPU (GPU arrays: generic broadcasts)
    fused = x0 isa Array

    history = NamedTuple{(:iteration, :r_rel, :s_rel, :ρ), Tuple{Int, Float64, Float64, Float64}}[]
    converged = false
    t0 = time()
    report(it, ρ) = begin
        isnothing(debug_f) || debug_f(z3)
        if !isnothing(progress)
            # errors of the progress recorder do not abort the solver (as in `invert`);
            # use `opt.callback` to stop
            try
                Base.invokelatest(record_progress!, progress, z3, it,
                                  admm_objective(z3, P, Pi, Hhat, y, measured, mask, loss, regularizer, λT, bg),
                                  time() - t0, ρ)
            catch
            end
        end
    end
    report(0, ρ)

    it = 0
    for outer it in 1:niter
        check = it % opt.check_every == 0 || it == niter
        # ---- x-update:  x = (HᵀH + ΣKᵀK + I + 2λ/ρ QᵀQ)⁻¹ [Hᵀ(z₁-w₁) + ΣKᵀ(z₂-w₂) + (z₃-w₃)]
        _tb!(-, r, z1, w1)
        mul!(R1, P, r)
        _tb!(-, r, z3, w3)
        for (bi, b) in enumerate(blocks), k in eachindex(b.stencils)
            apply_add_diff!(r, adj[bi][k], zs[bi][k], ws[bi][k])
        end
        mul!(R2, P, r)
        _tb!((hc, r1, r2, dn) -> (hc * r1 + r2) / dn, X, Hc, R1, R2, denom)
        _tb!(*, Xt, Hhat, X); mul!(Hx, Pi, Xt)
        copyto!(Xt, X);       mul!(x, Pi, Xt)

        # ---- z/w-updates, residuals ‖Kx - z‖ (primal) and ‖z - z_old‖ (dual, scaled)
        rp2 = 0.0; rd2 = 0.0; nKx2 = 0.0; nz2 = 0.0; nw2 = 0.0
        # data block
        check && copyto!(s2, z1)
        _tb!(+, t, Hx, w1)
        prox_data!(z1, lossp, t, y, ρ, bg, mask, opt.newton_steps)
        _tb!(-, w1, t, z1)
        if check
            rp2 += sum(abs2, Hx .- z1)
            rd2 += sum(abs2, z1 .- s2); nKx2 += sum(abs2, Hx); nz2 += sum(abs2, z1); nw2 += sum(abs2, w1)
        end
        # regularizer blocks
        for (bi, b) in enumerate(blocks)
            vs = ws[bi]          # w holds v = Kx + w during the prox
            olds = check ? [copy(z) for z in zs[bi]] : nothing
            for k in eachindex(b.stencils)
                if check
                    apply!(t, b.stencils[k], x)
                    nKx2 += sum(abs2, t)
                    _tb!(+, vs[k], vs[k], t)
                else
                    apply_add!(vs[k], b.stencils[k], x)
                end
            end
            if fused
                _prox_update!(Val(length(b.stencils)), Val(b.kind), zs[bi], vs, T(λT / ρ))
            else
                prox_block!(zs[bi], b.kind, vs, T(λT / ρ), s1, s2)
                foreach(k -> _tb!(-, vs[k], vs[k], zs[bi][k]), eachindex(b.stencils))
            end
            for k in eachindex(b.stencils)
                if check
                    apply!(t, b.stencils[k], x)
                    rp2 += sum(abs2, t .- zs[bi][k]); rd2 += sum(abs2, zs[bi][k] .- olds[k])
                    nz2 += sum(abs2, zs[bi][k]); nw2 += sum(abs2, vs[k])
                end
            end
        end
        # positivity block
        check && copyto!(s2, z3)
        if nonneg
            z0 = zero(T)
            _tb!((a, b) -> max(a + b, z0), z3, x, w3)
        else
            _tb!(+, z3, x, w3)
        end
        _tb!((a, b, z) -> a + b - z, w3, x, w3, z3)
        if check
            rp2 += sum(abs2, x .- z3); rd2 += sum(abs2, z3 .- s2)
            nKx2 += sum(abs2, x); nz2 += sum(abs2, z3); nw2 += sum(abs2, w3)
        end

        (it % opt.progress_every == 0 || it == niter) && report(it, ρ)
        if !isnothing(opt.callback) && opt.callback(it, z3) === true
            break       # stop requested: return the current (feasible) iterate
        end

        if check
            r_rel = sqrt(rp2) / max(sqrt(nKx2), sqrt(nz2), eps())
            s_rel = sqrt(rd2) / max(sqrt(nw2), eps())
            push!(history, (iteration=it, r_rel=r_rel, s_rel=s_rel, ρ=ρ))
            if opt.tol > 0 && r_rel < opt.tol && s_rel < opt.tol
                converged = true
                break
            end
            if opt.adaptive_ρ
                fac = r_rel > opt.ρ_balance * s_rel ? opt.ρ_factor :
                      s_rel > opt.ρ_balance * r_rel ? 1 / opt.ρ_factor : 1.0
                if fac != 1.0
                    ρ *= fac
                    w1 ./= T(fac); w3 ./= T(fac)
                    foreach(wb -> foreach(w -> (w ./= T(fac)), wb), ws)
                    set_denom!(ρ)
                end
            end
        end
    end
    return z3, ADMMResult(z3, it, converged, ρ, history)
end

# objective in the conventions of the gradient-based path (user loss and regularizer)
function admm_objective(x, P, Pi, Hhat, y, measured, mask, loss, regularizer, λ, bg)
    μ = Pi * (Hhat .* (P * x))
    μ = iszero(bg) ? μ : μ .+ bg
    μc = collect_if_view(center_extract(μ, size(measured)))
    val = sum(loss(μc, measured, similar(μc)))
    if !isnothing(regularizer)
        val += λ * Base.invokelatest(regularizer, x)
    end
    return val
end

collect_if_view(a::SubArray) = copy(a)
collect_if_view(a) = a
