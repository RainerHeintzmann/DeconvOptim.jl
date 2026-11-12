# Chambolle–Pock (primal–dual hybrid gradient, PDHG) solver for the deconvolution problems
#
#     min_x  L(C(h ∗ x) + b, y) + λ R(x)      s.t. x ≥ 0
#
# without automatic differentiation, written as the saddle-point problem
#
#     min_x max_p  ⟨K x, p⟩ + G(x) − F*(p),    K = [H; K₁; …; Q],  G = ι_{x ≥ 0}
#
# * A. Chambolle, T. Pock, "A first-order primal-dual algorithm for convex problems with
#   applications to imaging", J. Math. Imaging Vis. 40, 120 (2011)
# * T. Goldstein, M. Li, X. Yuan, "Adaptive primal-dual splitting methods for statistical
#   learning and image processing", NeurIPS 28 (2015)  [residual balancing of τ, σ]
#
# Compared with ADMM (admm.jl), which shares the regularizer stencils and the data-term proxes:
# every term enters through K and its adjoint only, so there is no linear solve (no Fourier-space
# division) and the boundaries of the operators need not be periodic in principle; each iteration
# costs one application of K and one of Kᵀ (two FFT pairs). The proximal maps of the conjugates
# follow from the Moreau identity  prox_{σF*}(q) = q − σ prox_{F/σ}(q/σ):
#   data term (F = L):          q − σ prox_L(q/σ; ρ = σ)    (the ADMM data prox), 0 in the padding
#   group norm (TV, TH):        projection of every pixel's vector onto the ball ‖p‖ ≤ λ; with the
#                               Huber norm (δ = √ϵ > 0, F* = δ/(2λ)‖p‖² + ι) first scaled by 1/(1 + σδ/λ)
#   Schatten-1 (HS):            eigenvalue clipping to [−λ, λ] (projection onto the spectral ball)
#   Tikhonov (F = λ‖z‖²):       q / (1 + σ/(2λ))
# The ℓ1 norm of the object (L1()) is part of G: its prox (with x ≥ 0) is a (one-sided) soft
# thresholding in the primal step. TGV (TGV()) adds the primal field w (one component per
# dimension) and K gets the blocks [∇, −I] (dual ball λ) and [0, ε] (dual ball ratio·λ):
#   K = [H 0; ∇ −I; 0 ε]  acting on (x, w)

export ChambollePock

"""
    ChambollePock(; iterations=300, tol=0, adaptive=true, step_ratio=nothing, step_factor=0.99,
                    check_every=10, progress_every=1, callback=nothing, newton_steps=6)

Select the Chambolle–Pock (primal–dual) solver in [`deconvolution`](@ref) via
`opt=ChambollePock()`. Like [`ADMM`](@ref) it minimizes the exact (non-smoothed) objective
without automatic differentiation and enforces `x ≥ 0` by projection, and it supports the same
combinations:
- `loss`: `Poisson()`, `Gauss()`, `Anscombe(b)`
- `regularizer`: `nothing`, `TV()`, `TH()`, `HS()` (only `p=1`), `Tikhonov()` (all modes), `L1()`,
  and (only here) `TGV()`
- `mapping`: any non-`nothing` mapping is interpreted as `x ≥ 0`, `nothing` as unconstrained.

As in ADMM, the `ϵ` of `TV()`/`TH()` is the threshold `√ϵ` of a Huber norm (see [`ADMM`](@ref)).
Unlike ADMM it needs no linear solve: every iteration applies the forward model and the
regularizer operators once and their adjoints once. The reconstruction grid is periodic; use
`padding>0` in `deconvolution` against wrap-around (the padded border is not constrained by the data).

# Arguments
- `iterations=300`: maximum number of iterations (overridden by the `iterations` keyword of
   `deconvolution`, if given).
- `tol=0`: stop when the relative primal and dual residuals are both below `tol`; `0` always runs
   all iterations.
- `adaptive=true`: balance the primal and dual step sizes τ, σ by their residuals (Goldstein et al.
   2015); `τσ‖K‖²` stays constant.
- `step_ratio=nothing`: initial step sizes τ = r/‖K‖ and σ = step_factor²/(r‖K‖), i.e. the ratio
   τ/σ ∝ r² balances the primal variable (on the scale of the data) against the dual variables
   (of order one). `nothing`: r = 10 × the mean of the (normalized) data.
- `step_factor=0.99`: `τσ‖K‖² = step_factor²` (must be < 1).
- `check_every=10`: residuals (and the step balancing) are evaluated every `check_every` iterations.
- `progress_every=1`: `progress`/`debug_f` are called every `progress_every` iterations.
- `callback=nothing`: `callback(iteration, x)` after every iteration; returning `true` stops.
- `newton_steps=6`: per-pixel Newton steps of the `Anscombe` prox.
"""
Base.@kwdef struct ChambollePock
    iterations::Int = 300
    tol::Float64 = 0.0
    adaptive::Bool = true
    step_ratio::Union{Nothing, Float64} = nothing
    step_factor::Float64 = 0.99
    check_every::Int = 10
    progress_every::Int = 1
    callback::Any = nothing
    newton_steps::Int = 6
    balance::Float64 = 1.5        # Δ: rebalance when one residual exceeds the other by this factor
    α0::Float64 = 0.5             # initial adaptation strength
    η::Float64 = 0.95             # decay of the adaptation strength
end

"""
    ChambollePockResult

Second return value of `deconvolution(...; opt=ChambollePock())`. Fields: `minimizer` (full,
padded reconstruction in normalized units), `iterations`, `converged`, the final step sizes `τ`,
`σ`, and `history` (a vector of `(iteration, r_rel, s_rel, τ, σ)` named tuples).
"""
struct ChambollePockResult{A}
    minimizer::A
    iterations::Int
    converged::Bool
    τ::Float64
    σ::Float64
    history::Vector{NamedTuple{(:iteration, :r_rel, :s_rel, :τ, :σ), NTuple{5, Float64}}}
end

# ---------------------------------------------------------------------------
# proximal maps of the conjugates (dual steps)
# ---------------------------------------------------------------------------
# data term via Moreau: q − σ prox_{F/σ}(q/σ), with prox_{F/σ} = the ADMM prox at ρ = σ
@inline _cp_data(l::ADMMLoss, q, y, σ, bg, nsteps) = q - σ * prox(l, q / σ, y, σ, bg)
@inline _cp_data(l::AnscombeProx, q, y, σ, bg, nsteps) = q - σ * prox(l, q / σ, y, σ, bg, nsteps)

# per-pixel projections of a regularizer block onto the dual ball of radius λ, after scaling by
# c = 1/(1 + σδ/λ) (Huber; c = 1 for the plain norm)
@inline function _cp_proj(::Val{:group}, v::NTuple{K, T}, λ, c) where {K, T}
    f = c / max(one(T), c * sqrt(sum(abs2, v)) / λ)
    return map(vi -> f * vi, v)
end
# symmetric 2×2 matrix [[a, c/√2], [c/√2, d]] (components (a, d, c), see admm.jl): clip its
# eigenvalues m ± r to [−λ, λ]
@inline function _cp_proj(::Val{:schatten}, v::NTuple{3, T}, λ, c) where T
    a, d, c = v
    r = sqrt(abs2((a - d) / 2) + abs2(c) / 2)
    m = (a + d) / 2
    e1 = clamp(m + r, -λ, λ); e2 = clamp(m - r, -λ, λ)
    mn = (e1 + e2) / 2
    f = r > floatmin(r) ? (e1 - e2) / (2r) : zero(r)
    return (mn + f * (a - d) / 2, mn - f * (a - d) / 2, f * c)
end

# fused CPU projection over all components of a block (in place)
function _cp_proj_update!(::Val{K}, kind::Val, ps, λ, c) where K
    pt = ntuple(k -> ps[k], Val(K))
    _tchunks(length(pt[1])) do lo, hi
        @inbounds @simd for i in lo:hi
            v = map(p -> p[i], pt)
            z = _cp_proj(kind, v, λ, c)
            map((pa, zi) -> (pa[i] = zi), pt, z)
        end
    end
    return nothing
end

# device-generic projection (broadcasts; s1, s2 scratch arrays)
function _cp_proj_block!(ps, kind, λ, hs, s1, s2)
    T = eltype(s1)
    o = one(T)
    if kind == :group
        _tb!((v...) -> hs / max(o, hs * sqrt(sum(abs2, v)) / λ), s1, ps...)
        foreach(p -> _tb!(*, p, s1, p), ps)
    else
        a, d, c = ps
        # eigenvalues m ± r → clipped e₁, e₂;  m' = (e₁ + e₂)/2,  f = (e₁ − e₂)/(2r)
        _tb!((a, d, c) -> sqrt(abs2((a - d) / 2) + abs2(c) / 2), s1, a, d, c)                          # s1 = r
        _tb!((a, d, r) -> (clamp((a + d) / 2 + r, -λ, λ) + clamp((a + d) / 2 - r, -λ, λ)) / 2, s2, a, d, s1)  # s2 = m'
        _tb!((a, d, r) -> r > floatmin(r) ? (clamp((a + d) / 2 + r, -λ, λ) - clamp((a + d) / 2 - r, -λ, λ)) / (2r) : zero(r),
             s1, a, d, s1)                                                                              # s1 = f
        _tb!(*, c, s1, c)                                                                               # c' = f c
        _tb!((f, a, d) -> f * (a - d) / 2, s1, s1, a, d)                                                # s1 = f (a − d)/2
        _tb!(+, a, s2, s1)                                                                              # a' = m' + s1
        _tb!(-, d, s2, s1)                                                                              # d' = m' − s1
    end
    return ps
end

# ---------------------------------------------------------------------------
# TGV²: K blocks [∇, −I] and [0, ε] on (x, w)
# ---------------------------------------------------------------------------
struct TGVOp{S}
    grad::Vector{S}                       # forward differences D⁺_d: component d of ∇x
    sym::Vector{Vector{Tuple{Int, S}}}    # component k of ε(w) = Σ S w_j over (j, S); off-diagonal
                                          # components scaled by √2 (Frobenius ↔ Euclidean norm)
    ratio::Float64
end

function cp_tgv(reg, N, T, sz)
    p = reg.params
    nd = isnothing(p.num_dims) ? N : min(p.num_dims, N)
    sd = _sum_dims(p.sum_dims, nd)
    w = _weights(p.weights, length(sd), T)
    keep = [k for k in eachindex(sd) if sz[sd[k]] > 1]       # no differences along singleton axes
    isempty(keep) && throw(ArgumentError("TGV: no dimension with more than one pixel"))
    sd = sd[keep]; c = sqrt.(w[keep])
    S = Stencil{N, T}
    grad = S[Stencil([_unit(N, d, 1), _unit(N, d, 0)], T[ck, -ck]) for (d, ck) in zip(sd, c)]
    # backward differences D⁻ for ε (so that εᵀ and ∇ᵀ line up as in Bredies et al.)
    bwd(k, f) = Stencil([_unit(N, sd[k], 0), _unit(N, sd[k], -1)], T[f * c[k], -f * c[k]])
    n = length(sd)
    sym = Vector{Tuple{Int, S}}[[(k, bwd(k, 1))] for k in 1:n]
    r = 1 / sqrt(2)                                          # √2 · (D⁻_k w_l + D⁻_l w_k)/2
    for k in 1:n, l in k+1:n
        push!(sym, [(l, bwd(k, r)), (k, bwd(l, r))])
    end
    return TGVOp{S}(grad, sym, Float64(p.ratio))
end

function _symbol(s::Stencil, sz, ::Type{T}) where T
    δ = zeros(T, sz); δ[1] = one(T)
    return rfft(apply!(similar(δ), s, δ))
end

# upper bound of ‖K‖² for K = [H 0; ∇ −I; 0 ε] (+ x-only blocks with Σ|S|² = S2): the largest
# eigenvalue of the per-frequency Gram matrix G(f) = K̂ᴴK̂, bounded by its largest absolute row sum
function _tgv_norm2(H2, S2, tgv::TGVOp, sz, ::Type{T}) where T
    n = length(tgv.grad)
    Gs = [_symbol(s, sz, T) for s in tgv.grad]
    E = [[zeros(Complex{T}, size(H2)) for _ in 1:n] for _ in tgv.sym]    # Ê[k][j]
    for (k, comp) in enumerate(tgv.sym), (j, st) in comp
        E[k][j] .+= _symbol(st, sz, T)
    end
    G00 = H2 .+ S2 .+ sum(g -> abs2.(g), Gs)
    bound = G00 .+ sum(g -> abs.(g), Gs)                                  # row 0
    for j in 1:n
        row = abs.(Gs[j])                                                 # |G_j0|
        for j2 in 1:n
            gjj = (j == j2 ? ones(T, size(H2)) : zeros(T, size(H2))) .+ zeros(Complex{T}, size(H2))
            for k in eachindex(tgv.sym)
                gjj .+= conj.(E[k][j]) .* E[k][j2]
            end
            row .+= abs.(gjj)
        end
        bound = max.(bound, row)
    end
    return Float64(maximum(bound))
end

# Σᵢ ‖(∇x − w)ᵢ‖ + ratio Σᵢ ‖ε(w)ᵢ‖ at the current (x, w) (≥ TGV(x)), scratch t, acc
function _tgv_value(x, wv, tgv::TGVOp, t, acc)
    fill!(acc, zero(eltype(acc)))
    for (d, s) in enumerate(tgv.grad)
        apply!(t, s, x); acc .+= abs2.(t .- wv[d])
    end
    v1 = Float64(sum(sqrt, acc))
    fill!(acc, zero(eltype(acc)))
    for comp in tgv.sym
        fill!(t, zero(eltype(t)))
        for (j, st) in comp
            apply_add!(t, st, wv[j])
        end
        acc .+= abs2.(t)
    end
    return v1 + tgv.ratio * Float64(sum(sqrt, acc))
end

# ---------------------------------------------------------------------------
# main solver
# ---------------------------------------------------------------------------
"""
    chambolle_pock_deconvolution(measured, psf, x0; loss, regularizer, λ, background, opt, ...)

Internal driver used by `deconvolution(...; opt=ChambollePock())`; arguments as for
`admm_deconvolution`.
"""
function chambolle_pock_deconvolution(measured::AbstractArray{T}, psf, x0;
        loss=Poisson(), regularizer=nothing, λ=T(0.05), background=zero(T),
        opt::ChambollePock=ChambollePock(), iterations=nothing, nonneg=true,
        debug_f=nothing, progress=nothing) where {T<:Real}
    N = ndims(x0)
    sz = size(x0)
    niter = isnothing(iterations) ? opt.iterations : iterations
    λT = T(λ); bg = T(background)
    0 < opt.step_factor < 1 || throw(ArgumentError("step_factor must be in (0, 1)"))

    lossp = admm_loss(loss, T)
    l1 = _is_l1(regularizer)
    S = Stencil{N, T}
    tgv = _reg_kind(regularizer) == :TGV ? cp_tgv(regularizer, N, T, sz) : nothing
    blocks, tik = isnothing(tgv) ? admm_reg(regularizer, N, T) : (SplitBlock{S}[], S[])

    # --- forward model on the reconstruction grid (identity along non-PSF dimensions)
    h = similar(x0); fill!(h, zero(T))
    view(h, ntuple(d -> d <= ndims(psf) ? (1:size(psf, d)) : (1:1), N)...) .= psf
    P = _plan_rfft(x0)
    Hhat = P * h
    Pi = _plan_irfft(Hhat, sz[1])
    Hc = conj.(Hhat)

    # --- data embedded into the reconstruction grid + mask (no data in the padding)
    if size(measured) == sz
        y = measured; mask = nothing
    else
        y = similar(x0); fill!(y, zero(T)); center_set!(y, measured)
        mask = similar(x0); fill!(mask, zero(T))
        o = similar(measured); fill!(o, one(T)); center_set!(mask, o)
    end

    # --- ‖K‖² = max over frequencies of |H|² + Σ|Sₖ|² (all operators are periodic convolutions)
    S2 = zeros(T, size(Hhat))
    for b in blocks, s in b.stencils
        S2 .+= _abs2_symbol(s, sz, T)
    end
    for s in tik
        S2 .+= _abs2_symbol(s, sz, T)
    end
    L = isnothing(tgv) ? sqrt(Float64(maximum(Array(abs2.(Hhat))) + maximum(S2))) :
                         sqrt(_tgv_norm2(Array(abs2.(Hhat)), S2, tgv, sz, T))
    # Tikhonov: F(z) = λ‖z‖² (as in the gradient-based path and ADMM)
    # initial step ratio: the primal variable lives on the scale of the data, the duals are O(1)
    # (10× the data mean: robust with the adaptive balancing on Poisson/Gauss/Anscombe tests)
    r = isnothing(opt.step_ratio) ? 10 * Float64(mean(measured)) : opt.step_ratio
    τ = r / L
    σ = opt.step_factor^2 / (r * L)

    # --- variables
    x = copy(x0); nonneg && (x .= max.(x, zero(T)))
    xbar = copy(x); xn = similar(x0); xs = similar(x0); t = similar(x0); kt = similar(x0)
    X = P * x; Xt = similar(X)
    pd = zero(x0)                                               # dual of the data term
    pb = [[zero(x0) for _ in b.stencils] for b in blocks]       # duals of the regularizer blocks
    pt = [zero(x0) for _ in tik]                                # duals of the Tikhonov terms
    adj = [[s' for s in b.stencils] for b in blocks]
    tadj = [s' for s in tik]
    s1 = similar(x0); s2 = similar(x0)
    fused = x0 isa Array
    # residual bookkeeping (only on check iterations)
    x_old = similar(x0); xbar_old = similar(x0)
    pd_old = similar(x0); pb_old = [[similar(x0) for _ in b.stencils] for b in blocks]; pt_old = [similar(x0) for _ in tik]
    # TGV: primal field w (+ extrapolation, σ·w̄), duals pg (of ∇x − w) and pe (of ε(w)); empty otherwise
    tgrad = isnothing(tgv) ? S[] : tgv.grad
    tsym = isnothing(tgv) ? Vector{Tuple{Int, S}}[] : tgv.sym
    nw = length(tgrad)
    wv = [zero(x0) for _ in 1:nw]; wn = [zero(x0) for _ in 1:nw]; wbar = [zero(x0) for _ in 1:nw]
    wsig = [similar(x0) for _ in 1:nw]
    pg = [zero(x0) for _ in 1:nw]; pe = [zero(x0) for _ in tsym]
    gadj = [s' for s in tgrad]
    # εᵀ, sorted by the w component it acts on: (k, Sᵀ) for every term S w_j of component k
    symadj = [[(k, st') for (k, comp) in enumerate(tsym) for (j2, st) in comp if j2 == j] for j in 1:nw]
    w_old = [similar(x0) for _ in 1:nw]; wbar_old = [similar(x0) for _ in 1:nw]
    pg_old = [similar(x0) for _ in 1:nw]; pe_old = [similar(x0) for _ in tsym]
    λe = T(isnothing(tgv) ? λ : tgv.ratio * λ)                    # dual radius of the ε(w) block

    # out = Kᵀ p
    function applyKt!(out, qd, qb, qt)
        mul!(X, P, qd); Xt .= Hc .* X; mul!(out, Pi, Xt)
        for (bi, b) in enumerate(blocks), k in eachindex(b.stencils)
            apply_add!(out, adj[bi][k], qb[bi][k])
        end
        for (k, s) in enumerate(tadj)
            apply_add!(out, s, qt[k])
        end
        for d in 1:nw
            apply_add!(out, gadj[d], pg[d])
        end
        return out
    end

    history = NamedTuple{(:iteration, :r_rel, :s_rel, :τ, :σ), NTuple{5, Float64}}[]
    converged = false
    α = opt.α0
    t0 = time()
    report(it) = begin
        isnothing(debug_f) || debug_f(x)
        if !isnothing(progress)
            try
                obj = isnothing(tgv) ?
                    admm_objective(x, P, Pi, Hhat, y, measured, mask, loss, regularizer, λT, bg) :
                    admm_objective(x, P, Pi, Hhat, y, measured, mask, loss, nothing, λT, bg) +
                    λ * _tgv_value(x, wv, tgv, s1, s2)
                Base.invokelatest(record_progress!, progress, x, it, obj, time() - t0, τ)
            catch
            end
        end
    end
    report(0)

    # constants captured by the elementwise kernels: each assigned exactly once (a reassigned
    # captured variable is boxed, which GPU kernels cannot take)
    zT = zero(T)
    it = 0
    for outer it in 1:niter
        check = it % opt.check_every == 0 || it == niter
        if check
            copyto!(x_old, x); copyto!(xbar_old, xbar); copyto!(pd_old, pd)
            foreach((a, b) -> foreach(copyto!, a, b), pb_old, pb); foreach(copyto!, pt_old, pt)
            foreach(copyto!, w_old, wv); foreach(copyto!, wbar_old, wbar)
            foreach(copyto!, pg_old, pg); foreach(copyto!, pe_old, pe)
        end
        σT = T(σ); τT = T(τ)

        # ---- dual step: p = prox_{σF*}(p + σ K x̄)
        _tb!(v -> σT * v, xs, xbar)
        mul!(X, P, xs); Xt .= Hhat .* X; mul!(t, Pi, Xt)           # σ H x̄
        nst = opt.newton_steps
        if isnothing(mask)
            _tb!((p, hx, yy) -> _cp_data(lossp, p + hx, yy, σT, bg, nst), pd, pd, t, y)
        else
            _tb!((p, hx, yy, m) -> m > 0 ? _cp_data(lossp, p + hx, yy, σT, bg, nst) : zT, pd, pd, t, y, mask)
        end
        for (bi, b) in enumerate(blocks)
            for k in eachindex(b.stencils)
                apply_add!(pb[bi][k], b.stencils[k], xs)
            end
            cb = T(λ > 0 ? 1 / (1 + σ * b.δ / λ) : 1)                # Huber scaling (1 for δ = 0)
            if fused
                _cp_proj_update!(Val(length(b.stencils)), Val(b.kind), pb[bi], λT, cb)
            else
                _cp_proj_block!(pb[bi], b.kind, λT, cb, s1, s2)
            end
        end
        if nw > 0                                                  # TGV duals
            for j in 1:nw
                _tb!(v -> σT * v, wsig[j], wbar[j])
            end
            for d in 1:nw
                apply_add!(pg[d], tgrad[d], xs)                    # + σ(D⁺_d x̄ − w̄_d)
                _tb!(-, pg[d], pg[d], wsig[d])
            end
            for (k, comp) in enumerate(tsym), (j, st) in comp
                apply_add!(pe[k], st, wsig[j])                     # + σ ε(w̄)
            end
            oT = one(T)
            if fused
                _cp_proj_update!(Val(nw), Val(:group), pg, λT, oT)
                _cp_proj_update!(Val(length(pe)), Val(:group), pe, λe, oT)
            else
                _cp_proj_block!(pg, :group, λT, oT, s1, s2)
                _cp_proj_block!(pe, :group, λe, oT, s1, s2)
            end
        end
        if !isempty(tik)
            ft = T(1 / (1 + σ / (2λ)))
            for (k, s) in enumerate(tik)
                apply_add!(pt[k], s, xs)
                _tb!(v -> ft * v, pt[k], pt[k])
            end
        end

        # ---- primal step: x⁺ = prox_{τG}(x − τ Kᵀ p),  x̄ = 2x⁺ − x
        applyKt!(kt, pd, pb, pt)
        κ1 = l1 ? τT * λT : zT                                     # L1(): prox of τλ‖x‖₁ (+ ι_{x ≥ 0})
        if nonneg
            _tb!((a, g) -> max(a - τT * g - κ1, zT), xn, x, kt)
        elseif l1
            _tb!((a, g) -> _soft(a - τT * g, κ1), xn, x, kt)
        else
            _tb!((a, g) -> a - τT * g, xn, x, kt)
        end
        _tb!((a, b) -> 2a - b, xbar, xn, x)
        x, xn = xn, x
        for j in 1:nw                                              # TGV: w⁺ = w − τ(−pg + εᵀpe)
            _tb!(-, t, pg[j])
            for (k, st) in symadj[j]
                apply_add!(t, st, pe[k])
            end
            _tb!((a, g) -> a - τT * g, wn[j], wv[j], t)
            _tb!((a, b) -> 2a - b, wbar[j], wn[j], wv[j])
        end
        wv, wn = wn, wv

        (it % opt.progress_every == 0 || it == niter) && report(it)
        if !isnothing(opt.callback) && opt.callback(it, x) === true
            break
        end

        if check
            # residuals of the optimality conditions at (x⁺, p⁺), for this ordering (p first):
            #   x⁺ = prox_{τG}(x − τKᵀp⁺)  ⇒  primal residual (x − x⁺)/τ
            #   p⁺ = prox_{σF*}(p + σKx̄)  ⇒  dual residual (p − p⁺)/σ + K(x̄ − x⁺)
            rp = sqrt(Float64(sum(abs2, x_old .- x)) + sum(j -> Float64(sum(abs2, w_old[j] .- wv[j])), 1:nw; init=0.0)) / τ
            nx = sqrt(Float64(sum(abs2, x)) + sum(j -> Float64(sum(abs2, wv[j])), 1:nw; init=0.0)) / τ
            _tb!(-, xbar_old, xbar_old, x)                             # x̄ − x⁺
            mul!(X, P, xbar_old); Xt .= Hhat .* X; mul!(t, Pi, Xt)     # K(x̄ − x⁺), data block
            mul!(X, P, x); Xt .= Hhat .* X; mul!(kt, Pi, Xt)           # K x⁺ (for the scale)
            if isnothing(mask)
                rd2 = Float64(sum(abs2, (pd_old .- pd) ./ σT .+ t)); nKx2 = Float64(sum(abs2, kt))
            else                                                       # no data (and residual) in the padding
                rd2 = Float64(sum(abs2, ((pd_old .- pd) ./ σT .+ t) .* mask)); nKx2 = Float64(sum(abs2, kt .* mask))
            end
            np2 = Float64(sum(abs2, pd))
            for (bi, b) in enumerate(blocks), k in eachindex(b.stencils)
                apply!(t, b.stencils[k], xbar_old); apply!(kt, b.stencils[k], x)
                rd2 += Float64(sum(abs2, (pb_old[bi][k] .- pb[bi][k]) ./ σT .+ t))
                nKx2 += Float64(sum(abs2, kt)); np2 += Float64(sum(abs2, pb[bi][k]))
            end
            for (k, s) in enumerate(tik)
                apply!(t, s, xbar_old); apply!(kt, s, x)
                rd2 += Float64(sum(abs2, (pt_old[k] .- pt[k]) ./ σT .+ t))
                nKx2 += Float64(sum(abs2, kt)); np2 += Float64(sum(abs2, pt[k]))
            end
            foreach((a, b) -> _tb!(-, a, a, b), wbar_old, wv)          # w̄ − w⁺
            for d in 1:nw                                              # ∇x − w block
                apply!(t, tgrad[d], xbar_old); t .-= wbar_old[d]
                apply!(kt, tgrad[d], x); kt .-= wv[d]
                rd2 += Float64(sum(abs2, (pg_old[d] .- pg[d]) ./ σT .+ t))
                nKx2 += Float64(sum(abs2, kt)); np2 += Float64(sum(abs2, pg[d]))
            end
            for (k, comp) in enumerate(tsym)                           # ε(w) block
                fill!(t, zT); fill!(kt, zT)
                for (j, st) in comp
                    apply_add!(t, st, wbar_old[j]); apply_add!(kt, st, wv[j])
                end
                rd2 += Float64(sum(abs2, (pe_old[k] .- pe[k]) ./ σT .+ t))
                nKx2 += Float64(sum(abs2, kt)); np2 += Float64(sum(abs2, pe[k]))
            end
            rd = sqrt(rd2)
            r_rel = rp / max(nx, eps())                                # = ‖Δx‖ / ‖x‖
            s_rel = rd / max(sqrt(nKx2), sqrt(np2) / σ, eps())
            push!(history, (iteration=Float64(it), r_rel=r_rel, s_rel=s_rel, τ=τ, σ=σ))
            if opt.tol > 0 && r_rel < opt.tol && s_rel < opt.tol
                converged = true
                break
            end
            if opt.adaptive
                if rp > opt.balance * rd
                    τ /= (1 - α); σ *= (1 - α); α *= opt.η
                elseif rd > opt.balance * rp
                    τ *= (1 - α); σ /= (1 - α); α *= opt.η
                end
            end
        end
    end
    return x, ChambollePockResult(x, it, converged, τ, σ, history)
end
