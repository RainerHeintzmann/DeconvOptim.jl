export map_cj_exact

"""
    map_cj_exact(measured, psf; <keyword arguments>)

Maximum-a-posteriori deconvolution with the square mapping `object = x.^2`,
Fletcher--Reeves conjugate gradients and an exact line search, in the spirit
of the MAPGR/MAPPR algorithms of Verveer and Jovin (J. Opt. Soc. Am. A 15,
1077 (1998)). Naming: MAP + noise model (G Gauss, P Poisson, A Anscombe) +
penalty (R Good's roughness, H Hessian), e.g. MAPGR, MAPGH, MAPPR, MAPAH.

Along a search line `x + α d` the blurred object is exactly
`μ(α) = H x² + 2α H(x d) + α² H(d²)`. Both convolutions are computed once per
iteration, after which the line function and its first two derivatives are
known at any `α` without further convolutions:

- `loss=:gauss`, `penalty=:good` (MAPGR): the line function is a quartic
  polynomial with scalar coefficients; its minimum is found in closed form
  from the real roots of the cubic `φ'(α) = 0`, without iteration.
- all other combinations: the data term is evaluated pointwise from the
  stored arrays (`:poisson`, `:anscombe`) or as the quartic (`:gauss`); Good's
  penalty is quadratic in `α` with scalar coefficients, and the Hessian
  penalty with its derivatives is evaluated by one fused stencil kernel. The
  minimum is found by up to `newton_iterations` safeguarded Newton steps,
  none of which needs a convolution.

Each iteration costs three convolutions (`H(x d)`, `H(d²)` and the adjoint for
the gradient), and the blurred object is updated exactly. Apart from the OTF
and the FFT buffer, six full-size arrays are used (`x`, blurred object,
gradient, search direction, `H(x d)`, `H(d²)`), plus a padded copy of the
data when padding.

# Keyword arguments
- `loss=:poisson`: `:poisson`, `:anscombe` or `:gauss` (`Σ ω (μ - measured)²`,
  the constant noise variance absorbed in `gamma`).
- `noise_weights=nothing`: per-pixel weights `ω` (inverse variances) of the
  `:gauss` loss, an array of the size of `measured`; `nothing` means `ω = 1`.
  The line function stays a quartic, so MAPGR keeps its closed-form step.
  See [`map_cj_lean`](@ref).
- `penalty=:good`: `:good` (Good's roughness `-4 xᵀLx`) or
  `:hessian_frobenius`.
- `anscombe_offset=3/8`: `β` of the `:anscombe` loss, in measured counts.
- `gamma=0.0`: nonnegative regularization strength.
- `epsilon=nothing`, `weights=nothing`, `background=0`, `iterations=60`,
  `saddle_escape=1.0`, `escape_threshold=1e-4`, `padding=0.0`,
  `initial=nothing`, `conv_dims`, `normalize_psf=true`, `callback=nothing`:
  as in [`map_cj_lean`](@ref).
- `newton_iterations=2`: maximal number of Newton steps of the line search
  (unused for MAPGR, whose line search is closed-form); the search also stops
  once a step changes `α` by less than 0.1%. With 1 the method takes the same
  steps as [`map_cj_lean`](@ref); 2 already gives the convergence of the exact
  line search. For the Hessian penalty with Poisson or Anscombe noise each
  extra step costs about as much as a convolution, so `map_cj_lean` is often
  faster there.

The returned array is the reconstructed object `x.^2`.
"""
function map_cj_exact(measured::AbstractArray, psf::AbstractArray;
                      loss=:poisson,
                      penalty=:good,
                      anscombe_offset=3 / 8,
                      gamma=0.0,
                      epsilon=nothing,
                      weights=nothing,
                      background=0.0,
                      iterations=60,
                      newton_iterations=2,
                      saddle_escape=1.0,
                      escape_threshold=1e-4,
                      initial=nothing,
                      conv_dims=1:ndims(measured),
                      normalize_psf=true,
                      padding=0.0,
                      noise_weights=nothing,
                      callback=nothing)
    iterations isa Integer && iterations >= 0 ||
        throw(ArgumentError("`iterations` must be a nonnegative integer"))
    newton_iterations isa Integer && newton_iterations >= 1 ||
        throw(ArgumentError("`newton_iterations` must be a positive integer"))
    isfinite(saddle_escape) && saddle_escape >= 0 ||
        throw(ArgumentError("`saddle_escape` must be nonnegative and finite"))
    isfinite(escape_threshold) && escape_threshold >= 0 ||
        throw(ArgumentError("`escape_threshold` must be nonnegative and finite"))
    c, x = _lean_setup(measured, psf; penalty, loss, anscombe_offset,
                       mapping=:square, mapping_scale=1.0, gamma, epsilon,
                       weights, background, initial, conv_dims, normalize_psf,
                       padding, noise_weights)
    T = eltype(x)
    mu = similar(x)
    gradient = similar(x)
    direction = similar(x)
    hxd = similar(x)
    hdd = similar(x)
    partials = similar(x, T, 3 * cld(length(x), _EXACT_WG))
    fill!(direction, zero(T))
    _lean_model!(c, mu, x)
    gnorm2_prev = zero(T)
    alpha_prev = 1.0
    restart = true

    for iteration in 1:iterations
        # hxd <- ∇_x Ψ, gradient <- object-space gradient g_f (for the escape)
        _lean_gradient!(c, hxd, gradient, x, mu)
        gnorm2 = sum(abs2, hxd)
        beta = restart || iszero(gnorm2_prev) ? zero(T) : T(gnorm2 / gnorm2_prev)
        @. direction = beta * direction - hxd
        steepest = iszero(beta)
        if sum(_lazy(*, hxd, direction)) >= 0
            @. direction = -hxd
            steepest = true
        end
        gnorm2_prev = gnorm2
        restart = false
        saddle_escape > 0 &&
            _lean_saddle_escape!(c, direction, x, gradient, T(saddle_escape),
                                 T(escape_threshold))

        @. hxd = x * direction
        c.conv!(hxd, hxd, false)
        @. hdd = direction * direction
        c.conv!(hdd, hdd, false)
        alpha = _exact_line_search(c, x, direction, mu, hxd, hdd, gradient,
                                   partials, alpha_prev, newton_iterations)

        if !(isfinite(alpha) && alpha > 0)
            steepest && break      # no descent even along -∇Ψ: converged
            restart = true
        else
            alpha_prev = alpha
            a = T(alpha)
            @. x += a * direction
            @. mu += 2 * a * hxd + a * a * hdd
        end

        if callback !== nothing
            @. gradient = x * x
            object = c.padded && c.crop ? center_extract(gradient, c.measured_size) : gradient
            model = c.padded && c.crop ? center_extract(mu, c.measured_size) : mu
            callback(iteration, object, model) === true && break
        end
    end

    @. gradient = x * x
    return c.padded && c.crop ? center_extract(gradient, c.measured_size) : gradient
end

# Step length minimizing Ψ(x + α d) for μ(α) = mu + 2α m1 + α² m2: closed
# form for Gauss + Good (MAPGR), otherwise safeguarded Newton on the exact
# line function. `work` is overwritten.
function _exact_line_search(c, x, d, mu, m1, m2, work, partials, alpha_guess,
                            newton_iterations)
    g = Float64(c.gamma)
    good = c.penalty === :good
    pen = good && !iszero(g) ? _good_line_scalars(c, x, d, work) : (0.0, 0.0, 0.0)
    if c.loss === :gauss
        q = _gauss_data_coefficients(c, mu, m1, m2)
        if good
            # MAPGR: the whole line function is a quartic
            xlx, dlx, dld = pen
            return _quartic_argmin((q[1] - 4g * xlx, q[2] - 8g * dlx,
                                    q[3] - 4g * dld, q[4], q[5]))
        end
        data_line = a -> (_quartic(q, a), _quartic1(q, a), _quartic2(q, a))
    else
        data_line = a -> _pointwise_data_line(c, mu, m1, m2, a)
    end
    pen_line = if iszero(g)
        a -> (0.0, 0.0, 0.0)
    elseif good
        xlx, dlx, dld = pen
        a -> (-4g * (xlx + 2a * dlx + a^2 * dld), -8g * (dlx + a * dld), -8g * dld)
    else
        a -> g .* _exact_hf_line_sums(c, x, d, a, partials)
    end
    return _newton_line_search(a -> data_line(a) .+ pen_line(a), alpha_guess,
                               newton_iterations)
end

# safeguarded Newton minimization of φ along α > 0; `line(α)` returns
# (φ, φ', φ''). Returns 0 if no decrease is found.
function _newton_line_search(line, alpha_guess, newton_iterations)
    best, slope, curv = line(0.0)
    slope < 0 || return 0.0
    a, best_a = 0.0, 0.0
    for _ in 1:newton_iterations
        cand = curv > 0 ? a - slope / curv : max(2 * a, alpha_guess)
        if !(isfinite(cand) && cand > 0)
            cand = best_a > 0 ? best_a / 2 : alpha_guess
        end
        v, s1, s2 = line(cand)
        tries = 0
        while !(isfinite(v) && v < best) && tries < 20
            cand = (cand + best_a) / 2
            v, s1, s2 = line(cand)
            tries += 1
        end
        isfinite(v) && v < best || break
        step = abs(cand - best_a)
        best_a, best = cand, v
        a, slope, curv = cand, s1, s2
        # converged when the Newton update no longer changes α noticeably
        step <= 1e-3 * best_a && break
        curv > 0 && abs(slope / curv) <= 1e-3 * best_a && break
    end
    return best_a
end

# Σ ℓ(μ(α)) and its α-derivatives over the measured region, evaluated
# pointwise from μ(α) = mu + 2α m1 + α² m2 (any loss of `map_cj_lean`)
function _pointwise_data_line(c, mu, m1, m2, alpha)
    L = Val(c.loss)
    b, o = c.background, c.offset
    a = eltype(mu)(alpha)
    s(f) = Float64(sum(_lazy(f, mu, m1, m2, c.data)))
    v0 = s((u, p, q, g) -> _lean_masked(_lean_loss, L, u + 2a * p + a^2 * q, g, b, o))
    v1 = s((u, p, q, g) -> 2 * (p + a * q) *
        _lean_masked(_lean_residual, L, u + 2a * p + a^2 * q, g, b, o))
    v2 = s((u, p, q, g) -> begin
        ua = u + 2a * p + a^2 * q
        dmu = 2 * (p + a * q)
        _lean_masked(_lean_loss2, L, ua, g, b, o) * dmu^2 +
            2 * q * _lean_masked(_lean_residual, L, ua, g, b, o)
    end)
    return (v0, v1, v2)
end

# Coefficients (q0, ..., q4) of the Gaussian data term Σ ω (r0 + 2α m1 + α² m2)²
# over the measured region, r0 = H x² + b - g, m1 = H(x d), m2 = H(d²).
function _gauss_data_coefficients(c, mu, m1, m2)
    b = c.background
    in_data(g, v) = ifelse(g >= zero(g), v, zero(v))
    s(f) = Float64(sum(_lazy(f, mu, m1, m2, c.data, c.noise_weights)))
    srr = s((u, p, q, g, w) -> in_data(g, w * abs2(u + b - g)))
    sr1 = s((u, p, q, g, w) -> in_data(g, w * (u + b - g) * p))
    s11 = s((u, p, q, g, w) -> in_data(g, w * p * p))
    sr2 = s((u, p, q, g, w) -> in_data(g, w * (u + b - g) * q))
    s12 = s((u, p, q, g, w) -> in_data(g, w * p * q))
    s22 = s((u, p, q, g, w) -> in_data(g, w * q * q))
    return (srr, 4 * sr1, 4 * s11 + 2 * sr2, 4 * s12, s22)
end

# (xᵀLx, dᵀLx, dᵀLd) with the weighted Laplacian of Good's penalty
# -4γ (x + αd)ᵀ L (x + αd); `work` is overwritten
function _good_line_scalars(c, x, d, work)
    m = c.mapping
    fill!(work, zero(eltype(work)))
    _lean_launch!(_lean_good_grad_kernel!, work, x, m, c.weights, one(eltype(work)))
    # work = -8 L_w x
    xlx = -Float64(sum(_lazy(*, x, work))) / 8
    dlx = -Float64(sum(_lazy(*, d, work))) / 8
    _lean_launch!(_lean_good_value_kernel!, work, d, m, c.weights)
    dld = -Float64(sum(work)) / 4          # 4 Σ w (Δd)² = -4 dᵀ L_w d
    return (xlx, dlx, dld)
end

_quartic(q, a) = (((q[5] * a + q[4]) * a + q[3]) * a + q[2]) * a + q[1]
_quartic1(q, a) = ((4 * q[5] * a + 3 * q[4]) * a + 2 * q[3]) * a + q[2]
_quartic2(q, a) = (12 * q[5] * a + 6 * q[4]) * a + 2 * q[3]

# Global minimizer of the quartic q0 + q1 α + ... + q4 α⁴ among the real roots
# of its derivative (closed form) and α = 0; returns 0 if nothing is lower.
function _quartic_argmin(q)
    best_a, best = 0.0, _quartic(q, 0.0)
    for a in _cubic_real_roots(4 * q[5], 3 * q[4], 2 * q[3], q[2])
        # one Newton polish step against round-off in the closed form
        d2 = _quartic2(q, a)
        a = d2 != 0 ? a - _quartic1(q, a) / d2 : a
        v = _quartic(q, a)
        if isfinite(v) && v < best
            best_a, best = a, v
        end
    end
    return best_a
end

# real roots of a x³ + b x² + c x + d (Cardano / trigonometric form)
function _cubic_real_roots(a, b, c, d)
    scale = max(abs(a), abs(b), abs(c), abs(d))
    scale == 0 && return Float64[]
    if abs(a) <= 1e-12 * scale
        if abs(b) <= 1e-12 * scale
            return abs(c) > 0 ? [-d / c] : Float64[]
        end
        disc = c^2 - 4 * b * d
        disc < 0 && return Float64[]
        sq = sqrt(disc)
        return [(-c + sq) / (2b), (-c - sq) / (2b)]
    end
    b, c, d = b / a, c / a, d / a
    shift = -b / 3
    p = c - b^2 / 3
    qq = 2 * b^3 / 27 - b * c / 3 + d
    disc = (qq / 2)^2 + (p / 3)^3
    if disc > 0
        sq = sqrt(disc)
        return [cbrt(-qq / 2 + sq) + cbrt(-qq / 2 - sq) + shift]
    elseif p == 0
        return [shift]
    else
        r = 2 * sqrt(-p / 3)
        phi = acos(clamp(3 * qq / (p * r), -1.0, 1.0)) / 3
        return [r * cos(phi - 2π * k / 3) + shift for k in 0:2]
    end
end

const _EXACT_WG = 256

# Σ over voxels of sqrt(s), d/dα sqrt(s), d²/dα² sqrt(s) for the Hessian
# penalty along the line, s(α) = ε² + Σ w h², h = Hessian of (x + αd)²
function _exact_hf_line_sums(c, x, d, alpha, partials)
    nblocks = cld(length(x), _EXACT_WG)
    backend = KernelAbstractions.get_backend(x)
    _exact_hf_sums_kernel!(backend, _EXACT_WG)(partials, x, d, eltype(x)(alpha),
        c.pairs, c.pair_weights, c.weights, c.eps2, length(x), nblocks;
        ndrange=(nblocks * _EXACT_WG,))
    p = Array(partials)
    return (Float64(sum(view(p, 1:nblocks))),
            Float64(sum(view(p, nblocks+1:2nblocks))),
            Float64(sum(view(p, 2nblocks+1:3nblocks))))
end

# the three per-voxel terms, summed per work group into `partials`
@kernel unsafe_indices = true function _exact_hf_sums_kernel!(partials, x, d, alpha,
        pairs, pair_weights, weights, eps2, total, nblocks)
    n = @uniform prod(@groupsize())
    levels = @uniform trailing_zeros(n)
    idx = @index(Global, Linear)
    li = @index(Local, Linear)
    gi = @index(Group, Linear)
    T = eltype(partials)
    v0, v1, v2 = zero(T), zero(T), zero(T)
    if idx <= total
        I = CartesianIndices(x)[idx]
        v0, v1, v2 = _exact_hf_line_terms(_SquareLine(alpha), x, d, I, pairs,
                                          pair_weights, weights, eps2)
    end
    s0 = @localmem eltype(partials) (n,)
    s1 = @localmem eltype(partials) (n,)
    s2 = @localmem eltype(partials) (n,)
    s0[li] = v0
    s1[li] = v1
    s2[li] = v2
    @synchronize()
    for level in 1:levels
        stride = n >> level
        if li <= stride
            s0[li] += s0[li + stride]
            s1[li] += s1[li + stride]
            s2[li] += s2[li + stride]
        end
        @synchronize()
    end
    if li == 1
        partials[gi] = s0[1]
        partials[gi + nblocks] = s1[1]
        partials[gi + 2 * nblocks] = s2[1]
    end
end

@inline function _exact_hf_line_terms(m, x, d, I, pairs, pair_weights, weights, eps2)
    h = _lean_hessian(m, Val(1), x, d, I, pairs, weights)
    hp = _lean_hessian(m, Val(2), x, d, I, pairs, weights)
    hpp = _lean_hessian(m, Val(3), x, d, I, pairs, weights)
    s = eps2
    s1 = zero(eps2)
    s2 = zero(eps2)
    for j in 1:length(h)
        s += pair_weights[j] * h[j]^2
        s1 += pair_weights[j] * h[j] * hp[j]
        s2 += pair_weights[j] * (hp[j]^2 + h[j] * hpp[j])
    end
    r = sqrt(s)
    return (r, s1 / r, s2 / r - s1^2 / (s * r))
end

# field of the penalty along the line: f(α) = (x + α d)², its α-derivatives
struct _SquareLine{T}
    alpha::T
end
@inline _lean_field(m::_SquareLine, ::Val{1}, x, d, J) =
    @inbounds abs2(x[J] + m.alpha * d[J])
@inline _lean_field(m::_SquareLine, ::Val{2}, x, d, J) =
    @inbounds 2 * (x[J] + m.alpha * d[J]) * d[J]
@inline _lean_field(m::_SquareLine, ::Val{3}, x, d, J) =
    @inbounds 2 * abs2(d[J])

# per voxel: k = 0 -> sqrt(s), 1 -> d/dα sqrt(s), 2 -> d²/dα² sqrt(s)
@kernel function _exact_hf_line_kernel!(out, x, d, alpha, k, pairs, pair_weights,
                                        weights, eps2)
    I = @index(Global, Cartesian)
    t = _exact_hf_line_terms(_SquareLine(alpha), x, d, I, pairs, pair_weights,
                             weights, eps2)
    @inbounds out[I] = t[k + 1]
end
