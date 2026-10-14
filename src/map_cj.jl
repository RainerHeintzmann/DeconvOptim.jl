export map_cj
import KernelAbstractions
using KernelAbstractions: @kernel, @index, @groupsize, @uniform, @localmem,
                          @synchronize

"""
    map_cj(measured, psf; <keyword arguments>)

Maximum-a-posteriori deconvolution using the square mapping
`object = x.^2`, Fletcher--Reeves conjugate gradients, and an analytic
Newton line search. The data term is the Poisson negative log-likelihood
(`loss=:poisson`) or the Anscombe-type variance-stabilized least squares
`sum(abs2.(sqrt.(μ .+ β) .- sqrt.(measured .+ β)))` (`loss=:anscombe`),
with `μ` the blurred object plus background. Both are functions of `μ`,
which is a quadratic polynomial in the step length, so the line search is
exact for either choice.

`penalty=:good` selects MAPPR's Good/Laplacian penalty on the signed
optimization variable `x`. `penalty=:hessian_frobenius` selects the
smoothed Frobenius norm of the Hessian of `x.^2`.

# Keyword arguments
- `penalty=:good`: either `:good` or `:hessian_frobenius`.
- `loss=:poisson`: data term, either `:poisson` or `:anscombe`.
- `anscombe_offset=3/8`: the offset `β` of the `:anscombe` loss, in units of
  the measured counts (the data are not normalized by `map_cj`).
- `gamma=0.0`: nonnegative regularization strength.
- `epsilon=nothing`: positive Frobenius smoothing; defaults to
  `sqrt(eps(eltype(measured)))`.
- `weights=nothing`: one nonnegative weight per dimension, with the same
  meaning as in `GR`, `Tikhonov`, `TV` and `TH`: `weights[d]` multiplies the
  second difference along dimension `d`. For sampling distances `S_d` use
  `weights[d] = 1 / S_d^2`. The Good Laplacian is `sum_d weights[d] * Δ²_d`;
  in the Hessian, the diagonal entry `(d, d)` is scaled by `weights[d]` and
  the mixed entry `(d, e)` by `sqrt(weights[d] * weights[e])`, which matches
  `TH`. `nothing` means all ones.
- `background=0`: nonnegative Poisson background.
- `iterations=60`: number of conjugate-gradient iterations.
- `newton_iterations=3`: Newton sub-iterations in each line search.
- `saddle_escape=1.0`: strength of the saddle-escape term added to the search
  direction. With `object = x.^2` the gradient `2x .* g_f` vanishes at
  `x == 0` even where the object-space gradient `g_f` asks the voxel to grow;
  such points are saddles with curvature `2g_f < 0` along the voxel. For
  voxels with `object <= escape_threshold * maximum(object)` and `g_f < 0`
  the direction gets an extra component
  `saddle_escape * rms(direction) * sqrt(g_f / minimum(g_f))`, which the exact
  polynomial line search handles like any other direction. `0` disables it.
- `escape_threshold=1e-4`: relative object level below which voxels are
  considered stuck.
- `padding=0.0`: fractional padding per side, matching `deconvolution`.
  Positive values pad the optimization domain and crop the returned object
  back to the measured size; negative values retain the padded output.
- `initial=nothing`: initial object estimate (scalar or measured-/padded-size array);
  the optimization starts at its nonnegative square root.
- `conv_dims=1:ndims(measured)`: dimensions over which to convolve.
- `normalize_psf=true`: normalize the PSF to unit sum.
- `callback=nothing`: optional function `callback(iteration, object, model)`
  called after every iteration with the current reconstruction `x.^2` and
  forward model (without background). Returning `true` stops the iteration
  early (the current reconstruction is returned).

The returned array is the reconstructed object `x.^2`. Unlike the
historical C++ implementation, the signed variable is not folded with
`abs` after an update: doing so changes the Good/Laplacian penalty while
leaving the data term unchanged. Elementwise operations use fused
broadcasts, and the finite-difference operators run through
KernelAbstractions on the input array's backend.
"""
function map_cj(measured::AbstractArray, psf::AbstractArray;
               penalty=:good,
               gamma=0.0,
               loss=:poisson,
               anscombe_offset=3 / 8,
               epsilon=nothing,
               weights=nothing,
               background=0.0,
               iterations=60,
               newton_iterations=3,
               saddle_escape=1.0,
               escape_threshold=1e-4,
               initial=nothing,
               conv_dims=1:ndims(measured),
               normalize_psf=true,
               padding=0.0,
               callback=nothing)
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
    isfinite(anscombe_offset) && anscombe_offset > 0 ||
        throw(ArgumentError("`anscombe_offset` must be positive and finite"))
    isfinite(gamma) && gamma >= 0 ||
        throw(ArgumentError("`gamma` must be nonnegative and finite"))
    isfinite(background) && background >= 0 ||
        throw(ArgumentError("`background` must be nonnegative and finite"))
    iterations isa Integer && iterations >= 0 ||
        throw(ArgumentError("`iterations` must be a nonnegative integer"))
    newton_iterations isa Integer && newton_iterations >= 0 ||
        throw(ArgumentError("`newton_iterations` must be a nonnegative integer"))
    isfinite(saddle_escape) && saddle_escape >= 0 ||
        throw(ArgumentError("`saddle_escape` must be nonnegative and finite"))
    isfinite(escape_threshold) && escape_threshold >= 0 ||
        throw(ArgumentError("`escape_threshold` must be nonnegative and finite"))

    T = promote_type(float(eltype(measured)), float(eltype(psf)))
    measured_data = T.(measured)
    any((.!isfinite.(measured_data)) .| (measured_data .< zero(T))) &&
        throw(ArgumentError("Poisson measurements must be nonnegative and finite"))
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
            pad = next_fast_fft_size(max(4, 2 * round(Int,
                size(measured, d) * abs(padding))))
            size(measured, d) + pad
        end
    end
    padded = any(size_padded[d] != size(measured, d) for d in 1:N)
    data = similar(measured_data, T, size_padded)
    fill!(data, zero(T))
    mask = padded ? similar(data) : nothing
    if padded
        fill!(mask, zero(T))
        center_set!(data, measured_data)
        measured_mask = similar(measured_data)
        fill!(measured_mask, one(T))
        center_set!(mask, measured_mask)
    else
        copyto!(data, measured_data)
    end

    psf_full = similar(data, T, size(data))
    fill!(psf_full, zero(T))
    center_set!(psf_full, fftshift(psf_data, conv_dims))
    psf_full = ifftshift(psf_full, conv_dims)
    otf, convolve = plan_conv(data, psf_full, conv_dims)
    otf_conj = conj.(otf)

    eps_mu = eps(T)
    eps_hessian = epsilon === nothing ? sqrt(eps(T)) : T(epsilon)
    isfinite(eps_hessian) && eps_hessian > 0 ||
        throw(ArgumentError("`epsilon` must be positive and finite"))
    dim_weights = weights === nothing ? ntuple(_ -> one(T), N) : Tuple(T.(weights))
    length(dim_weights) == N && all(w -> isfinite(w) && w >= 0, dim_weights) ||
        throw(ArgumentError("`weights` must contain one nonnegative finite value per dimension"))
    offset = T(anscombe_offset)
    sqrt_data = loss === :anscombe ? sqrt.(data .+ offset) : nothing

    x = similar(data)
    if initial === nothing
        fill!(x, sqrt(max(mean(measured_data) - T(background), eps_mu)))
    elseif initial isa Number
        isfinite(initial) && initial >= 0 ||
            throw(ArgumentError("`initial` must be nonnegative and finite"))
        fill!(x, sqrt(T(initial)))
    else
        size(initial) == size(measured_data) || size(initial) == size(data) ||
            throw(ArgumentError("array-valued `initial` must have the measured or padded size"))
        any((.!isfinite.(initial)) .| (initial .< 0)) &&
            throw(ArgumentError("`initial` must be nonnegative and finite"))
        initial_data = similar(data, T, size(initial))
        copyto!(initial_data, initial)
        @. initial_data = sqrt(initial_data)
        if size(initial_data) == size(data)
            copyto!(x, initial_data)
        else
            fill!(x, sqrt(max(mean(measured_data) - T(background), eps_mu)))
            center_set!(x, initial_data)
        end
    end

    object = similar(data)
    object .= abs2.(x)
    model = similar(data)
    copyto!(model, convolve(object, otf))

    gradient = similar(data)
    previous_gradient = similar(data)
    direction = similar(data)
    residual = similar(data)
    l_x = similar(data)
    l_direction = penalty === :good ? similar(data) : nothing
    m1 = similar(data)
    m2 = similar(data)
    direction_sq = similar(data)
    cross_term = similar(data)
    line_mu = similar(data)
    line_mu1 = similar(data)
    line_mu2 = similar(data)
    line_work = similar(data)
    dot_partials = similar(data, T, cld(length(data), 256))

    hessian_pairs, hessian_weights = _mappr_hessian_pairs(N)
    hessian_weights = T.(hessian_weights)
    hessian_a = penalty === :hessian_frobenius ?
        [similar(data) for _ in eachindex(hessian_pairs)] : nothing
    sigma = penalty === :hessian_frobenius ?
        ntuple(_ -> similar(data), 5) : nothing
    hessian_b = penalty === :hessian_frobenius ? similar(data) : nothing
    hessian_c = penalty === :hessian_frobenius ? similar(data) : nothing
    hessian_r = penalty === :hessian_frobenius ? similar(data) : nothing
    hessian_q = penalty === :hessian_frobenius ? similar(data) : nothing
    hessian_grad = penalty === :hessian_frobenius ? similar(data) : nothing

    if penalty === :hessian_frobenius
        for k in eachindex(hessian_pairs)
            _mappr_hessian!(hessian_a[k], object, hessian_pairs[k]..., dim_weights)
        end
    end

    good_l0 = 0.0

    for iteration in 1:iterations
        if loss === :poisson
            @. residual = one(T) - data / max(model + T(background), eps_mu)
        else
            @. residual = one(T) - sqrt_data /
                sqrt(max(model + T(background), zero(T)) + offset)
        end
        if mask !== nothing
            @. residual *= mask
        end
        copyto!(gradient, convolve(residual, otf_conj))

        # line_work holds half the diagonal curvature d²Psi/dx_i² at x_i = 0,
        # i.e. the object-space gradient g_f for the square-mapped penalty,
        # until the line search reuses the buffer.
        if penalty === :good
            _mappr_laplacian!(l_x, x, dim_weights)
            good_curvature = 8 * T(gamma) * sum(dim_weights)
            @. line_work = gradient + good_curvature
            @. gradient = 2 * x * gradient - 8 * T(gamma) * l_x
            good_l0 = _map_cj_dot(x, l_x, dot_partials)
        else
            @. hessian_r = T(eps_hessian)^2
            for k in eachindex(hessian_a)
                a = hessian_a[k]
                w = hessian_weights[k]
                @. hessian_r += w * a * a
            end
            @. hessian_r = sqrt(hessian_r)
            fill!(hessian_grad, zero(T))
            for k in eachindex(hessian_a)
                a = hessian_a[k]
                w = hessian_weights[k]
                @. hessian_q = w * a / hessian_r
                _mappr_add_hessian_adjoint!(hessian_grad, hessian_q,
                                             hessian_pairs[k]..., dim_weights)
            end
            @. line_work = gradient + T(gamma) * hessian_grad
            @. gradient = 2 * x * line_work
        end

        if iteration == 1
            @. direction = -gradient
        else
            denom = _map_cj_dot(previous_gradient, previous_gradient,
                                dot_partials)
            beta = iszero(denom) ? zero(T) :
                   _map_cj_dot(gradient, gradient, dot_partials) / denom
            @. direction = beta * direction - gradient
            if _map_cj_dot(gradient, direction, dot_partials) >= 0
                @. direction = -gradient
            end
        end

        copyto!(previous_gradient, gradient)
        if saddle_escape > 0
            _mappr_saddle_escape!(direction, x, object, line_work, line_mu,
                                  T(saddle_escape), T(escape_threshold),
                                  dot_partials)
        end
        _mappr_convolve!(m1, cross_term, x, direction, convolve, otf)
        _mappr_convolve!(m2, direction_sq, direction, direction, convolve, otf)

        if penalty === :hessian_frobenius
            @. cross_term = 2 * x * direction
            @. direction_sq = abs2(direction)
            fill!(sigma[1], T(eps_hessian)^2)
            for j in 2:5
                fill!(sigma[j], zero(T))
            end
            for k in eachindex(hessian_a)
                d, e = hessian_pairs[k]
                _mappr_hessian!(hessian_b, cross_term, d, e, dim_weights)
                _mappr_hessian!(hessian_c, direction_sq, d, e, dim_weights)
                a = hessian_a[k]
                w = hessian_weights[k]
                @. sigma[2] += 2 * w * a * hessian_b
                @. sigma[3] += w * (hessian_b * hessian_b + 2 * a * hessian_c)
                @. sigma[4] += 2 * w * hessian_b * hessian_c
                @. sigma[5] += w * hessian_c * hessian_c
            end
        end

        if penalty === :good
            _mappr_laplacian!(l_direction, direction, dim_weights)
            d_ld = _map_cj_dot(direction, l_direction, dot_partials)
            d_lx = _map_cj_dot(direction, l_x, dot_partials)
            alpha = _mappr_newton_step(model, m1, m2, data, T(background),
                                       penalty, T(gamma), newton_iterations,
                                       good_l0, d_lx, d_ld, sigma, mask,
                                       line_mu, line_mu1, line_mu2, line_work;
                                       loss, sqrt_data, beta=offset)
        else
            alpha = _mappr_newton_step(model, m1, m2, data, T(background),
                                       penalty, T(gamma), newton_iterations,
                                       0.0, 0.0, 0.0, sigma, mask,
                                       line_mu, line_mu1, line_mu2, line_work;
                                       loss, sqrt_data, beta=offset)
        end

        if penalty === :hessian_frobenius
            @. cross_term = 2 * x * direction
            @. direction_sq = abs2(direction)
            alpha_sq = alpha * alpha
            for k in eachindex(hessian_a)
                d, e = hessian_pairs[k]
                _mappr_hessian!(hessian_b, cross_term, d, e, dim_weights)
                _mappr_hessian!(hessian_c, direction_sq, d, e, dim_weights)
                @. hessian_a[k] += alpha * hessian_b + alpha_sq * hessian_c
            end
        end

        @. x += alpha * direction
        @. object = abs2(x)
        @. model += 2 * alpha * m1 + alpha * alpha * m2
        if callback !== nothing &&
           callback(iteration, padded && padding > 0 ?
                    center_extract(object, size(measured_data)) : object,
                    padded && padding > 0 ?
                    center_extract(model, size(measured_data)) : model) === true
            break
        end
    end

    return padded && padding > 0 ? center_extract(object, size(measured_data)) :
           object
end

# Adds a negative-curvature component to `direction` at voxels that sit (almost)
# at x = 0 although the object-space gradient `g_f` pulls them upwards. Along
# the line these voxels change as (x + αd)^2 ≈ α^2 d^2, so the line function
# stays polynomial and the exact line search remains valid.
function _mappr_saddle_escape!(direction, x, object, g_f, pull, strength,
                               threshold, dot_partials)
    T = eltype(direction)
    level = threshold * maximum(object)
    @. pull = ifelse((object <= level) & (g_f < 0), -g_f, zero(T))
    max_pull = maximum(pull)
    max_pull > 0 || return direction
    rms_direction = sqrt(_map_cj_dot(direction, direction, dot_partials) /
                         length(direction))
    rms_direction > 0 || return direction
    scale = strength * rms_direction
    @. direction += ifelse(x < 0, -scale, scale) * sqrt(pull / max_pull)
    return direction
end

function _mappr_convolve!(out, tmp, x, d, convolve, otf)
    @. tmp = x * d
    copyto!(out, convolve(tmp, otf))
    return out
end

const MAP_CJ_REDUCTION_WG = 256

@kernel unsafe_indices = true function _map_cj_dot_kernel!(partials, a, b, total)
    n = @uniform prod(@groupsize())
    levels = @uniform trailing_zeros(n)
    idx = @index(Global, Linear)
    local_idx = @index(Local, Linear)
    group_idx = @index(Group, Linear)
    value = idx <= total ? @inbounds(a[idx] * b[idx]) : zero(eltype(a))
    shared = @localmem eltype(a) (n,)
    shared[local_idx] = value
    @synchronize()
    for level in 1:levels
        stride = n >> level
        if local_idx <= stride
            shared[local_idx] += shared[local_idx + stride]
        end
        @synchronize()
    end
    if local_idx == 1
        partials[group_idx] = shared[1]
    end
end

function _map_cj_dot(a, b, partials)
    nblocks = cld(length(a), MAP_CJ_REDUCTION_WG)
    backend = KernelAbstractions.get_backend(a)
    _map_cj_dot_kernel!(backend, MAP_CJ_REDUCTION_WG)(partials, a, b, length(a);
        ndrange=(nblocks * MAP_CJ_REDUCTION_WG,))
    return sum(Array(view(partials, 1:nblocks)))
end

function _mappr_hessian_pairs(N)
    pairs = Tuple{Int,Int}[]
    weights = Float64[]
    for d in 1:N
        push!(pairs, (d, d))
        push!(weights, 1.0)
    end
    for d in 1:N-1, e in d+1:N
        push!(pairs, (d, e))
        push!(weights, 2.0)
    end
    return pairs, weights
end

@kernel function _mappr_hessian_kernel!(out, x, d, e, weights)
    I = @index(Global, Cartesian)
    N = ndims(x)
    sz = size(x)
    if d == e
        ip = ntuple(k -> k == d ? min(I[k] + 1, sz[k]) : I[k], Val(N))
        im = ntuple(k -> k == d ? max(I[k] - 1, 1) : I[k], Val(N))
        scale = weights[d]
        @inbounds out[I] = (x[ip...] - 2 * x[I] + x[im...]) * scale
    elseif I[d] < sz[d] && I[e] < sz[e]
        id = ntuple(k -> k == d ? I[k] + 1 : I[k], Val(N))
        ie = ntuple(k -> k == e ? I[k] + 1 : I[k], Val(N))
        ide = ntuple(k -> k == d || k == e ? I[k] + 1 : I[k], Val(N))
        scale = sqrt(weights[d] * weights[e])
        @inbounds out[I] = (x[ide...] - x[id...] - x[ie...] + x[I]) * scale
    else
        @inbounds out[I] = zero(eltype(out))
    end
end

@kernel function _mappr_hessian_adjoint_kernel!(out, x, d, e, weights)
    I = @index(Global, Cartesian)
    N = ndims(x)
    sz = size(x)
    v = zero(eltype(out))
    if d == e
        n = sz[d]
        if n > 1
            scale = weights[d]
            v = (I[d] == 1 || I[d] == n ? -one(eltype(out)) :
                 -2 * one(eltype(out))) * x[I] * scale
            if I[d] > 1
                im = ntuple(k -> k == d ? I[k] - 1 : I[k], Val(N))
                @inbounds v += x[im...] * scale
            end
            if I[d] < n
                ip = ntuple(k -> k == d ? I[k] + 1 : I[k], Val(N))
                @inbounds v += x[ip...] * scale
            end
        end
    else
        scale = sqrt(weights[d] * weights[e])
        if I[d] < sz[d] && I[e] < sz[e]
            @inbounds v += x[I] * scale
        end
        if I[d] > 1 && I[e] < sz[e]
            id = ntuple(k -> k == d ? I[k] - 1 : I[k], Val(N))
            @inbounds v -= x[id...] * scale
        end
        if I[d] < sz[d] && I[e] > 1
            ie = ntuple(k -> k == e ? I[k] - 1 : I[k], Val(N))
            @inbounds v -= x[ie...] * scale
        end
        if I[d] > 1 && I[e] > 1
            ide = ntuple(k -> k == d || k == e ? I[k] - 1 : I[k], Val(N))
            @inbounds v += x[ide...] * scale
        end
    end
    @inbounds out[I] += v
end

@kernel function _mappr_laplacian_kernel!(out, x, weights)
    I = @index(Global, Cartesian)
    N = ndims(x)
    sz = size(x)
    v = zero(eltype(out))
    for d in 1:N
        ip = ntuple(k -> k == d ? min(I[k] + 1, sz[k]) : I[k], Val(N))
        im = ntuple(k -> k == d ? max(I[k] - 1, 1) : I[k], Val(N))
        @inbounds v += weights[d] * (x[ip...] - 2 * x[I] + x[im...])
    end
    @inbounds out[I] = v
end

function _mappr_hessian!(out, x, d, e, weights)
    backend = KernelAbstractions.get_backend(x)
    _mappr_hessian_kernel!(backend)(out, x, d, e, weights; ndrange=size(x))
    return out
end

function _mappr_add_hessian_adjoint!(out, x, d, e, weights)
    backend = KernelAbstractions.get_backend(x)
    _mappr_hessian_adjoint_kernel!(backend)(out, x, d, e, weights;
                                            ndrange=size(x))
    return out
end

function _mappr_laplacian!(out, x, weights=ntuple(_ -> one(eltype(x)), ndims(x)))
    backend = KernelAbstractions.get_backend(x)
    _mappr_laplacian_kernel!(backend)(out, x, weights; ndrange=size(x))
    return out
end

function _mappr_line_eval(alpha, model, m1, m2, data, background, penalty,
                          gamma, good_l0, d_lx, d_ld, sigma, mask, line_mu,
                          line_mu1, line_mu2, line_work;
                          loss=:poisson, sqrt_data=nothing, beta=3 / 8)
    alpha2 = alpha * alpha
    floor_mu = eps(eltype(data))
    @. line_mu1 = ifelse(model + 2 * alpha * m1 + alpha2 * m2 +
                         background > floor_mu,
                         2 * (m1 + alpha * m2), zero(eltype(data)))
    @. line_mu2 = ifelse(model + 2 * alpha * m1 + alpha2 * m2 +
                         background > floor_mu,
                         2 * m2, zero(eltype(data)))
    @. line_mu = max(model + 2 * alpha * m1 + alpha2 * m2 +
                     background, floor_mu)
    if loss === :poisson
        phi, dphi, d2phi = _mappr_poisson_line(data, mask, line_mu, line_mu1,
                                               line_mu2, line_work)
    else
        sqrt_g = sqrt_data === nothing ? sqrt.(data .+ beta) : sqrt_data
        phi, dphi, d2phi = _mappr_anscombe_line(sqrt_g, beta, mask, line_mu,
                                                line_mu1, line_mu2, line_work)
    end

    if penalty === :good
        phi -= 4 * gamma * (good_l0 + 2 * alpha * d_lx + alpha2 * d_ld)
        dphi -= 8 * gamma * (d_lx + alpha * d_ld)
        d2phi -= 8 * gamma * d_ld
    else
        s0, s1, s2, s3, s4 = sigma
        @. line_mu = max(s0 + alpha * (s1 + alpha *
            (s2 + alpha * (s3 + alpha * s4))), eps(eltype(data))^2)
        @. line_mu1 = s1 + alpha * (2 * s2 + alpha *
            (3 * s3 + 4 * alpha * s4))
        @. line_mu2 = 2 * s2 + alpha * (6 * s3 + 12 * alpha * s4)
        @. line_work = gamma * sqrt(line_mu)
        phi += sum(line_work)
        @. line_work = gamma * line_mu1 / (2 * sqrt(line_mu))
        dphi += sum(line_work)
        @. line_work = gamma * (line_mu2 / (2 * sqrt(line_mu)) -
                                 line_mu1^2 / (4 * line_mu * sqrt(line_mu)))
        d2phi += sum(line_work)
    end
    return phi, dphi, d2phi
end

# Poisson negative log-likelihood sum(μ - g log μ) and its first two
# derivatives along the line, given μ, μ' and μ'' per voxel.
function _mappr_poisson_line(data, mask, mu, mu1, mu2, work)
    if mask === nothing
        @. work = mu - data * log(mu)
    else
        @. work = mask * (mu - data * log(mu))
    end
    phi = sum(work)
    if mask === nothing
        @. work = (1 - data / mu) * mu1
    else
        @. work = mask * (1 - data / mu) * mu1
    end
    dphi = sum(work)
    if mask === nothing
        @. work = (1 - data / mu) * mu2 + data * (mu1 / mu)^2
    else
        @. work = mask * ((1 - data / mu) * mu2 + data * (mu1 / mu)^2)
    end
    return phi, dphi, sum(work)
end

# Anscombe loss sum((sqrt(μ + β) - sqrt(g + β))^2) along the line, with
# M = μ + β and sqrt_g = sqrt(g + β):
#   phi'  = sum((1 - sqrt_g / sqrt(M)) μ')
#   phi'' = sum((1 - sqrt_g / sqrt(M)) μ'' + sqrt_g μ'^2 / (2 M sqrt(M)))
function _mappr_anscombe_line(sqrt_g, beta, mask, mu, mu1, mu2, work)
    if mask === nothing
        @. work = abs2(sqrt(mu + beta) - sqrt_g)
    else
        @. work = mask * abs2(sqrt(mu + beta) - sqrt_g)
    end
    phi = sum(work)
    if mask === nothing
        @. work = (1 - sqrt_g / sqrt(mu + beta)) * mu1
    else
        @. work = mask * (1 - sqrt_g / sqrt(mu + beta)) * mu1
    end
    dphi = sum(work)
    if mask === nothing
        @. work = (1 - sqrt_g / sqrt(mu + beta)) * mu2 +
                  sqrt_g * mu1^2 / (2 * (mu + beta) * sqrt(mu + beta))
    else
        @. work = mask * ((1 - sqrt_g / sqrt(mu + beta)) * mu2 +
                  sqrt_g * mu1^2 / (2 * (mu + beta) * sqrt(mu + beta)))
    end
    return phi, dphi, sum(work)
end

function _mappr_newton_step(model, m1, m2, data, background, penalty,
                            gamma, newton_iterations, good_l0, d_lx, d_ld,
                            sigma, mask, line_mu, line_mu1, line_mu2, line_work;
                            line_kw...)
    phi0, _, _ = _mappr_line_eval(0.0, model, m1, m2, data, background,
                                  penalty, gamma, good_l0, d_lx, d_ld, sigma,
                                  mask, line_mu, line_mu1, line_mu2, line_work; line_kw...)
    alpha = one(eltype(data))
    phi, _, _ = _mappr_line_eval(alpha, model, m1, m2, data, background,
                                 penalty, gamma, good_l0, d_lx, d_ld, sigma,
                                 mask, line_mu, line_mu1, line_mu2, line_work; line_kw...)
    while (!isfinite(phi) || phi >= phi0) && alpha > eps(eltype(data))
        alpha *= 0.5
        phi, _, _ = _mappr_line_eval(alpha, model, m1, m2, data, background,
                                     penalty, gamma, good_l0, d_lx, d_ld, sigma,
                                     mask, line_mu, line_mu1, line_mu2, line_work; line_kw...)
    end
    if !isfinite(phi) || phi >= phi0
        return 0.0
    end

    best_alpha = alpha
    best_phi = phi
    for _ in 1:newton_iterations
        _, dphi, d2phi = _mappr_line_eval(alpha, model, m1, m2, data,
                                          background, penalty, gamma,
                                          good_l0, d_lx, d_ld, sigma,
                                          mask, line_mu, line_mu1, line_mu2,
                                          line_work; line_kw...)
        candidate = isfinite(d2phi) && d2phi > 0 ?
                    alpha - dphi / d2phi : alpha / 2
        if !isfinite(candidate) || candidate <= 0
            candidate = alpha / 2
        end
        candidate_phi, _, _ = _mappr_line_eval(candidate, model, m1, m2,
                                               data, background, penalty,
                                               gamma, good_l0, d_lx, d_ld,
                                               sigma, mask, line_mu, line_mu1,
                                               line_mu2, line_work; line_kw...)
        while (!isfinite(candidate_phi) || candidate_phi >= best_phi) &&
              candidate > eps(eltype(data))
            candidate *= 0.5
            candidate_phi, _, _ = _mappr_line_eval(candidate, model, m1, m2,
                                                   data, background, penalty,
                                                   gamma, good_l0, d_lx, d_ld,
                                                   sigma, mask, line_mu, line_mu1,
                                                   line_mu2, line_work; line_kw...)
        end
        if isfinite(candidate_phi) && candidate_phi < best_phi
            alpha = candidate
            best_alpha = candidate
            best_phi = candidate_phi
        else
            break
        end
    end
    return best_alpha
end
