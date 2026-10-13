export map_cj
import KernelAbstractions
using KernelAbstractions: @kernel, @index

"""
    map_cj(measured, psf; <keyword arguments>)

Poisson maximum-a-posteriori deconvolution using the square mapping
`object = x.^2`, Fletcher--Reeves conjugate gradients, and an analytic
Newton line search.

`penalty=:good` selects MAPPR's Good/Laplacian penalty on the signed
optimization variable `x`. `penalty=:hessian_frobenius` selects the
smoothed Frobenius norm of the Hessian of `x.^2`.

# Keyword arguments
- `penalty=:good`: either `:good` or `:hessian_frobenius`.
- `gamma=0.0`: nonnegative regularization strength.
- `epsilon=nothing`: positive Frobenius smoothing; defaults to
  `sqrt(eps(eltype(measured)))`.
- `spacing=ones(ndims(measured))`: sample spacing for Hessian differences.
- `background=0`: nonnegative Poisson background.
- `iterations=60`: number of conjugate-gradient iterations.
- `newton_iterations=3`: Newton sub-iterations in each line search.
- `initial=nothing`: initial object estimate (scalar or same-size array);
  the optimization starts at its nonnegative square root.
- `conv_dims=1:ndims(measured)`: dimensions over which to convolve.
- `normalize_psf=true`: normalize the PSF to unit sum.

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
               epsilon=nothing,
               spacing=nothing,
               background=0.0,
               iterations=60,
               newton_iterations=3,
               initial=nothing,
               conv_dims=1:ndims(measured),
               normalize_psf=true)
    N = ndims(measured)
    eltype(measured) <: Real && eltype(psf) <: Real ||
        throw(ArgumentError("`measured` and `psf` must contain real-valued data"))
    N == ndims(psf) ||
        throw(ArgumentError("`measured` and `psf` must have the same number of dimensions"))
    all(size(psf, d) <= size(measured, d) for d in 1:N) ||
        throw(ArgumentError("`psf` dimensions must not exceed `measured` dimensions"))
    penalty in (:good, :hessian_frobenius) ||
        throw(ArgumentError("`penalty` must be :good or :hessian_frobenius"))
    isfinite(gamma) && gamma >= 0 ||
        throw(ArgumentError("`gamma` must be nonnegative and finite"))
    isfinite(background) && background >= 0 ||
        throw(ArgumentError("`background` must be nonnegative and finite"))
    iterations isa Integer && iterations >= 0 ||
        throw(ArgumentError("`iterations` must be a nonnegative integer"))
    newton_iterations isa Integer && newton_iterations >= 0 ||
        throw(ArgumentError("`newton_iterations` must be a nonnegative integer"))

    T = promote_type(float(eltype(measured)), float(eltype(psf)))
    data = T.(measured)
    any(v -> !isfinite(v) || v < zero(T), data) &&
        throw(ArgumentError("Poisson measurements must be nonnegative and finite"))
    psf_data = T.(psf)
    all(isfinite, psf_data) ||
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
    spacings = spacing === nothing ? ntuple(_ -> one(T), N) : Tuple(T.(spacing))
    length(spacings) == N && all(s -> isfinite(s) && s > 0, spacings) ||
        throw(ArgumentError("`spacing` must contain one positive finite value per dimension"))

    x = similar(data)
    if initial === nothing
        fill!(x, sqrt(max(mean(data) - T(background), eps_mu)))
    elseif initial isa Number
        isfinite(initial) && initial >= 0 ||
            throw(ArgumentError("`initial` must be nonnegative and finite"))
        fill!(x, sqrt(T(initial)))
    else
        size(initial) == size(data) ||
            throw(ArgumentError("array-valued `initial` must have the same size as `measured`"))
        any(v -> !isfinite(v) || v < 0, initial) &&
            throw(ArgumentError("`initial` must be nonnegative and finite"))
        copyto!(x, initial)
        @. x = sqrt(x)
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
    m1 = similar(data)
    m2 = similar(data)
    direction_sq = similar(data)
    cross_term = similar(data)
    line_mu = similar(data)
    line_mu1 = similar(data)
    line_mu2 = similar(data)
    line_work = similar(data)

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
            _mappr_hessian!(hessian_a[k], object, hessian_pairs[k]..., spacings)
        end
    end

    good_l0 = 0.0
    if penalty === :good
        _mappr_laplacian!(l_x, x)
        good_l0 = dot(x, l_x)
    end

    for iteration in 1:iterations
        @. residual = one(T) - data / max(model + T(background), eps_mu)
        copyto!(gradient, convolve(residual, otf_conj))

        if penalty === :good
            _mappr_laplacian!(l_x, x)
            @. gradient = 2 * x * gradient - 8 * T(gamma) * l_x
            good_l0 = dot(x, l_x)
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
                                             hessian_pairs[k]..., spacings)
            end
            @. gradient = 2 * x * (gradient + T(gamma) * hessian_grad)
        end

        if iteration == 1
            @. direction = -gradient
        else
            denom = dot(previous_gradient, previous_gradient)
            beta = iszero(denom) ? zero(T) : dot(gradient, gradient) / denom
            @. direction = beta * direction - gradient
            if dot(gradient, direction) >= 0
                @. direction = -gradient
            end
        end

        copyto!(previous_gradient, gradient)
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
                _mappr_hessian!(hessian_b, cross_term, d, e, spacings)
                _mappr_hessian!(hessian_c, direction_sq, d, e, spacings)
                a = hessian_a[k]
                w = hessian_weights[k]
                @. sigma[2] += 2 * w * a * hessian_b
                @. sigma[3] += w * (hessian_b * hessian_b + 2 * a * hessian_c)
                @. sigma[4] += 2 * w * hessian_b * hessian_c
                @. sigma[5] += w * hessian_c * hessian_c
            end
        end

        if penalty === :good
            _mappr_laplacian!(l_x, direction)
            d_ld = dot(direction, l_x)
            _mappr_laplacian!(l_x, x)
            d_lx = dot(direction, l_x)
            alpha = _mappr_newton_step(model, m1, m2, data, T(background),
                                       penalty, T(gamma), newton_iterations,
                                       good_l0, d_lx, d_ld, sigma,
                                       line_mu, line_mu1, line_mu2, line_work)
        else
            alpha = _mappr_newton_step(model, m1, m2, data, T(background),
                                       penalty, T(gamma), newton_iterations,
                                       0.0, 0.0, 0.0, sigma,
                                       line_mu, line_mu1, line_mu2, line_work)
        end

        if penalty === :hessian_frobenius
            @. cross_term = 2 * x * direction
            @. direction_sq = abs2(direction)
            alpha_sq = alpha * alpha
            for k in eachindex(hessian_a)
                d, e = hessian_pairs[k]
                _mappr_hessian!(hessian_b, cross_term, d, e, spacings)
                _mappr_hessian!(hessian_c, direction_sq, d, e, spacings)
                @. hessian_a[k] += alpha * hessian_b + alpha_sq * hessian_c
            end
        end

        @. x += alpha * direction
        @. object = abs2(x)
        @. model += 2 * alpha * m1 + alpha * alpha * m2
    end

    return object
end

function _mappr_convolve!(out, tmp, x, d, convolve, otf)
    @. tmp = x * d
    copyto!(out, convolve(tmp, otf))
    return out
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

@kernel function _mappr_hessian_kernel!(out, x, d, e, spacing)
    I = @index(Global, Cartesian)
    N = ndims(x)
    sz = size(x)
    if d == e
        ip = ntuple(k -> k == d ? min(I[k] + 1, sz[k]) : I[k], Val(N))
        im = ntuple(k -> k == d ? max(I[k] - 1, 1) : I[k], Val(N))
        scale = inv(spacing[d] * spacing[d])
        @inbounds out[I] = (x[ip...] - 2 * x[I] + x[im...]) * scale
    elseif I[d] < sz[d] && I[e] < sz[e]
        id = ntuple(k -> k == d ? I[k] + 1 : I[k], Val(N))
        ie = ntuple(k -> k == e ? I[k] + 1 : I[k], Val(N))
        ide = ntuple(k -> k == d || k == e ? I[k] + 1 : I[k], Val(N))
        scale = inv(spacing[d] * spacing[e])
        @inbounds out[I] = (x[ide...] - x[id...] - x[ie...] + x[I]) * scale
    else
        @inbounds out[I] = zero(eltype(out))
    end
end

@kernel function _mappr_hessian_adjoint_kernel!(out, x, d, e, spacing)
    I = @index(Global, Cartesian)
    N = ndims(x)
    sz = size(x)
    v = zero(eltype(out))
    if d == e
        n = sz[d]
        if n > 1
            scale = inv(spacing[d] * spacing[d])
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
        scale = inv(spacing[d] * spacing[e])
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

@kernel function _mappr_laplacian_kernel!(out, x)
    I = @index(Global, Cartesian)
    N = ndims(x)
    sz = size(x)
    v = -2 * N * @inbounds(x[I])
    for d in 1:N
        ip = ntuple(k -> k == d ? min(I[k] + 1, sz[k]) : I[k], Val(N))
        im = ntuple(k -> k == d ? max(I[k] - 1, 1) : I[k], Val(N))
        @inbounds v += x[ip...] + x[im...]
    end
    @inbounds out[I] = v
end

function _mappr_hessian!(out, x, d, e, spacing)
    backend = KernelAbstractions.get_backend(x)
    _mappr_hessian_kernel!(backend)(out, x, d, e, spacing; ndrange=size(x))
    return out
end

function _mappr_add_hessian_adjoint!(out, x, d, e, spacing)
    backend = KernelAbstractions.get_backend(x)
    _mappr_hessian_adjoint_kernel!(backend)(out, x, d, e, spacing;
                                            ndrange=size(x))
    return out
end

function _mappr_laplacian!(out, x)
    backend = KernelAbstractions.get_backend(x)
    _mappr_laplacian_kernel!(backend)(out, x; ndrange=size(x))
    return out
end

function _mappr_line_eval(alpha, model, m1, m2, data, background, penalty,
                          gamma, good_l0, d_lx, d_ld, sigma,
                          line_mu, line_mu1, line_mu2, line_work)
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
    @. line_work = line_mu - data * log(line_mu)
    phi = sum(line_work)
    @. line_work = (1 - data / line_mu) * line_mu1
    dphi = sum(line_work)
    @. line_work = (1 - data / line_mu) * line_mu2 +
                   data * (line_mu1 / line_mu)^2
    d2phi = sum(line_work)

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

function _mappr_newton_step(model, m1, m2, data, background, penalty,
                            gamma, newton_iterations, good_l0, d_lx, d_ld,
                            sigma, line_mu, line_mu1, line_mu2, line_work)
    phi0, _, _ = _mappr_line_eval(0.0, model, m1, m2, data, background,
                                  penalty, gamma, good_l0, d_lx, d_ld, sigma,
                                  line_mu, line_mu1, line_mu2, line_work)
    alpha = one(eltype(data))
    phi, _, _ = _mappr_line_eval(alpha, model, m1, m2, data, background,
                                 penalty, gamma, good_l0, d_lx, d_ld, sigma,
                                 line_mu, line_mu1, line_mu2, line_work)
    while (!isfinite(phi) || phi >= phi0) && alpha > eps(eltype(data))
        alpha *= 0.5
        phi, _, _ = _mappr_line_eval(alpha, model, m1, m2, data, background,
                                     penalty, gamma, good_l0, d_lx, d_ld, sigma,
                                     line_mu, line_mu1, line_mu2, line_work)
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
                                          line_mu, line_mu1, line_mu2, line_work)
        candidate = isfinite(d2phi) && d2phi > 0 ?
                    alpha - dphi / d2phi : alpha / 2
        if !isfinite(candidate) || candidate <= 0
            candidate = alpha / 2
        end
        candidate_phi, _, _ = _mappr_line_eval(candidate, model, m1, m2,
                                               data, background, penalty,
                                               gamma, good_l0, d_lx, d_ld,
                                               sigma, line_mu, line_mu1,
                                               line_mu2, line_work)
        while (!isfinite(candidate_phi) || candidate_phi >= best_phi) &&
              candidate > eps(eltype(data))
            candidate *= 0.5
            candidate_phi, _, _ = _mappr_line_eval(candidate, model, m1, m2,
                                                   data, background, penalty,
                                                   gamma, good_l0, d_lx, d_ld,
                                                   sigma, line_mu, line_mu1,
                                                   line_mu2, line_work)
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
