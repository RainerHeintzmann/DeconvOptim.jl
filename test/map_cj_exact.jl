using Test
using DeconvOptim
using Random
using LinearAlgebra
using FFTW, Statistics

function _exact_test_problem(n=(16, 15); seed=1)
    rng = MersenneTwister(seed)
    obj = [2 + 40 * exp(-((i - 7)^2 + (j - 6)^2) / 5) for i in 1:n[1], j in 1:n[2]]
    psf = [exp(-((i - 1 - n[1] ÷ 2)^2 + (j - 1 - n[2] ÷ 2)^2) / 3)
           for i in 1:n[1], j in 1:n[2]]
    psf = ifftshift(psf ./ sum(psf))
    measured = max.(real.(ifft(fft(obj) .* fft(psf))) .+ 1 .+ randn(rng, n...), 0)
    return obj, psf, measured, rng
end

_exact_context(measured, psf; loss, penalty, gamma, epsilon=0.5, weights=nothing,
               padding=0.0, noise_weights=nothing) =
    DeconvOptim._lean_setup(measured, psf; penalty, loss, anscombe_offset=3 / 8,
        mapping=:square, mapping_scale=1.0, gamma, epsilon, weights,
        background=1.0, initial=nothing, conv_dims=1:2, normalize_psf=true,
        padding, noise_weights)

# the functional, computed directly from x
function _exact_objective(c, x)
    mu, tmp = similar(x), similar(x)
    DeconvOptim._lean_model!(c, mu, x)
    return DeconvOptim._lean_objective(c, x, mu, tmp)
end

@testset "MAP-CJ exact: closed-form quartic minimum" begin
    rng = MersenneTwister(2)
    for _ in 1:200
        q = (randn(rng), randn(rng), 3 * randn(rng), randn(rng), rand(rng) + 0.01)
        a = DeconvOptim._quartic_argmin(q)
        grid = range(-20, 20; length=400_001)
        @test DeconvOptim._quartic(q, a) <=
              minimum(DeconvOptim._quartic(q, t) for t in grid) + 1e-8
    end
    @test DeconvOptim._cubic_real_roots(1.0, -6.0, 11.0, -6.0) |> sort ≈ [1, 2, 3]
end

@testset "MAP-CJ exact: line function equals the objective" begin
    obj, psf, measured, rng = _exact_test_problem()
    for loss in (:gauss, :poisson, :anscombe), penalty in (:good, :hessian_frobenius),
        padding in (0.0, 0.3)

        c, x = _exact_context(measured, psf; loss, penalty, gamma=0.4,
                              weights=(1.0, 0.6), padding)
        x .= 1 .+ rand(rng, size(x)...)
        d = 0.2 .* randn(rng, size(x)...)
        mu, m1, m2, work = similar(x), similar(x), similar(x), similar(x)
        DeconvOptim._lean_model!(c, mu, x)
        m1 .= x .* d; c.conv!(m1, m1, false)
        m2 .= d .* d; c.conv!(m2, m2, false)
        partials = similar(x, 3 * cld(length(x), DeconvOptim._EXACT_WG))
        g = c.gamma
        data_line(a) = loss === :gauss ?
            (q = DeconvOptim._gauss_data_coefficients(c, mu, m1, m2);
             (DeconvOptim._quartic(q, a), DeconvOptim._quartic1(q, a),
              DeconvOptim._quartic2(q, a))) :
            DeconvOptim._pointwise_data_line(c, mu, m1, m2, a)
        pen_line(a) = if penalty === :good
            xlx, dlx, dld = DeconvOptim._good_line_scalars(c, x, d, work)
            (-4g * (xlx + 2a * dlx + a^2 * dld), -8g * (dlx + a * dld), -8g * dld)
        else
            g .* DeconvOptim._exact_hf_line_sums(c, x, d, a, partials)
        end
        line(a) = data_line(a) .+ pen_line(a)
        h = 1e-5
        for a in (0.0, 0.4)
            v, d1, d2 = line(a)
            @test v ≈ _exact_objective(c, x .+ a .* d) rtol=1e-9
            @test d1 ≈ (line(a + h)[1] - line(a - h)[1]) / (2h) rtol=1e-5
            @test d2 ≈ (line(a + h)[2] - line(a - h)[2]) / (2h) rtol=1e-5
        end
        # the slope at α = 0 is the directional derivative of the gradient
        grad, gf = similar(x), similar(x)
        DeconvOptim._lean_gradient!(c, grad, gf, x, mu)
        @test line(0.0)[2] ≈ dot(grad, d) rtol=1e-8
        if loss === :gauss && penalty === :good
            # MAPGR: closed-form minimizer of the quartic
            xlx, dlx, dld = DeconvOptim._good_line_scalars(c, x, d, work)
            q = DeconvOptim._gauss_data_coefficients(c, mu, m1, m2)
            q = (q[1] - 4g * xlx, q[2] - 8g * dlx, q[3] - 4g * dld, q[4], q[5])
            a = DeconvOptim._quartic_argmin(q)
            @test abs(DeconvOptim._quartic1(q, a)) <= 1e-8 * abs(q[2])
        end
    end
end

@testset "MAP-CJ exact: line search and reconstructions" begin
    obj, psf, measured, _ = _exact_test_problem((22, 20); seed=3)
    start_error = sum(abs2, mean(measured) - 1 .- obj)
    for loss in (:gauss, :poisson, :anscombe), penalty in (:good, :hessian_frobenius)
        gamma = loss === :gauss ? 0.05 : 0.02
        c, _ = _exact_context(measured, psf; loss, penalty, gamma, epsilon=1.0)
        values = Float64[]
        exact = map_cj_exact(measured, psf; loss, penalty, gamma, epsilon=1.0,
            background=1.0, iterations=150, saddle_escape=0,
            callback=(it, o, m) -> (push!(values, _exact_objective(c, sqrt.(o))); false))
        @test all(isfinite, exact) && all(exact .>= 0)
        @test sum(abs2, exact .- obj) < 0.3 * start_error
        # the true objective decreases monotonically
        @test all(diff(values) .<= 1e-9 .* abs.(values[1:end-1]))
        # at least as low as the one-step Newton version with twice the work
        lean = map_cj_lean(measured, psf; loss, penalty, gamma, epsilon=1.0,
            background=1.0, iterations=300, saddle_escape=0)
        lean_value = _exact_objective(c, sqrt.(lean))
        @test _exact_objective(c, sqrt.(exact)) <= lean_value + 1e-4 * abs(lean_value)
        if loss !== :gauss
            # same functional as map_cj, whose line search is exact as well
            ref = map_cj(measured, psf; loss, penalty, gamma, epsilon=1.0,
                         background=1.0, iterations=150, saddle_escape=0)
            @test _exact_objective(c, sqrt.(exact)) ≈
                  _exact_objective(c, sqrt.(ref)) rtol=1e-3
        end
    end
    padded = map_cj_exact(measured, psf; loss=:anscombe, penalty=:hessian_frobenius,
                          gamma=0.02, iterations=5, padding=0.2)
    @test size(padded) == size(measured) && all(isfinite, padded)
    @test_throws ArgumentError map_cj_exact(measured, psf; penalty=:tv)
    @test_throws ArgumentError map_cj_exact(measured, psf; loss=:laplace)
    @test_throws ArgumentError map_cj_exact(measured, psf; newton_iterations=0)
end


@testset "MAP-CJ exact: per-pixel noise weights (MAPGR / MAPGH)" begin
    obj, psf, measured, rng = _exact_test_problem((18, 17); seed=5)
    omega = 0.2 .+ 2 .* rand(rng, size(measured)...)
    for penalty in (:good, :hessian_frobenius), padding in (0.0, 0.3)
        c, x = _exact_context(measured, psf; loss=:gauss, penalty, gamma=0.4,
                              weights=(1.0, 0.6), padding, noise_weights=omega)
        x .= 1 .+ rand(rng, size(x)...)
        d = 0.2 .* randn(rng, size(x)...)
        mu, m1, m2, work = similar(x), similar(x), similar(x), similar(x)
        DeconvOptim._lean_model!(c, mu, x)
        m1 .= x .* d; c.conv!(m1, m1, false)
        m2 .= d .* d; c.conv!(m2, m2, false)
        q = DeconvOptim._gauss_data_coefficients(c, mu, m1, m2)
        # the weighted data term along the line equals the objective without penalty
        c0 = merge(c, (gamma=zero(c.gamma),))
        for a in (0.0, 0.4)
            @test DeconvOptim._quartic(q, a) ≈ _exact_objective(c0, x .+ a .* d) rtol=1e-9
        end
        # gradient and curvature match the directional derivatives of the objective
        grad, gf = similar(x), similar(x)
        DeconvOptim._lean_gradient!(c, grad, gf, x, mu)
        h = 1e-5
        phi(a) = _exact_objective(c, x .+ a .* d)
        @test dot(grad, d) ≈ (phi(h) - phi(-h)) / (2h) rtol=1e-5
        d2 = DeconvOptim._lean_curvature!(c, gf, x, d, mu)      # gf = g_f from the gradient
        h2 = 1e-3
        @test d2 ≈ (phi(h2) - 2phi(0.0) + phi(-h2)) / h2^2 rtol=1e-3
    end

    # a constant weight c is the same as the unweighted loss with gamma / c
    for penalty in (:good, :hessian_frobenius)
        a = map_cj_exact(measured, psf; loss=:gauss, penalty, gamma=0.05, epsilon=1.0,
            background=1.0, iterations=40, saddle_escape=0, noise_weights=fill(4.0, size(measured)))
        b = map_cj_exact(measured, psf; loss=:gauss, penalty, gamma=0.05 / 4, epsilon=1.0,
            background=1.0, iterations=40, saddle_escape=0)
        @test a ≈ b rtol=1e-4
    end

    # a pixel with weight 0 has no influence, whatever its value
    w = ones(size(measured)); w[9, 9] = 0
    spoiled = copy(measured); spoiled[9, 9] += 500
    for f in (map_cj_exact, map_cj_lean)
        r1 = f(measured, psf; loss=:gauss, gamma=0.05, background=1.0, iterations=30,
               saddle_escape=0, initial=1.0, noise_weights=w)
        r2 = f(spoiled, psf; loss=:gauss, gamma=0.05, background=1.0, iterations=30,
               saddle_escape=0, initial=1.0, noise_weights=w)
        @test r1 ≈ r2 rtol=1e-6
        # ... while without the weight the outlier does matter
        u2 = f(spoiled, psf; loss=:gauss, gamma=0.05, background=1.0, iterations=30,
               saddle_escape=0, initial=1.0)
        @test !isapprox(r1, u2; rtol=1e-3)
    end

    # weighted reconstruction decreases the weighted objective monotonically
    c, _ = _exact_context(measured, psf; loss=:gauss, penalty=:hessian_frobenius,
                          gamma=0.05, epsilon=1.0, noise_weights=omega)
    values = Float64[]
    rec = map_cj_exact(measured, psf; loss=:gauss, penalty=:hessian_frobenius, gamma=0.05,
        epsilon=1.0, background=1.0, iterations=60, saddle_escape=0, noise_weights=omega,
        callback=(it, o, m) -> (push!(values, _exact_objective(c, sqrt.(o))); false))
    @test all(isfinite, rec) && all(diff(values) .<= 1e-9 .* abs.(values[1:end-1]))

    # argument checks
    @test_throws ArgumentError map_cj_exact(measured, psf; loss=:poisson, noise_weights=omega)
    @test_throws ArgumentError map_cj_exact(measured, psf; loss=:gauss, noise_weights=omega[1:end-1, :])
    @test_throws ArgumentError map_cj_exact(measured, psf; loss=:gauss, noise_weights=-omega)
    @test_throws ArgumentError map_cj_lean(measured, psf; loss=:anscombe, noise_weights=omega)
end
