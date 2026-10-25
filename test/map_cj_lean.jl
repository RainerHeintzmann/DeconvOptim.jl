using Test
using DeconvOptim
using Random
using LinearAlgebra
using FFTW, Statistics

function _lean_test_problem(n=(14, 13); seed=1)
    rng = MersenneTwister(seed)
    obj = [2 + 40 * exp(-((i - 7)^2 + (j - 6)^2) / 5) for i in 1:n[1], j in 1:n[2]]
    psf = [exp(-((i - 1 - n[1] ÷ 2)^2 + (j - 1 - n[2] ÷ 2)^2) / 3)
           for i in 1:n[1], j in 1:n[2]]
    psf = ifftshift(psf ./ sum(psf))
    measured = max.(real.(ifft(fft(obj) .* fft(psf))) .+ 1 .+ randn(rng, n...), 0)
    return obj, psf, measured, rng
end

@testset "MAP-CJ lean kernels match the map_cj operators" begin
    rng = MersenneTwister(3)
    x = randn(rng, 6, 7)
    d = randn(rng, 6, 7)
    weights = (1.0, 0.4)
    eps2 = 0.05^2
    pairs, hw = DeconvOptim._mappr_hessian_pairs(2)
    for m in (DeconvOptim._LeanSquare(), DeconvOptim._LeanContinuousPositive(4.0))
        f = DeconvOptim._f.(Ref(m), x)
        fd = DeconvOptim._df.(Ref(m), x) .* d
        h = [DeconvOptim._mappr_hessian!(similar(x), f, p..., weights) for p in pairs]
        hp = [DeconvOptim._mappr_hessian!(similar(x), fd, p..., weights) for p in pairs]
        s = eps2 .+ sum(hw[k] .* h[k] .^ 2 for k in eachindex(h))
        r = sqrt.(s)
        out = similar(x)
        DeconvOptim._lean_launch!(DeconvOptim._lean_hf_value_kernel!, out, x, m,
                                  Tuple(pairs), Tuple(hw), weights, eps2)
        @test out ≈ r
        grad_ref = zeros(size(x))
        for (k, p) in enumerate(pairs)
            DeconvOptim._mappr_add_hessian_adjoint!(grad_ref, hw[k] .* h[k] ./ r,
                                                    p..., weights)
        end
        fill!(out, 0)
        DeconvOptim._lean_launch!(DeconvOptim._lean_hf_grad_kernel!, out, x, r, m,
                                  Tuple(pairs), Tuple(hw), weights, 0.7)
        @test out ≈ 0.7 .* grad_ref
        cross = sum(hw[k] .* h[k] .* hp[k] for k in eachindex(h))
        slope = sum(hw[k] .* hp[k] .^ 2 for k in eachindex(h))
        DeconvOptim._lean_launch!(DeconvOptim._lean_hf_curv_kernel!, out, x, d, m,
                                  Tuple(pairs), Tuple(hw), weights, eps2)
        @test out ≈ slope ./ r .- cross .^ 2 ./ (s .* r)
    end
    # Good's roughness for f = x^2 is MAPPR's -4 xᵀLx with gradient -8 L x
    m = DeconvOptim._LeanSquare()
    lx = DeconvOptim._mappr_laplacian!(similar(x), x, weights)
    out = similar(x)
    DeconvOptim._lean_launch!(DeconvOptim._lean_good_value_kernel!, out, x, m, weights)
    @test sum(out) ≈ -4 * dot(x, lx)
    fill!(out, 0)
    DeconvOptim._lean_launch!(DeconvOptim._lean_good_grad_kernel!, out, x, m, weights, 1.0)
    @test out ≈ -8 .* lx
end

@testset "MAP-CJ lean exact line derivatives" begin
    obj, psf, measured, rng = _lean_test_problem()
    for mapping in (:square, :continuous_positive),
        penalty in (:good, :hessian_frobenius),
        loss in (:poisson, :anscombe, :gauss)

        c, x = DeconvOptim._lean_setup(measured, psf; penalty, loss,
            anscombe_offset=3 / 8, mapping, mapping_scale=1.3, gamma=0.3,
            epsilon=0.5, weights=(1.0, 0.6), background=1.0, initial=nothing,
            conv_dims=1:2, normalize_psf=true, padding=0.0)
        x .= mapping === :square ? 1 .+ rand(rng, size(x)...) :
                                    randn(rng, size(x)...)
        d = 0.1 .* randn(rng, size(x)...)
        mu, grad, tmp = similar(x), similar(x), similar(x)
        phi(a) = (xa = x .+ a .* d;
                  DeconvOptim._lean_model!(c, mu, xa);
                  DeconvOptim._lean_objective(c, xa, mu, tmp))
        DeconvOptim._lean_model!(c, mu, x)
        DeconvOptim._lean_gradient!(c, grad, tmp, x, mu)
        d1 = dot(grad, d)
        d2 = DeconvOptim._lean_curvature!(c, tmp, x, d, mu)
        h = 1e-4
        p0, pp, pm = phi(0.0), phi(h), phi(-h)
        @test d1 ≈ (pp - pm) / (2h) rtol=1e-5
        @test d2 ≈ (pp - 2p0 + pm) / h^2 rtol=1e-3
    end
end

@testset "MAP-CJ lean reconstructions" begin
    obj, psf, measured, _ = _lean_test_problem((20, 18); seed=4)
    start_error = sum(abs2, mean(measured) - 1 .- obj)
    for penalty in (:good, :hessian_frobenius), loss in (:poisson, :anscombe)
        reference = map_cj(measured, psf; penalty, loss, gamma=1e-3, epsilon=1.0,
                           background=1.0, iterations=40)
        reference_error = sum(abs2, reference .- obj)
        for mapping in (:square, :continuous_positive)
            result = map_cj_lean(measured, psf; mapping, penalty, loss,
                                 gamma=1e-3, epsilon=1.0, background=1.0,
                                 iterations=40)
            @test size(result) == size(measured)
            @test all(isfinite, result)
            @test all(result .>= 0)
            @test sum(abs2, result .- obj) < 0.3 * start_error
            # same objective as map_cj for f = x^2
            mapping === :square &&
                @test sum(abs2, result .- obj) ≈ reference_error rtol=0.1
        end
    end
    padded = map_cj_lean(measured, psf; penalty=:hessian_frobenius, gamma=1e-3,
                         iterations=3, padding=0.1)
    @test size(padded) == size(measured)
    @test_throws ArgumentError map_cj_lean(measured, psf; mapping=:exp)
    @test_throws ArgumentError map_cj_lean(measured, psf; mapping_scale=0)
end

@testset "MAP-CJ lean saddle escape" begin
    obj, psf, measured, _ = _lean_test_problem((24, 24); seed=5)
    initial = fill(mean(measured) - 1, size(measured))
    initial[3:11, 2:10] .= 0
    stuck = map_cj_lean(measured, psf; penalty=:hessian_frobenius, gamma=1e-3,
                        epsilon=1.0, background=1.0, iterations=30, initial,
                        saddle_escape=0)
    freed = map_cj_lean(measured, psf; penalty=:hessian_frobenius, gamma=1e-3,
                        epsilon=1.0, background=1.0, iterations=30, initial)
    @test all(iszero, stuck[3:11, 2:10])
    @test sum(abs2, freed .- obj) < 0.5 * sum(abs2, stuck .- obj)
end

@testset "MAP-CJ lean leaves its inputs untouched" begin
    obj, psf, measured, _ = _lean_test_problem((16, 15); seed=6)
    small_psf = fftshift(psf)[5:12, 4:11]   # centred, smaller than the data
    for (meas, p) in ((measured, psf), (Float32.(measured), psf),
                      (measured, ifftshift(small_psf)), (measured, 3 .* psf))
        meas_copy, p_copy = copy(meas), copy(p)
        for padding in (0.0, 0.2)
            result = map_cj_lean(meas, p; penalty=:hessian_frobenius, gamma=1e-3,
                                 epsilon=1.0, background=1.0, iterations=5, padding)
            @test all(isfinite, result)
            @test meas == meas_copy
            @test p == p_copy
        end
    end
    # PSF normalization on the OTF: a scaled PSF gives the same result
    a = map_cj_lean(measured, psf; iterations=5, background=1.0)
    b = map_cj_lean(measured, 3 .* psf; iterations=5, background=1.0)
    @test a ≈ b
end

@testset "MAP-CJ lean padding border is masked" begin
    obj, psf, measured, rng = _lean_test_problem((14, 13); seed=8)
    for loss in (:poisson, :anscombe, :gauss)
        c, x = DeconvOptim._lean_setup(measured, psf; penalty=:hessian_frobenius,
            loss, anscombe_offset=3 / 8, mapping=:square, mapping_scale=1.0,
            gamma=0.0, epsilon=0.5, weights=nothing, background=1.0,
            initial=nothing, conv_dims=1:2, normalize_psf=true, padding=0.3)
        @test size(c.data) != size(measured)
        x .= 1 .+ rand(rng, size(x)...)
        mu, grad, tmp = similar(x), similar(x), similar(x)
        DeconvOptim._lean_model!(c, mu, x)
        inner = DeconvOptim.center_extract(mu, size(measured))
        direct = loss === :poisson ?
            sum(@. (inner + 1) - measured * log(inner + 1)) :
            loss === :anscombe ?
            sum(@. abs2(sqrt(inner + 1 + 3 / 8) - sqrt(measured + 3 / 8))) :
            sum(@. abs2(inner + 1 - measured))
        @test DeconvOptim._lean_objective(c, x, mu, tmp) ≈ direct
        d = 0.1 .* randn(rng, size(x)...)
        phi(a) = (xa = x .+ a .* d; DeconvOptim._lean_model!(c, mu, xa);
                  DeconvOptim._lean_objective(c, xa, mu, tmp))
        DeconvOptim._lean_model!(c, mu, x)
        DeconvOptim._lean_gradient!(c, grad, tmp, x, mu)
        d1 = dot(grad, d)
        d2 = DeconvOptim._lean_curvature!(c, tmp, x, d, mu)
        h = 1e-4
        p0, pp, pm = phi(0.0), phi(h), phi(-h)
        @test d1 ≈ (pp - pm) / (2h) rtol=1e-5
        @test d2 ≈ (pp - 2p0 + pm) / h^2 rtol=1e-3
    end
end
