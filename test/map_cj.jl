using Test
using DeconvOptim
using Random
using LinearAlgebra
using FFTW, Statistics

@testset "MAP-CJ Hessian operators and adjoints" begin
    x = randn(6, 7)
    y = randn(6, 7)
    weights = (1.0, 0.35)
    pairs, _ = DeconvOptim._mappr_hessian_pairs(ndims(x))
    dx = similar(x)
    dty = similar(x)
    for (d, e) in pairs
        DeconvOptim._mappr_hessian!(dx, x, d, e, weights)
        fill!(dty, 0)
        DeconvOptim._mappr_add_hessian_adjoint!(dty, y, d, e, weights)
        @test dot(dx, y) ≈ dot(x, dty) atol=1e-10
    end
end

@testset "MAP-CJ analytic Newton line derivatives" begin
    rng = MersenneTwister(12)
    data = rand(rng, 12) .+ 0.3
    model = rand(rng, 12) .+ 1.0
    m1 = randn(rng, 12) .* 0.05
    m2 = rand(rng, 12) .* 0.01
    line_buffers = ntuple(_ -> similar(data), 4)
    h = 1e-4
    for penalty in (:good, :hessian_frobenius)
        sigma = nothing
        if penalty === :hessian_frobenius
            sigma = (fill(1.5, 12), rand(rng, 12) .* 0.1,
                     rand(rng, 12) .* 0.1, rand(rng, 12) .* 0.02,
                     rand(rng, 12) .* 0.01)
        end
        args = (model, m1, m2, data, 0.2, penalty, 0.7, 0.3, 0.08,
                -0.04, sigma, nothing, line_buffers...)
        phi, dphi, d2phi = DeconvOptim._mappr_line_eval(0.23, args...)
        phi_plus = DeconvOptim._mappr_line_eval(0.23 + h, args...)[1]
        phi_minus = DeconvOptim._mappr_line_eval(0.23 - h, args...)[1]
        @test isfinite(phi)
        @test dphi ≈ (phi_plus - phi_minus) / (2h) rtol=2e-5 atol=2e-7
        @test d2phi ≈ (phi_plus - 2phi + phi_minus) / h^2 rtol=3e-3 atol=3e-5
    end
    sigma = (fill(1.5, 12), rand(rng, 12) .* 0.1, rand(rng, 12) .* 0.1,
             rand(rng, 12) .* 0.02, rand(rng, 12) .* 0.01)
    args = (model, m1, m2, data, 0.2, :hessian_frobenius, 0.7, 0.0, 0.0, 0.0,
            sigma, nothing, line_buffers...)
    kw = (loss=:anscombe, beta=3 / 8)
    phi, dphi, d2phi = DeconvOptim._mappr_line_eval(0.23, args...; kw...)
    phi_plus = DeconvOptim._mappr_line_eval(0.23 + h, args...; kw...)[1]
    phi_minus = DeconvOptim._mappr_line_eval(0.23 - h, args...; kw...)[1]
    mu = model .+ 2 * 0.23 .* m1 .+ 0.23^2 .* m2 .+ 0.2
    s = sum(sigma[p] .* 0.23^(p - 1) for p in 1:5)
    @test phi ≈ sum(abs2, sqrt.(mu .+ 3 / 8) .- sqrt.(data .+ 3 / 8)) +
                0.7 * sum(sqrt.(s))
    @test dphi ≈ (phi_plus - phi_minus) / (2h) rtol=2e-5 atol=2e-7
    @test d2phi ≈ (phi_plus - 2phi + phi_minus) / h^2 rtol=3e-3 atol=3e-5

end

@testset "MAP-CJ Frobenius line coefficients" begin
    # s(α) from the cached coefficients must equal the directly computed
    # ε² + Σ_k w_k [D_k (x + αd)²]² (regression: σ0 once lacked Σ w a²)
    rng = MersenneTwister(7)
    x = rand(rng, 7, 8) .+ 0.2
    d = randn(rng, 7, 8)
    weights = (1.0, 0.6)
    epsilon = 0.03
    pairs, hw = DeconvOptim._mappr_hessian_pairs(2)
    a = [DeconvOptim._mappr_hessian!(similar(x), x .^ 2, p..., weights)
         for p in pairs]
    sigma = ntuple(_ -> similar(x), 5)
    buffers = ntuple(_ -> similar(x), 4)
    DeconvOptim._mappr_frobenius_coefficients!(sigma, a, x, d, buffers...,
                                               pairs, hw, weights, epsilon)
    h = similar(x)
    for alpha in (0.0, 0.3, -0.7)
        direct = fill(epsilon^2, size(x))
        for (k, p) in enumerate(pairs)
            DeconvOptim._mappr_hessian!(h, (x .+ alpha .* d) .^ 2, p..., weights)
            direct .+= hw[k] .* h .^ 2
        end
        @test sum(sigma[j] .* alpha^(j - 1) for j in 1:5) ≈ direct
    end
end

@testset "MAP-CJ weighted Laplacian" begin
    # weights[d] multiplies the second difference along d, as in GR/Tikhonov
    x = [3.0 * i^2 + 5.0 * j^2 for i in 1:6, j in 1:7]
    lx = similar(x)
    DeconvOptim._mappr_laplacian!(lx, x, (0.5, 2.0))
    @test lx[3, 4] ≈ 0.5 * 6 + 2.0 * 10
    y = randn(6, 7)
    ly = similar(y)
    DeconvOptim._mappr_laplacian!(ly, y, (0.5, 2.0))
    @test dot(lx, y) ≈ dot(x, ly)
end

@testset "MAP-CJ padding" begin
    measured = reshape(Float32[1, 2, 1, 3, 2, 1, 1, 2, 4], 3, 3)
    psf = zeros(Float32, 3, 3)
    psf[2, 2] = 1
    callback_shapes = Tuple{Tuple,Tuple}[]
    callback = (iteration, object, model) -> begin
        push!(callback_shapes, (size(object), size(model)))
        false
    end

    cropped = map_cj(measured, psf; penalty=:good, gamma=0.01,
                     iterations=2, padding=0.1, callback)
    @test size(cropped) == size(measured)
    @test all(shape -> shape == (size(measured), size(measured)),
              callback_shapes)

    padded = map_cj(measured, psf; penalty=:hessian_frobenius, gamma=0.01,
                    iterations=1, padding=-0.1)
    @test size(padded) == (7, 7)
    @test all(isfinite, padded)

    padded_initial = fill(0.5f0, size(measured))
    initialized = map_cj(measured, psf; iterations=0, padding=0.1,
                         initial=padded_initial)
    @test size(initialized) == size(measured)
    @test_throws ArgumentError map_cj(measured, psf; padding=Inf)
end

@testset "MAP-CJ signed Good penalty and square-mapped Frobenius" begin
    x = [1.0, 1.0, 1.0]
    flipped = [1.0, -1.0, 1.0]
    lx = similar(x)
    lflip = similar(x)
    DeconvOptim._mappr_laplacian!(lx, x)
    DeconvOptim._mappr_laplacian!(lflip, flipped)
    @test x .^ 2 == flipped .^ 2
    @test dot(x, lx) != dot(flipped, lflip)

    measured = reshape(Float32[1, 2, 1, 3, 2, 1, 1, 2, 4], 3, 3)
    psf = zeros(Float32, 3, 3)
    psf[2, 2] = 1
    for penalty in (:good, :hessian_frobenius)
        result = map_cj(measured, psf; penalty=penalty, gamma=0.01,
                       iterations=2, newton_iterations=2)
        @test size(result) == size(measured)
        @test all(isfinite, result)
        @test all(result .>= 0)
    end
    @test_throws ArgumentError map_cj(measured, psf; newton_iterations=-1)
end

@testset "MAP-CJ saddle escape from x == 0" begin
    n = 24
    obj = [2 + 50 * exp(-((i - 12)^2 + (j - 12)^2) / 10) for i in 1:n, j in 1:n]
    psf = [exp(-((i - 1 - n ÷ 2)^2 + (j - 1 - n ÷ 2)^2) / 4) for i in 1:n, j in 1:n]
    psf = ifftshift(psf ./ sum(psf))
    measured = real.(ifft(fft(obj) .* fft(psf))) .+ 1
    initial = fill(mean(measured) - 1, n, n)
    initial[8:16, 8:16] .= 0
    for penalty in (:good, :hessian_frobenius)
        stuck = map_cj(measured, psf; penalty, gamma=1e-3, background=1.0,
                       epsilon=1.0, iterations=30, initial, saddle_escape=0)
        freed = map_cj(measured, psf; penalty, gamma=1e-3, background=1.0,
                       epsilon=1.0, iterations=30, initial)
        # Good's -8γLx term slowly frees the patch from its edges;
        # with the square-mapped Frobenius penalty it stays exactly zero.
        penalty === :hessian_frobenius && @test all(iszero, stuck[8:16, 8:16])
        @test all(>(1), freed[10:14, 10:14])
        @test sum(abs2, freed .- obj) < 0.1 * sum(abs2, stuck .- obj)
    end
    @test_throws ArgumentError map_cj(measured, psf; saddle_escape=-1)
end

@testset "MAP-CJ Anscombe loss" begin
    n = 24
    obj = [2 + 50 * exp(-((i - 12)^2 + (j - 12)^2) / 10) for i in 1:n, j in 1:n]
    psf = [exp(-((i - 1 - n ÷ 2)^2 + (j - 1 - n ÷ 2)^2) / 4) for i in 1:n, j in 1:n]
    psf = ifftshift(psf ./ sum(psf))
    measured = real.(ifft(fft(obj) .* fft(psf))) .+ 1
    start_error = sum(abs2, mean(measured) - 1 .- obj)
    for penalty in (:good, :hessian_frobenius)
        objective = Float64[]
        result = map_cj(measured, psf; penalty, loss=:anscombe, gamma=1e-3,
                        background=1.0, epsilon=1.0, iterations=30,
                        weights=(1.0, 0.5),
                        callback=(it, o, m) -> (push!(objective,
                            sum(abs2, sqrt.(m .+ 1 .+ 3 / 8) .-
                                      sqrt.(measured .+ 3 / 8))); false))
        @test all(isfinite, result)
        @test all(result .>= 0)
        @test sum(abs2, result .- obj) < 0.1 * start_error
        @test objective[end] < objective[1]
    end
    @test_throws ArgumentError map_cj(measured, psf; loss=:gaussian)
    @test_throws ArgumentError map_cj(measured, psf; weights=(1.0,))
end

@testset "MAP-CJ Frobenius line search matches the true objective" begin
    # Regression: the quartic s(α) must include the current Hessian in σ0;
    # otherwise the line search optimizes the wrong function and stalls for
    # small epsilon while the true objective increases.
    n = 20
    rng = MersenneTwister(5)
    obj = [2 + 40 * exp(-((i - 10)^2 + (j - 9)^2) / 6) for i in 1:n, j in 1:n]
    psf = [exp(-((i - 1 - n ÷ 2)^2 + (j - 1 - n ÷ 2)^2) / 3) for i in 1:n, j in 1:n]
    psf = ifftshift(psf ./ sum(psf))
    measured = real.(ifft(fft(obj) .* fft(psf))) .+ 1 .+ 0.5 .* randn(rng, n, n)
    measured = max.(measured, 0)
    gamma, epsilon, weights = 0.05, 1e-2, (1.0, 0.7)
    pairs, hw = DeconvOptim._mappr_hessian_pairs(2)
    h = zeros(n, n)
    function objective(o, m)
        s = fill(epsilon^2, n, n)
        for (k, (d, e)) in enumerate(pairs)
            DeconvOptim._mappr_hessian!(h, o, d, e, weights)
            s .+= hw[k] .* h .^ 2
        end
        mu = m .+ 1
        return sum(mu .- measured .* log.(mu)) + gamma * sum(sqrt.(s))
    end
    for cg_update in (:fletcher_reeves, :polak_ribiere)
        values = Float64[]
        map_cj(measured, psf; penalty=:hessian_frobenius, gamma, epsilon,
               weights, background=1.0, iterations=40, cg_update,
               callback=(it, o, m) -> (push!(values, objective(o, m)); false))
        @test all(diff(values) .<= 1e-9 .* abs.(values[1:end-1]))
        @test values[end] < values[1] - 1
    end
    @test_throws ArgumentError map_cj(measured, psf; cg_update=:hestenes)
end
