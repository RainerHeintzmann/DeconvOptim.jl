using Test
using DeconvOptim
using Random
using LinearAlgebra

@testset "MAP-CJ Hessian operators and adjoints" begin
    x = randn(6, 7)
    y = randn(6, 7)
    spacing = (1.0, 1.7)
    pairs, _ = DeconvOptim._mappr_hessian_pairs(ndims(x))
    dx = similar(x)
    dty = similar(x)
    for (d, e) in pairs
        DeconvOptim._mappr_hessian!(dx, x, d, e, spacing)
        fill!(dty, 0)
        DeconvOptim._mappr_add_hessian_adjoint!(dty, y, d, e, spacing)
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
                -0.04, sigma, line_buffers...)
        phi, dphi, d2phi = DeconvOptim._mappr_line_eval(0.23, args...)
        phi_plus = DeconvOptim._mappr_line_eval(0.23 + h, args...)[1]
        phi_minus = DeconvOptim._mappr_line_eval(0.23 - h, args...)[1]
        @test isfinite(phi)
        @test dphi ≈ (phi_plus - phi_minus) / (2h) rtol=2e-5 atol=2e-7
        @test d2phi ≈ (phi_plus - 2phi + phi_minus) / h^2 rtol=3e-3 atol=3e-5
    end
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
