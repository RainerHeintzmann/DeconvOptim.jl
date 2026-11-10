# `padding` as one number (as before) or one value per dimension (src/padding.jl)
@testset "Padding per dimension" begin
    Random.seed!(7)
    img = 50f0 .* rand(Float32, 24, 20, 12) .+ 5f0
    psf = zeros(Float32, 24, 20, 12)
    psf[1, 1, 1] = 4; psf[2, 1, 1] = 1; psf[end, 1, 1] = 1; psf[1, 1, 2] = 1; psf[1, 1, end] = 1
    psf ./= sum(psf)

    @testset "helpers" begin
        @test DeconvOptim._padded_size((24, 20, 12), 0) == (24, 20, 12)
        @test DeconvOptim._padded_size((24, 20, 12), 0.1) == DeconvOptim._padded_size((24, 20, 12), (0.1, 0.1, 0.1))
        ps = DeconvOptim._padded_size((24, 20, 12), (0.0, 0.1, 0.5))
        @test ps[1] == 24 && ps[2] > 20 && ps[3] > 12
        @test DeconvOptim._padded_size((24, 1, 12), (0.1, 0.1, 0.1))[2] == 1     # no padding of singletons
        @test DeconvOptim._result_size((24, 20, 12), (28, 24, 20), (0.1, -0.1, 0.25)) == (24, 24, 12)
        @test_throws ArgumentError DeconvOptim._check_padding((0.1, 0.1), 3)
        @test_throws ArgumentError DeconvOptim._check_padding((0.1, NaN, 0.1), 3)
        @test_throws ArgumentError DeconvOptim._check_padding("0.1", 3)
    end

    @testset "deconvolution" begin
        r1, _ = deconvolution(img, psf; regularizer=nothing, iterations=5, padding=0.1)
        r2, _ = deconvolution(img, psf; regularizer=nothing, iterations=5, padding=(0.1, 0.1, 0.1))
        @test size(r1) == size(img)
        @test r1 ≈ r2                                     # a tuple of equal values is the number
        r3, _ = deconvolution(img, psf; regularizer=nothing, iterations=5, padding=[0.05, 0.05, 0.25])
        @test size(r3) == size(img) && all(isfinite, r3)
        sp = DeconvOptim._padded_size(size(img), (0.1, -0.1, 0.25))
        r4, _ = deconvolution(img, psf; regularizer=nothing, iterations=5, padding=(0.1, -0.1, 0.25))
        @test size(r4) == (24, sp[2], 12)                 # negative: that dimension stays padded
        r5, _ = deconvolution(img, psf; regularizer=TV(num_dims=3), iterations=5, padding=(0.05, 0.05, 0.25),
                              opt=ADMM(iterations=5))
        @test size(r5) == size(img) && all(isfinite, r5)
        @test_throws ArgumentError deconvolution(img, psf; iterations=2, padding=(0.1, 0.1))
    end

    @testset "map_cj, map_cj_lean, map_cj_exact" begin
        for f in (map_cj, map_cj_lean, map_cj_exact)
            a = f(img, psf; iterations=3, padding=0.1)
            b = f(img, psf; iterations=3, padding=(0.1, 0.1, 0.1))
            @test size(a) == size(img)
            @test a ≈ b
            c = f(img, psf; iterations=3, padding=(0.05, 0.05, 0.25))
            @test size(c) == size(img) && all(isfinite, c)
            sp = DeconvOptim._padded_size(size(img), (0.1, 0.1, -0.25))
            d = f(img, psf; iterations=3, padding=(0.1, 0.1, -0.25))
            @test size(d) == (24, 20, sp[3])
        end
    end
end
