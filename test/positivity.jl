using DeconvOptim: PoissonContinued, AnscombeContinued, stage_weights, stage_iterations
using Optim

@testset "Positivity" begin
    fd(f, x; h=1e-6) = [(f(x .+ h .* (eachindex(x) .== i)) - f(x .- h .* (eachindex(x) .== i))) / 2h for i in eachindex(x)]

    @testset "penalty shapes and gradients" begin
        rng = MersenneTwister(1)
        x = randn(rng, 50) .* 0.3
        for mode in ("softplus", "huber", "quadratic")
            p = Positivity(mode=mode, σ=0.1)
            @test Zygote.gradient(p, x)[1] ≈ fd(p, x) atol = 1e-6
        end
        @test Positivity(mode="huber")([0.0, 1.0, 5.0]) == 0
        @test Positivity(mode="quadratic")([-2.0, 3.0]) ≈ 4
        @test Positivity(mode="huber", σ=0.1)([-1.0]) ≈ 1 - 0.05
        @test_throws ArgumentError Positivity(mode="foo")
        @test stage_weights(Positivity(weight=8, ramp=2, stages=3)) ≈ [2, 4, 8]
        @test stage_iterations(10, 3) == [3, 3, 4]
        @test stage_iterations(2, 3) == [1, 1]
    end

    @testset "continued losses" begin
        rng = MersenneTwister(2)
        μ = collect(range(-0.5, 2, length=30)); y = abs.(randn(rng, 30)) .* 2
        for l in (PoissonContinued(0.1), AnscombeContinued(0.2, 0.1))
            @test Zygote.gradient(m -> l(m, y), μ)[1] ≈ fd(m -> l(m, y), μ) atol = 1e-5
        end
        # identical to the original losses above the threshold
        μp = 0.5 .+ rand(rng, 20); yp = rand(rng, 20)
        @test PoissonContinued(0.1)(μp, yp) ≈ poisson_aux(copy(μp), yp)
        @test AnscombeContinued(0.2, 0.1)(μp, yp) ≈ anscombe_aux(copy(μp), yp, b=0.2)
    end

    rng = MersenneTwister(3)
    sz = 48
    obj = zeros(Float32, sz, sz)
    for _ in 1:15
        i, j = rand(rng, 4:sz-4), rand(rng, 4:sz-4); obj[i, j] += 100
    end
    psf = Float32.(ifftshift(exp.(-[((i - sz÷2 - 1)^2 + (j - sz÷2 - 1)^2) / 6 for i in 1:sz, j in 1:sz])))
    psf ./= sum(psf)
    img = max.(conv(obj, psf), 0f0)
    img = Float32.(max.(round.(img .+ sqrt.(img) .* randn(rng, Float32, size(img))), 0f0))

    @testset "deconvolution with a positivity penalty ($lname)" for (lname, loss) in
            (("Poisson", Poisson()), ("Anscombe", Anscombe(0.01)), ("Gauss", Gauss()))
        r, _ = deconvolution(img, psf; loss=loss, regularizer=TH(), λ=0.002, iterations=60,
                             mapping=nothing, positivity=Positivity())
        @test all(isfinite, r)
        @test all(>=(0), r)
        # fits the data like the reconstruction with the x² mapping (forward models agree)
        rm, _ = deconvolution(img, psf; loss=loss, regularizer=TH(), λ=0.002, iterations=60)
        # (the penalty path converges more slowly on this zero-background data, hence 0.9)
        @test cor(vec(conv(r, psf)), vec(conv(rm, psf))) > 0.9
        r2, _ = deconvolution(img, psf; loss=loss, regularizer=TH(), λ=0.002, iterations=60,
                              mapping=nothing, positivity=Positivity(clip=false))
        @test minimum(r2) > -0.05 * maximum(r2)
    end

    @testset "ramp: progress and stop" begin
        calls = Ref(0)
        opts = Optim.Options(iterations=60, callback=os -> (calls[] += 1; calls[] >= 4))
        deconvolution(img, psf; regularizer=TV(), opt_options=opts, mapping=nothing, positivity=Positivity())
        @test calls[] == 4    # a stop request ends the whole ramp
        # ADMM: positivity is exact (projection), the penalty only implies x ≥ 0
        r, _ = deconvolution(img, psf; regularizer=TV(), opt=ADMM(iterations=20), mapping=nothing,
                             positivity=Positivity())
        @test all(>=(0), r)
    end
end
