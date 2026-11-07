using DeconvOptim: prox, PoissonProx, GaussProx, AnscombeProx, apply!, admm_reg

# a progress recorder that always fails (ADMM must keep running)
struct _Thrower end
DeconvOptim.record_progress!(::_Thrower, args...) = error("progress failed")

@testset "ADMM" begin
    @testset "proximal operators are minimizers" begin
        rng = MersenneTwister(1)
        losses = [(PoissonProx(), (μ, y) -> μ - y * log(μ), 0.0),
                  (GaussProx(), (μ, y) -> (μ - y)^2, -Inf),
                  (AnscombeProx(0.375), (μ, y) -> (sqrt(y + 0.375) - sqrt(μ + 0.375))^2, -0.375)]
        for (l, f, lo) in losses, _ in 1:200
            v = 3randn(rng); y = abs(3randn(rng)); ρ = exp(2randn(rng)); bg = 0.5rand(rng)
            u = l isa AnscombeProx ? prox(l, v, y, ρ, bg, 8) : prox(l, v, y, ρ, bg)
            obj(u) = f(u + bg, y) + ρ / 2 * (u - v)^2
            for uu in (u + 1e-4, u - 1e-4)
                uu + bg > lo && @test obj(u) <= obj(uu) + 1e-12
            end
        end
    end

    @testset "stencil adjoints" begin
        rng = MersenneTwister(2)
        # explicit num_dims: constructing TV()/TH()/… without it generates code for all ranks (slow)
        for reg in (TV(num_dims=3), TV(num_dims=3, mode="central", step=2), TH(num_dims=3), HS(),
                    Tikhonov(num_dims=3), Tikhonov(num_dims=3, mode="spatial_grad_square"),
                    TV(num_dims=3, weights=[1, 1, 0.3]))
            blocks, tik = admm_reg(reg, 3, Float64)
            for s in vcat([b.stencils for b in blocks]..., tik)
                a = randn(rng, 9, 8, 3); b = randn(rng, 9, 8, 3)
                @test dot(apply!(similar(a), s, a), b) ≈ dot(a, apply!(similar(b), s', b))
            end
        end
    end

    # test object, PSF and noisy data
    rng = MersenneTwister(3)
    sz = 64
    obj = zeros(Float32, sz, sz)
    for _ in 1:20
        i, j = rand(rng, 6:sz-6), rand(rng, 6:sz-6)
        obj[i-2:i+2, j-2:j+2] .+= 100
    end
    psf = Float32.(ifftshift(exp.(-[((i - sz÷2 - 1)^2 + (j - sz÷2 - 1)^2) / 8 for i in 1:sz, j in 1:sz])))
    psf ./= sum(psf)
    img = max.(conv(obj, psf) .+ 5, 0f0)
    img = max.(img .+ sqrt.(img) .* randn(rng, Float32, size(img)), 0f0)
    nrm = mean(img)
    function objective(x, loss, reg, λ)
        xn = x ./ nrm
        μ = conv(xn, psf)
        v = sum(loss(copy(μ), img ./ nrm, similar(μ)))
        isnothing(reg) ? v : v + λ * Base.invokelatest(reg, xn)
    end

    # L-BFGS reference for three distinct prox types (group shrinkage, Schatten, Newton prox)
    @testset "reaches the L-BFGS objective ($name)" for (name, loss, reg, λ) in [
            ("Poisson+TV", Poisson(), TV(num_dims=2), 0.05), ("Poisson+HS", Poisson(), HS(), 0.05),
            ("Anscombe+TV", Anscombe(3 / 8 / nrm), TV(num_dims=2), 0.005)]
        r1, _ = deconvolution(img, psf; loss=loss, regularizer=reg, λ=λ, iterations=150)
        r2, o = deconvolution(img, psf; loss=loss, regularizer=reg, λ=λ, opt=ADMM(iterations=300))
        @test o isa DeconvOptim.ADMMResult
        @test size(r2) == size(img)
        @test all(>=(0), r2)
        @test all(isfinite, r2)
        F1, F2 = objective(r1, loss, reg, λ), objective(r2, loss, reg, λ)
        # both near the same optimum (TV in the objective has non-periodic boundaries)
        @test F2 <= F1 + 1e-3 * abs(F1)
    end

    # the other regularizers/losses: ADMM must clearly decrease the objective
    @testset "decreases the objective ($name)" for (name, loss, reg, λ) in [
            ("Poisson+TH", Poisson(), TH(num_dims=2), 0.05), ("Poisson+Tikhonov", Poisson(), Tikhonov(num_dims=2), 0.05),
            ("Gauss+TV", Gauss(), TV(num_dims=2), 0.005), ("Poisson+none", Poisson(), nothing, 0.05)]
        r, o = deconvolution(img, psf; loss=loss, regularizer=reg, λ=λ, opt=ADMM(iterations=40))
        @test all(>=(0), r)
        @test all(isfinite, r)
        F0 = objective(fill(mean(img), size(img)), loss, reg, λ)
        Fopt = objective(r, loss, reg, λ)
        @test Fopt < F0
    end

    @testset "padding, tolerance, 3D" begin
        r, o = deconvolution(img, psf; regularizer=TV(num_dims=2), opt=ADMM(iterations=20), padding=0.1)
        @test size(r) == size(img)
        @test size(o.minimizer, 1) > size(img, 1)
        # callback stop; an erroring progress recorder does not abort the solver
        r, o = deconvolution(img, psf; regularizer=TV(num_dims=2), progress=_Thrower(),
                             opt=ADMM(iterations=50, callback=(it, x) -> it >= 7))
        @test o.iterations == 7
        @test all(>=(0), r)
        r, o = deconvolution(img, psf; regularizer=TV(num_dims=2), opt=ADMM(tol=1e-2, iterations=500))
        @test o.converged
        @test o.iterations < 500
        # a z-constant stack with a 2D PSF must give the 2D result (the iterates are
        # identical in every iteration, so a few iterations suffice)
        img3 = repeat(img[1:32, 1:32], 1, 1, 3)
        p2 = psf[[1:16; end-15:end], [1:16; end-15:end]]; p2 ./= sum(p2)
        r3, _ = deconvolution(img3, p2; regularizer=TV(num_dims=3), opt=ADMM(iterations=50))
        r2, _ = deconvolution(img[1:32, 1:32], p2; regularizer=TV(num_dims=2), opt=ADMM(iterations=50))
        @test r3[:, :, 2] ≈ r2 rtol = 1e-3
        @test r3[:, :, 1] ≈ r3[:, :, 3] rtol = 1e-4
    end

    @testset "unsupported combinations" begin
        @test_throws ArgumentError deconvolution(img, psf; regularizer=GR(num_dims=2), opt=ADMM())
        @test_throws ArgumentError deconvolution(img, psf; loss=ScaledGauss(), opt=ADMM())
        @test_throws ArgumentError deconvolution(img, psf; regularizer=HS(p=2), opt=ADMM())
    end
end
