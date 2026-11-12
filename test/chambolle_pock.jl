using DeconvOptim: _cp_data, _cp_proj, PoissonProx, GaussProx, AnscombeProx

@testset "Chambolle-Pock" begin
    @testset "dual proximal maps" begin
        rng = MersenneTwister(11)
        # data term: Moreau identity  prox_{σF*}(q) + σ prox_{F/σ}(q/σ) = q, and p = prox_{σF*}(q)
        # maximizes  ⟨p, u⟩ − F(u)  in its optimality condition: p = ∂F(u) at u = prox_{F/σ}(q/σ)
        dF = [(PoissonProx(), (μ, y) -> 1 - y / μ), (GaussProx(), (μ, y) -> 2(μ - y)),
              (AnscombeProx(0.375), (μ, y) -> 1 - sqrt(y + 0.375) / sqrt(μ + 0.375))]
        for (l, df) in dF, _ in 1:200
            q = 3randn(rng); y = abs(3randn(rng)); σ = exp(2randn(rng)); bg = 0.5rand(rng)
            p = _cp_data(l, q, y, σ, bg, 12)
            u = (q - p) / σ                                  # = prox_{F/σ}(q/σ)
            @test p ≈ df(u + bg, y) rtol = 1e-6 atol = 1e-9
        end
        # group norm: projection onto the ball of radius λ
        for _ in 1:100
            v = Tuple(3randn(rng, 3)); λ = rand(rng) + 0.1
            z = _cp_proj(Val(:group), v, λ, 1.0)
            @test sqrt(sum(abs2, z)) <= λ * (1 + 1e-12)
            sqrt(sum(abs2, v)) <= λ && @test all(z .≈ v)
        end
        # Schatten: projection onto the spectral ball = eigenvalue clipping of [[a, c/√2], [c/√2, d]]
        for _ in 1:100
            a, d, c = 2randn(rng), 2randn(rng), 2randn(rng); λ = rand(rng) + 0.1
            za, zd, zc = _cp_proj(Val(:schatten), (a, d, c), λ, 1.0)
            M = [a c/sqrt(2); c/sqrt(2) d]
            E = eigen(Symmetric(M))
            Mp = E.vectors * Diagonal(clamp.(E.values, -λ, λ)) * E.vectors'
            @test [za zc/sqrt(2); zc/sqrt(2) zd] ≈ Mp atol = 1e-10
        end
    end

    # test object, PSF and noisy data (as for ADMM)
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

    @testset "reaches the L-BFGS objective ($name)" for (name, loss, reg, λ) in [
            ("Poisson+TV", Poisson(), TV(num_dims=2), 0.05), ("Poisson+HS", Poisson(), HS(), 0.05),
            ("Poisson+Tikhonov", Poisson(), Tikhonov(num_dims=2), 0.05)]
        r1, _ = deconvolution(img, psf; loss=loss, regularizer=reg, λ=λ, iterations=150)
        r2, o = deconvolution(img, psf; loss=loss, regularizer=reg, λ=λ, opt=ChambollePock(iterations=500))
        @test o isa DeconvOptim.ChambollePockResult
        @test size(r2) == size(img)
        @test all(>=(0), r2)
        @test all(isfinite, r2)
        F1, F2 = objective(r1, loss, reg, λ), objective(r2, loss, reg, λ)
        @test F2 <= F1 + 1e-3 * abs(F1)
    end

    @testset "decreases the objective ($name)" for (name, loss, reg, λ) in [
            ("Poisson+TH", Poisson(), TH(num_dims=2), 0.05), ("Gauss+TV", Gauss(), TV(num_dims=2), 0.005),
            ("Anscombe+TV", Anscombe(3 / 8 / nrm), TV(num_dims=2), 0.005), ("Poisson+none", Poisson(), nothing, 0.05)]
        r, _ = deconvolution(img, psf; loss=loss, regularizer=reg, λ=λ, opt=ChambollePock(iterations=60))
        @test all(>=(0), r)
        @test all(isfinite, r)
        @test objective(r, loss, reg, λ) < objective(fill(mean(img), size(img)), loss, reg, λ)
    end

    @testset "same optimum as ADMM" begin
        rA, _ = deconvolution(img, psf; regularizer=TV(num_dims=2), opt=ADMM(iterations=500))
        rC, _ = deconvolution(img, psf; regularizer=TV(num_dims=2), opt=ChambollePock(iterations=1500))
        @test rC ≈ rA rtol = 2e-2
    end

    @testset "padding, tolerance, callback, 3D" begin
        r, o = deconvolution(img, psf; regularizer=TV(num_dims=2), opt=ChambollePock(iterations=20), padding=0.1)
        @test size(r) == size(img)
        @test size(o.minimizer, 1) > size(img, 1)
        @test all(>=(0), r)
        r, o = deconvolution(img, psf; regularizer=TV(num_dims=2),
                             opt=ChambollePock(iterations=50, callback=(it, x) -> it >= 7))
        @test o.iterations == 7
        r, o = deconvolution(img, psf; regularizer=TV(num_dims=2), opt=ChambollePock(tol=1e-3, iterations=3000))
        @test o.converged
        @test o.iterations < 3000
        img3 = repeat(img[1:32, 1:32], 1, 1, 3)
        p2 = psf[[1:16; end-15:end], [1:16; end-15:end]]; p2 ./= sum(p2)
        # (unlike ADMM the iterates differ: the step sizes depend on ‖K‖, which includes the
        # z-gradient; the minimizer is the same)
        r3, _ = deconvolution(img3, p2; regularizer=TV(num_dims=3), opt=ChambollePock(iterations=1500))
        r2, _ = deconvolution(img[1:32, 1:32], p2; regularizer=TV(num_dims=2), opt=ChambollePock(iterations=1500))
        @test r3[:, :, 2] ≈ r2 rtol = 1e-2
        @test r3[:, :, 1] ≈ r3[:, :, 3] rtol = 1e-4
    end

    @testset "unsupported combinations" begin
        @test_throws ArgumentError deconvolution(img, psf; regularizer=GR(num_dims=2), opt=ChambollePock())
        @test_throws ArgumentError deconvolution(img, psf; loss=ScaledGauss(), opt=ChambollePock())
        @test_throws ArgumentError deconvolution(img, psf; regularizer=HS(p=2), opt=ChambollePock())
    end
end
