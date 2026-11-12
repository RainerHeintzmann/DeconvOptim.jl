using DeconvOptim: _pix_prox, _cp_proj, cp_tgv, _tgv_norm2, apply!, admm_reg

@testset "L1, Huber and TGV" begin
    rng = MersenneTwister(5)
    @testset "Huber proximal maps" begin
        for _ in 1:300
            v = Tuple(2randn(rng, 3)); κ = exp(randn(rng)); δ = rand(rng) * 2
            z = _pix_prox(Val(:group), v, κ, δ)
            # optimality of argmin κ H_δ(‖z‖) + ½‖z − v‖²: z − v + κ ∇H_δ(z) = 0
            nz = sqrt(sum(abs2, z))
            gH = nz <= δ ? z ./ δ : z ./ nz
            nz > 1e-12 && @test all(abs.(z .- v .+ κ .* gH) .< 1e-9)
            nz <= 1e-12 && @test sqrt(sum(abs2, v)) <= κ + δ + 1e-12
            # Chambolle–Pock dual map = Moreau identity of the same prox (F = λH_δ, at step σ)
            λ = κ; σ = exp(randn(rng))
            q = v
            p = _cp_proj(Val(:group), q, λ, 1 / (1 + σ * δ / λ))
            @test all(p .≈ q .- σ .* _pix_prox(Val(:group), q ./ σ, λ / σ, δ))
        end
        # ϵ of TV/TH sets δ = √ϵ
        @test only(admm_reg(TV(num_dims=2, ϵ=0.04), 2, Float64)[1]).δ ≈ 0.2
        @test only(admm_reg(TH(num_dims=2, ϵ=0.04), 2, Float64)[1]).δ ≈ 0.2
    end

    # test data: beads on a ramp
    sz = 48
    obj = Float32[5 + 0.2i for i in 1:sz, j in 1:sz]
    for _ in 1:15
        obj[rand(rng, 4:sz-4), rand(rng, 4:sz-4)] += 60
    end
    psf = Float32.(ifftshift(exp.(-[((i - sz÷2 - 1)^2 + (j - sz÷2 - 1)^2) / 6 for i in 1:sz, j in 1:sz])))
    psf ./= sum(psf)
    img = conv(obj, psf) .+ 0.5f0 .* randn(rng, Float32, sz, sz)
    nrm = mean(img)
    objective(x, reg, λ) = (xn = x ./ nrm; μ = conv(xn, psf); sum(abs2, μ .- img ./ nrm) + λ * reg(xn))

    @testset "L1 (Gauss): ADMM, Chambolle–Pock and L-BFGS agree" begin
        λ = 0.01
        rA, _ = deconvolution(img, psf; loss=Gauss(), regularizer=L1(), λ=λ, opt=ADMM(iterations=1000))
        rC, _ = deconvolution(img, psf; loss=Gauss(), regularizer=L1(), λ=λ, opt=ChambollePock(iterations=2000))
        rL, _ = deconvolution(img, psf; loss=Gauss(), regularizer=L1(), λ=λ, iterations=300)
        oA, oC, oL = objective(rA, L1(), λ), objective(rC, L1(), λ), objective(rL, L1(), λ)
        @test oA ≈ oL rtol = 1e-3
        @test oC ≈ oL rtol = 1e-3
        @test minimum(rA) >= 0 && minimum(rC) >= 0
        # stronger λ: more exact zeros than without the prior
        r0, _ = deconvolution(img .- 5, psf; loss=Gauss(), regularizer=nothing, opt=ChambollePock(iterations=300))
        r1, _ = deconvolution(img .- 5, psf; loss=Gauss(), regularizer=L1(), λ=0.5, opt=ChambollePock(iterations=300))
        @test count(iszero, r1) > count(iszero, r0)
    end

    @testset "L1 with Poisson and no background only rescales" begin
        imgp = img .- minimum(img) .+ 1
        r0, _ = deconvolution(imgp, psf; regularizer=nothing, opt=ADMM(iterations=200))
        r1, _ = (@test_logs (:warn,) match_mode=:any deconvolution(imgp, psf; regularizer=L1(), λ=0.25, opt=ADMM(iterations=200)))
        @test sum(r1) / sum(r0) ≈ 1 / 1.25 rtol = 1e-3
    end

    @testset "Huber TV: ADMM and Chambolle–Pock agree" begin
        reg = TV(num_dims=2, ϵ=0.01)
        rA, _ = deconvolution(img, psf; loss=Gauss(), regularizer=reg, λ=0.01, opt=ADMM(iterations=1500))
        rC, _ = deconvolution(img, psf; loss=Gauss(), regularizer=reg, λ=0.01, opt=ChambollePock(iterations=3000))
        @test rA ≈ rC rtol = 2e-2
    end

    @testset "TGV" begin
        tg = cp_tgv(TGV(num_dims=2), 2, Float64, (sz, sz))
        @test length(tg.grad) == 2 && length(tg.sym) == 3
        # adjoint of K = [∇ −I; 0 ε] (random test vectors)
        x = randn(rng, sz, sz); w = [randn(rng, sz, sz) for _ in 1:2]
        pg = [randn(rng, sz, sz) for _ in 1:2]; pe = [randn(rng, sz, sz) for _ in 1:3]
        Kg = [apply!(similar(x), s, x) .- w[d] for (d, s) in enumerate(tg.grad)]
        Ke = [sum(apply!(similar(x), st, w[j]) for (j, st) in comp) for comp in tg.sym]
        ax = sum(apply!(similar(x), s', pg[d]) for (d, s) in enumerate(tg.grad))
        aw = [-pg[j] for j in 1:2]
        for (k, comp) in enumerate(tg.sym), (j, st) in comp
            aw[j] .+= apply!(similar(x), st', pe[k])
        end
        @test sum(dot.(Kg, pg)) + sum(dot.(Ke, pe)) ≈ dot(x, ax) + sum(dot.(w, aw)) rtol = 1e-10
        # ‖K‖² bound ≥ power iteration of KᵀK (without H)
        bound = _tgv_norm2(zeros(sz ÷ 2 + 1, sz), zeros(sz ÷ 2 + 1, sz), tg, (sz, sz), Float64)
        ev = 0.0
        for _ in 1:200
            Kg = [apply!(similar(x), s, x) .- w[d] for (d, s) in enumerate(tg.grad)]
            Ke = [sum(apply!(similar(x), st, w[j]) for (j, st) in comp) for comp in tg.sym]
            ax = sum(apply!(similar(x), s', Kg[d]) for (d, s) in enumerate(tg.grad))
            aw = [-Kg[j] for j in 1:2]
            for (k, comp) in enumerate(tg.sym), (j, st) in comp
                aw[j] .+= apply!(similar(x), st', Ke[k])
            end
            nn = sqrt(sum(abs2, ax) + sum(sum.(abs2, aw)))
            ev = nn / sqrt(sum(abs2, x) + sum(sum.(abs2, w)))
            x = ax ./ nn; w = aw ./ nn
        end
        @test ev <= bound
        @test bound <= 1.2ev

        rT, _ = deconvolution(img, psf; loss=Gauss(), regularizer=TV(num_dims=2), λ=0.05, opt=ChambollePock(iterations=1500))
        rG, o = deconvolution(img, psf; loss=Gauss(), regularizer=TGV(num_dims=2), λ=0.05, opt=ChambollePock(iterations=1500))
        @test o.history[end].r_rel < 1e-4
        # large ratio: ε(w) = 0 leaves only a constant w (periodic grid), i.e. TGV → TV up to a global
        # slope; without a global slope in the object (beads only) TGV → TV
        imgb = conv(obj .- Float32[0.2i for i in 1:sz, j in 1:sz], psf) .+ 0.5f0 .* randn(rng, Float32, sz, sz)
        rTb, _ = deconvolution(imgb, psf; loss=Gauss(), regularizer=TV(num_dims=2), λ=0.05, opt=ChambollePock(iterations=1500))
        rG∞, _ = deconvolution(imgb, psf; loss=Gauss(), regularizer=TGV(num_dims=2, ratio=1000), λ=0.05,
                               opt=ChambollePock(iterations=1500))
        @test rG∞ ≈ rTb rtol = 2e-3
        # no staircasing on the ramp: TV leaves flat steps, TGV does not
        flat(r) = count(<(0.02), abs.(diff(r[8:40, 10])))
        @test flat(rT) > flat(rG)
        @test flat(rG) <= 1
        # only Chambolle–Pock supports TGV
        @test_throws ArgumentError deconvolution(img, psf; regularizer=TGV(), opt=ADMM())
        @test_throws ArgumentError deconvolution(img, psf; regularizer=TGV(), iterations=2)
    end
end
