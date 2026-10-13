export richardson_lucy_iterative

"""
    richardson_lucy_iterative(measured, psf; <keyword arguments>)

Classical iterative Richardson-Lucy iteration scheme for deconvolution.
`measured` is the measured array and `psf` the point spread function.
Converges slower than the optimization approach of `deconvolution`

# Keyword Arguments
- `regularizer=GR()`: A regularizer function. Can be exchanged
- `λ=0.05`: A float indicating the total weighting of the regularizer with 
    respect to the global loss function
- `iterations=100`: Specifies number of iterations.
- `progress`: if not `nothing`, the progress will be monitored in a summary dictionary as obtained by
              DeconvOptim.options_trace_deconv()
- `rec0=nothing`: optional initial reconstruction guess (same size as `measured`). When `nothing`
              (the default), the iteration starts from `abs.(conv_temp(measured, otf))` as before.

# Example
```julia-repl
julia> using DeconvOptim, TestImages, Colors, Noise;

julia> img = Float32.(testimage("resolution_test_512"));

julia> psf = Float32.(generate_psf(size(img), 30));

julia> img_b = conv(img, psf);

julia> img_n = poisson(img_b, 300);

julia> @time res = richardson_lucy_iterative(img_n, psf);
```
"""
function richardson_lucy_iterative(measured, psf;
                                   regularizer=GR(),
                                   λ=0.05,
                                   iterations=100,
                                   conv_dims=1:ndims(psf),
                                   progress = nothing,
                                   rec0 = nothing)


    otf, conv_temp = plan_conv(measured, psf, conv_dims)
    otf_conj = conj.(otf)
    # initializer — a copy since `rec` is mutated in place (`.*=`) below, and
    # the caller may want to reuse `rec0` elsewhere.
    rec = rec0 === nothing ? abs.(conv_temp(measured, otf)) : copy(rec0)
    
    # buffer for gradient
    # we need Base.invokelatest because of world age issues with generated
    # regularizers
    buffer_grad =  let 
        if !isnothing(regularizer)
            Base.invokelatest(gradient, regularizer, rec)[1]
        else
            nothing
        end
    end

    ∇reg(x) = buffer_grad .= Base.invokelatest(gradient, regularizer, x)[1]

    buffer = copy(measured)

    # Float32 FFT round-off makes the forward convolution slightly negative in
    # dark (near-zero) regions; dividing by / taking the log of that yields
    # Inf/NaN, which the next FFT then spreads over the whole array. Clamp the
    # forward model to a tiny positive floor.
    fwd_floor = eps(real(eltype(measured)))
    iter_without_reg(rec) = begin
        buffer .= measured ./ max.(conv_temp(rec, otf), fwd_floor)
        conv_temp(buffer, otf_conj)
    end
    iter_with_reg(rec) = buffer .= (iter_without_reg(rec) .- λ .* ∇reg(rec))

    iter = isnothing(regularizer) ? iter_without_reg : iter_with_reg

    # the loss function is only needed for logging, not for LR itself
    loss(myrec) = begin
        fwd = max.(conv_temp(myrec, otf), fwd_floor)
        return sum(fwd .- measured .* log.(fwd))
    end

    # logging part
    tmp_time = 0.0
    if progress !== nothing
        record_progress!(progress, rec, 0, loss(rec), 0.0, 1.0)
        tmp_time=time()
    end
    code_time = 0.0

    # do actual optimization
    for i in 1:iterations
        # the multiplicative update must stay non-negative (round-off or the
        # regularizer term can push it slightly below zero)
        rec .*= max.(iter(rec), zero(eltype(rec)))
        if progress !== nothing
            # do not count the time for evaluating the loss here.
            code_time += time() .- tmp_time
            record_progress!(progress, copy(rec), i, loss(rec), code_time, 1.0)
            tmp_time=time()
        end
    end

    return rec
end

