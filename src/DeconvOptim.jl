module DeconvOptim
    
export gpu_or_cpu

 # to check whether CUDA is enabled
using Requires
# for fast array regularizers
using Tullio
# optional CUDA dependency
include("requires.jl")

 # for optimization
using Optim
 #mean
using Statistics
using StatsBase
using FFTW
# Note: a top-level FFTW.set_num_threads(...) here would only run during precompilation.
# Multithreaded FFTW is deliberately not enabled at load time: it crashed
# (EXCEPTION_ACCESS_VIOLATION in FFTW's spawn_apply) in the test suite on Windows.
using LineSearches

# possible up_sampling 
using Interpolations

# for defining custom derivatives
using ChainRulesCore
using LinearAlgebra

using FillArrays

using PrecompileTools


include("forward_models.jl")
include("lossfunctions.jl")
include("mappings.jl")
# special CUDA regularizers
include("regularizer_cuda.jl")
include("regularizer.jl")
include("positivity.jl")
# include("regularizer_ka.jl")  # is fully functional but not faster and therefor currently not included
include("utils.jl")
include("padding.jl")
include("conv.jl")
include("generic_invert.jl")
include("lucy_richardson.jl")
include("map_cj.jl")
include("map_cj_lean.jl")
include("map_cj_exact.jl")
include("admm.jl")
include("deconvolution.jl")
include("analysis_tools.jl")

# refresh Zygote to load the custom rrules defined with ChainRulesCore
using Zygote: gradient

# doesn't save too much but a little
@setup_workload begin
    img = abs.(randn((4,4,2)))
    psf = abs.(randn((4,4,2)))

    @compile_workload begin
        deconvolution(Float32.(img), Float32.(psf), regularizer=TV(num_dims=3), iterations=2)
        deconvolution(img, psf, regularizer=TV(num_dims=3), iterations=2)
    end

end


# end module
end
