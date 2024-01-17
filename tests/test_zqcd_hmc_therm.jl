"""
    Test zqcd simulations 
"""

using CUDA, Logging, StructArrays, Random, TimerOutputs, ADerrors, Printf, BDIO
CUDA.allowscalar(false)

import Pkg
Pkg.activate("LatticeGPU")

using LatticeGPU


# whichGPU = ARGS[1]
# device!(1)

# Set lattice/block size
lp = SpaceParm{3}((8,8,8), (4,4,4))
println("Space  Parameters: ", lp)

# Seed RNG
println("Seeding CURAND...")
Random.seed!(CURAND.default_rng(), 1234)
Random.seed!(1234)

# Set group and precision
GRP  = SU2
ALG  = SU2alg
PREC = Float64
println("Precision:         ", PREC)

# Set gauge parameters
gp = GaugeParm{PREC}(GRP{PREC}, 6., 1.)
println("Gauge  Parameters: \n", gp)

# Set ZQCD parameters
zp = ZQCDParm{PREC}(5.,6.7,gp.beta)
println("ZQCD  Parameters: \n", zp)

# println("Allocating ZQCD workspace")
ymws = YMworkspace(GRP, PREC, lp)
zws  = ZQCDworkspace(PREC, lp)

println("Allocating U")
U = vector_field(GRP{PREC}, lp)

println("Allocating Z")
Sigma = scalar_field(PREC,lp)
fill!(Sigma,one(PREC))

Pi = scalar_field(ALG{PREC},lp)
fill!(Pi,zero(ALG{PREC}))



## ==============================================================
## ========================== TESTS =============================
## ==============================================================

    for i in 1:1000
        dh, acc = HMC!(U,Sigma,Pi,0.05,20,lp,gp,zp,ymws,zws)
        println("######################################################")
        println("# ΔH( $i): ", dh)
        println("# Plaquette( $i): ", plaquette(U,lp, gp, ymws))
        # println("# trZ( $i): ", CUDA.sum(Sigma))
        # println("# |trZ( $i)|: ", CUDA.mapreduce(abs,+,Sigma))
    end

## ==============================================================
## ==============================================================
## ==============================================================




# function gaugeheater!(f, lp::SpaceParm, ymws::YMworkspace)
#     @timeit "Randomize SU(2) gauge field" begin
#         m = CUDA.randn(ymws.PRC, lp.bsz,lp.ndim,4,lp.rsz)
#         CUDA.@sync begin
#             CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_gaugeheater!(f,m,lp)
#         end
#         f .= unitarize.(f)
#     end
#     return nothing
# end
# function krnl_gaugeheater!(f, m, lp::SpaceParm{N,M,BC_PERIODIC,D}) where {N,M,D}
#     b, r = CUDA.threadIdx().x, CUDA.blockIdx().x
#     for id in 1:lp.ndim
#         f[b,id,r] = SU2(complex(m[b,id,1,r], m[b,id,2,r]), complex(m[b,id,3,r],m[b,id,4,r]))
#     end
#     return nothing
# end


