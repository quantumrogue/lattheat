"""
    Test zqcd simulations 
"""

using CUDA, Logging, StructArrays, Random, TimerOutputs, ADerrors, Printf, BDIO
CUDA.allowscalar(false)

import Pkg
Pkg.activate("LatticeGPU")

using LatticeGPU



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
gp = GaugeParm{PREC}(GRP{PREC}, 12, 0.)
println("Gauge  Parameters: \n", gp)

# Set ZQCD parameters
zp = ZQCDParm{PREC}(5.,6.7,12.)
println("ZQCD  Parameters: \n", zp)


println("Allocating ZQCD workspace")
ymws = YMworkspace(GRP, PREC, lp)
zws  = ZQCDworkspace(PREC, lp)

println("Allocating U")
U = vector_field(GRP{PREC}, lp)
fill!(U, one(GRP{PREC}))

println("Allocating Z")
Σ = scalar_field(PREC,lp)
Π = scalar_field(ALG{PREC},lp)


# println("Try to compute the action")
# @time S = zqcd_action(U,Σ,Π,lp,zp,gp,ymws)
# println(S)

# println("Try to compute the forces")
# @time zqcd_force(ymws,zws,U,Σ,Π,zp,gp,lp)
# @time zqcd_force(ymws,zws,U,Σ,Π,zp,gp,lp)

int = omf4(PREC,0.05,20)



function gaugeheater!(f, lp::SpaceParm, ymws::YMworkspace)
    @timeit "Randomize SU(2) gauge field" begin
        m = CUDA.randn(ymws.PRC, lp.bsz,lp.ndim,4,lp.rsz)
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_gaugeheater!(f,m,lp)
        end
        f .= unitarize.(f)
    end
    return nothing
end
function krnl_gaugeheater!(f, m, lp::SpaceParm{N,M,BC_PERIODIC,D}) where {N,M,D}
    b, r = CUDA.threadIdx().x, CUDA.blockIdx().x
    for id in 1:lp.ndim
        f[b,id,r] = SU2(complex(m[b,id,1,r], m[b,id,2,r]), complex(m[b,id,3,r],m[b,id,4,r]))
    end
    return nothing
end









## ==============================================================
## ========================== TESTS =============================
## ==============================================================

    # LatticeGPU.YM.randomize!(ymws.mom,lp,ymws)
    # # gaugeheater!(U,lp,ymws)
    # randomize!(zws.momSigma,zws.momPi,lp,ymws)
    # # randomize!(Σ,Π,lp,ymws)
    # zqcd_action(U,Σ,Π,lp,zp,gp,ymws) 
    # zqcd_MD!(ymws.mom, U, zws.momSigma, Σ, zws.momPi, Π, int, lp, gp, zp, ymws, zws)
    # zqcd_action(U,Σ,Π,lp,zp,gp,ymws) 

    HMC!(U,Σ,Π,int,lp,gp,zp,ymws,zws)


## ==============================================================
## ==============================================================
## ==============================================================