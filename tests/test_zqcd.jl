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


# Main program ==========================================
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


Sigma = Σ
Pi    = Π
int = omf4(PREC,0.05,20)

HMC!(U,Σ,Π,int,lp,gp,zp,ymws,zws)
