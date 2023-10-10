using CUDA, Logging, StructArrays, Random, TimerOutputs

CUDA.allowscalar(true)

import Pkg
Pkg.activate("/home/aconigli/lattheat")
using  LatticeGPU

# Set Hyperparameters
GRP  = SU2
ALG  = SU2alg
PREC = Float64
println("Precision:        ", PREC)

println("Seeding CURAND...")
Random.seed!(CURAND.default_rng(), 1234)
Random.seed!(1234)

lp = SpaceParm{3}((16,16,16), (4,4,4))
println("Space Parameters: ", lp)

gp = GaugeParm{PREC}(GRP{PREC}, 12, 0.0)
println("Gauge Parameters:  ", gp)

hp = HotSU2Param(sqrt(5), 5.1, gp.beta)
println("HotSU2 Parameters: ", hp) 

# Allocate workspaces
ymws   = YMWorkspace(GRP, PREC, lp)
hsu2ws = HotSU2Workspace(PREC, lp)

# Main program
println("Allocating gauge fields")
U = vector_field(GRP{PREC}, lp)
fill!(U, one(GRP{PREC}))

println("Allocating Σ and Π fields")
Sigma = scalar_field(PREC, lp)
fill!(Sigma, zero(PREC))
Pi = scalar_field(ALG{PREC}, lp)
fill!(Pi, one(ALG{PREC}))

println("Computing initial action...")
@time begin
    S_g   = gauge_action(U, lp, gp, ymws)
    S_hot = hotSU2_action(U, Sigma, Pi, lp, hp, gp, ymws)
end
println("Initial gauge action:  ", S_g)
println("Initial hotSU2 action: ", S_hot)
println("Total initial action:  ", S_g + S_hot)

