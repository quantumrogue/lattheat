using CUDA, Logging, StructArrays, Random

CUDA.allowscalar(true)
import Pkg
Pkg.activate("/lhome/ific/a/alramos/s.images/julia/workspace/LatticeGPU")
#Pkg.activate("/home/alberto/code/julia/LatticeGPU")
using LatticeGPU

println(CUDA.device())


GRP  = SU3
ALG  = SU3alg
PREC = Float64
lp = SpaceParm{4}((64,64,64,64), (4,4,4,4))
gp = GaugeParm{PREC}(6.0, 5.0/3.0, (0.0,0.0), 3)

println("Space  Parameters: ", lp)
println("Gauge  Parameters: ", gp)

println("Seeding CURAND...")
Random.seed!(CURAND.default_rng(), 1234)
Random.seed!(1234)

println("Precision: ", PREC)

println("Allocating gauge field")
U = vector_field(GRP{PREC}, lp)
fill!(U, one(GRP{PREC}))

println("Take to take the configuration to memory: ")
@time Ucpu = Array(U)


println("Allocating YM workspace")
ymws = YMworkspace(GRP, PREC, lp)

#CUDA.@sync begin
#    @device_code_warntype CUDA.@cuda threads=lp.bsz blocks=lp.rsz LatticeGPU.krnl_plaq!(ymws.cm, U, lp)
#end
    
@time S = gauge_action(U, lp, gp, ymws)
@time S = gauge_action(U, lp, gp, ymws)
println("Initial Action: ", S)


println(" - Randomize momenta: ")
@time randomize!(ymws.frc1, lp, ymws)

eps = 0.1
ns  = 10
ntot = 3
pl = Vector{Float64}()
for i in 1:3
    @time dh, acc = HMC!(U,eps,ns,lp, gp, ymws, noacc=true)
    println("# HMC: ", acc, " ", dh)
    push!(pl, plaquette(U,lp, gp, ymws))
    println("# Plaquette: ", pl[end], "\n")
end
for i in 1:ntot
    CUDA.@profile dh, acc = HMC!(U,eps,ns,lp, gp, ymws)
    println("# HMC: ", acc, " ", dh)
    push!(pl, plaquette(U,lp, gp, ymws))
    println("# Plaquette: ", pl[end], "\n")
end

@time wfl_rk3(U, 1, 0.01, lp, ymws)

println("Action: ", gauge_action(U, lp, gp, ymws))
println("Time for 100 steps of RK3 flow integrator: ")
@time wfl_rk3(U, 100, 0.01, lp, ymws)
println("Action: ", gauge_action(U, lp, gp, ymws))


println("END")

