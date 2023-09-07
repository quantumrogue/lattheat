using CUDA, Logging, StructArrays, Random, TimerOutputs

CUDA.allowscalar(true)
import Pkg
Pkg.activate("/lhome/ific/a/alramos/s.images/julia/workspace/LatticeGPU")
#Pkg.activate("/home/alberto/code/julia/LatticeGPU")
using LatticeGPU

lp = SpaceParm{4}((16,16,16,16), (4,4,4,4))
gp = GaugeParm(6.0, 1.0, (0.0,0.0), 2)
sp = ScalarParm((0.2,0.3), (1.0,0.4), 0.5, (0.234,0.13,0.145,0.1))

NSC = length(sp.kap) #number of scalars = # of k coupling
println("Space  Parameters: ", lp)
println("Gauge  Parameters: ", gp)
println("Scalar Parameters: ", sp)
GRP  = SU2
ALG  = SU2alg
SCL  = SU2fund
PREC = Float64
println("Precision:         ", PREC)

println("Allocating YM workspace")
ymws = YMworkspace(GRP, PREC, lp)
println("Allocating Scalar workspace")
sws  = ScalarWorkspace(PREC, NSC, lp)

# Seed RNG
println("Seeding CURAND...")
Random.seed!(CURAND.default_rng(), 1234)
Random.seed!(1234)

# Main program
println("Allocating gauge field")
U = vector_field(GRP{PREC}, lp)
fill!(U, one(GRP{PREC}))
println("Allocating scalar field")
Phi = nscalar_field(SCL{PREC}, NSC, lp)
fill!(Phi, zero(SCL{PREC}))

println("Initial Action: ")
@time S = gauge_action(U, lp, gp, ymws) + scalar_action(U, Phi, lp, sp, ymws)


dt  = 0.05
ns  = 20

println("## Thermalization")
pl = Vector{Float64}()
rho2_v = Vector{Float64}()
lphi_v = Vector{complex(Float64)}()
lalpha_v = Vector{complex(Float64)}()

for i in 1:10
    @time dh, acc = HMC!(U,Phi, dt,ns,lp, gp, sp, ymws, sws, noacc=true)
    println("# HMC: ", acc, " ", dh)
    push!(pl, plaquette(U,lp, gp, ymws))
    println("# Plaquette: ", pl[end], "\n")
end

println("## Production")
for i in 1:10
    @time dh, acc = HMC!(U,Phi, dt,ns,lp, gp, sp, ymws, sws)
    println("# HMC: ", acc, " ", dh)
    push!(pl, plaquette(U,lp, gp, ymws))
    println("# Plaquette: ", pl[end], "\n")
end

print_timer(linechars = :ascii)
