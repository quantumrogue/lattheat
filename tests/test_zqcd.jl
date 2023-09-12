"""
    Simulations with a single scalar
    Originally intended to compare Atsuki's results without gauge field
    (gauge dynamics turned off and β=0)
    This explores only the kinetic term of the scalar action -- Gaussian free theory
"""
using CUDA, Logging, StructArrays, Random, TimerOutputs, ADerrors, Printf, BDIO
CUDA.allowscalar(false)

import Pkg
Pkg.activate("latticeGPU")

using LatticeGPU






lp = SpaceParm{4}((8,8,8,8), (4,4,4,4))
beta = 0.0



gp = GaugeParm(beta, 1.0, (0.0,0.0), 2)
eta = 0.0
# NSC = tryparse(Int64, ARGS[1])
NSC = 1
println("Space  Parameters: \n", lp)
println("Gauge  Parameters: \n", gp)
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
# Gauge fields U_μ=1
println("Allocating gauge field")
U = vector_field(GRP{PREC}, lp)
fill!(U, one(GRP{PREC}))
# Scalar fields φ=0
println("Allocating scalar field")
Phi = nscalar_field(SCL{PREC}, NSC, lp)
fill!(Phi, zero(SCL{PREC}))
dt  = 0.05
nsteps  = 100
nth    = 100 #thermalization length
niter  = 20000 # MC length
startk = 0.02
nrks   = 10 # nr of k values
h      = 0.1/nrks*2
# Observables
pl     = Vector{Float64}(undef, niter+nth)
rho    = Vector{Float64}(undef, niter+nth)
Lphi   = Vector{Float64}(undef, niter+nth)
Lalp   = Vector{Float64}(undef, niter+nth)
dh     = Vector{Float64}(undef, niter+nth)
acc    = Vector{Bool}(undef, niter+nth)
# Save observables in BDIO file
# Uinfo 1:  Simulation parameters
# Uinfo 2:  List of kappa values
# Uinfo 3:  Thermalization plaquette
# Uinfo 4:  Measurements   plaquette
# Uinfo 5:  Thermalization rho2
# Uinfo 6:  Measurements   rho2
# Uinfo 8:  Thermalization Lphi
# Uinfo 7:  hash
# Uinfo 9:  Measurements   Lphi
# Uinfo 10:  Thermalization Lalp
# Uinfo 11: Measurements   Lalp
# Uinfo 12: Thermalization dh
# Uinfo 13: Measurement    dh
# Lattice dimensions
global dm = ""
for i in 1:lp.ndim-1
    global dm *= string(lp.iL[i])*"x"
end
dm *= string(lp.iL[end])
filename = string("var_k/scalar",NSC,"_",dm,"_vark",nrks,"_beta",beta,"_eta",eta,"_niter", niter,"_eps",dt,"_nsteps",nsteps,".bdio")
fb = BDIO_open(filename, "d",
               "Test of scalar simulations with $NSC scalar fields")
# Simulation param
iv = [nth, niter, nrks]
BDIO_start_record!(fb, BDIO_BIN_INT64LE, 1, true)
BDIO_write!(fb, iv)
BDIO_write_hash!(fb)
# k values
kv = Vector{Float64}()
for i in 1:nrks
    push!(kv, h*(i-1)+startk)
    # push!(kv, 0.0)
end
BDIO_start_record!(fb, BDIO_BIN_F64LE, 2, true)
BDIO_write!(fb, kv)
BDIO_write_hash!(fb)
for i in 1:nrks
    if NSC == 1
        sp = ScalarParm((h*(i-1)+startk,), (eta,))
        # sp = ScalarParm((0.1,), (eta,))
    else
        sp = ScalarParm((h*(i-1),h*(i-1)), (0.5,0.5))
    end
    println("## Simulating Scalar parameters: ")
    println(sp)
    
    k = 0
    HMC!(U,Phi, dt,nsteps,lp, gp, sp, ymws, sws; noacc=true)
    # Thermalization
    for j in 1:nth
        k = k + 1
        dh[k], acc[k] = HMC!(U,Phi, dt,nsteps,lp, gp, sp, ymws, sws)
        pl[k]  = plaquette(U,lp, gp, ymws)
        rho[k],Lphi[k],Lalp[k] = scalar_obs(U, Phi, sp, lp, ymws)
        
        @printf("  THM %d/%d (kappa: %4.3f):   %s   %6.2e    %20.12e    %20.12e    %20.12e    %20.12e\n",
                j, nth, sp.kap[1], acc[k] ? "true " : "false", dh[k],
                pl[k], rho[k], Lphi[k], Lalp[k])
    end
    println(" ")
    # MC chain
    for j in 1:niter
        k = k + 1
        dh[k], acc[k] = HMC!(U,Phi, dt,nsteps,lp, gp, sp, ymws, sws)
        pl[k]  = plaquette(U,lp, gp, ymws)
        rho[k],Lphi[k],Lalp[k] = scalar_obs(U, Phi, sp, lp, ymws)
        
        @printf("  MSM %d/%d (kappa: %4.3f):   %s   %6.2e    %20.12e    %20.12e    %20.12e    %20.12e\n",
                j, niter, sp.kap[1], acc[k] ? "true " : "false", dh[k],
                pl[k], rho[k], Lphi[k], Lalp[k])
    end
    # Write to BDIO
    BDIO_start_record!(fb, BDIO_BIN_F64LE, 3, true)
    BDIO_write!(fb, pl[1:nth])
    BDIO_write_hash!(fb)
    BDIO_start_record!(fb, BDIO_BIN_F64LE, 4, true)
    BDIO_write!(fb, pl[nth+1:end])
    BDIO_write_hash!(fb)
    BDIO_start_record!(fb, BDIO_BIN_F64LE, 5, true)
    BDIO_write!(fb, rho[1:nth])
    BDIO_write_hash!(fb)
    BDIO_start_record!(fb, BDIO_BIN_F64LE, 6, true)
    BDIO_write!(fb, rho[nth+1:end])
    BDIO_write_hash!(fb)
    BDIO_start_record!(fb, BDIO_BIN_F64LE, 8, true)
    BDIO_write!(fb, Lphi[1:nth])
    BDIO_write_hash!(fb)
    BDIO_start_record!(fb, BDIO_BIN_F64LE, 9, true)
    BDIO_write!(fb, Lphi[nth+1:end])
    BDIO_write_hash!(fb)
    BDIO_start_record!(fb, BDIO_BIN_F64LE, 10, true)
    BDIO_write!(fb, Lalp[1:nth])
    BDIO_write_hash!(fb)
    BDIO_start_record!(fb, BDIO_BIN_F64LE, 11, true)
    BDIO_write!(fb, Lalp[nth+1:end])
    BDIO_write_hash!(fb)
    BDIO_start_record!(fb, BDIO_BIN_F64LE, 12, true)
    BDIO_write!(fb, dh[1:nth])
    BDIO_write_hash!(fb)
    BDIO_start_record!(fb, BDIO_BIN_F64LE, 13, true)
    BDIO_write!(fb, dh[nth+1:end])
    BDIO_write_hash!(fb)
    println("\n\n")
end
println("## Timming results")
print_timer(linechars = :ascii)
BDIO_close!(fb)
println("