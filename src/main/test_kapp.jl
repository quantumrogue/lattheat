using CUDA, Logging, StructArrays, Random, TimerOutputs, ADerrors, Printf, BDIO


CUDA.allowscalar(false)
import Pkg
Pkg.activate("/lhome/ific/a/alramos/s.images/julia/workspace/LatticeGPU")
#Pkg.activate("/home/alberto/code/julia/LatticeGPU")
using LatticeGPU

lp = SpaceParm{4}((16,16,16,16), (4,4,4,4))
gp = GaugeParm(2.25, 1.0, (0.0,0.0), 2)

NSC = tryparse(Int64, ARGS[1])
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
println("Allocating gauge field")
U = vector_field(GRP{PREC}, lp)
fill!(U, one(GRP{PREC}))
println("Allocating scalar field")
Phi = nscalar_field(SCL{PREC}, NSC, lp)
fill!(Phi, zero(SCL{PREC}))

dt  = 0.2
ns  = 10

nth    = 500
nms    = 10000
nsteps = 60
h      = 0.6/nsteps
pl     = Array{Float64, 2}(undef, nms+nth, nsteps)
rho    = Array{Float64, 2}(undef, nms+nth, nsteps)
Lphi   = Array{Float64, 2}(undef, nms+nth, nsteps)
Lalp   = Array{Float64, 2}(undef, nms+nth, nsteps)
dh     = Array{Float64, 2}(undef, nms+nth, nsteps)
acc    = Array{Bool, 2}(undef, nms+nth, nsteps)

for i in 1:nsteps
    if NSC == 1
        sp = ScalarParm((h*(i-1),), (0.5,))
    else
        sp = ScalarParm((h*(i-1),h*(i-1)), (0.5,0.5))
    end
    println("## Simulating Scalar parameters: ")
    println(sp)
    
    k = 0
    HMC!(U,Phi, dt,ns,lp, gp, sp, ymws, sws; noacc=true)
    for j in 1:nth
        k = k + 1
        dh[k,i], acc[k,i] = HMC!(U,Phi, dt,ns,lp, gp, sp, ymws, sws)
        pl[k,i]  = plaquette(U,lp, gp, ymws)
        rho[k,i],Lphi[k,i],Lalp[k,i] = scalar_obs(U, Phi, sp, lp, ymws)
        
        @printf("  THM %d/%d (kappa: %4.3f):   %s   %6.2e    %20.12e    %20.12e    %20.12e    %20.12e\n",
                j, nth, sp.kap[1], acc[k,i] ? "true " : "false", dh[k,i],
                pl[k,i], rho[k,i], Lphi[k,i], Lalp[k,i])
    end
    println(" ")
    
    for j in 1:nms
        k = k + 1
        dh[k,i], acc[k,i] = HMC!(U,Phi, dt,ns,lp, gp, sp, ymws, sws)
        pl[k,i]  = plaquette(U,lp, gp, ymws)
        rho[k,i],Lphi[k,i],Lalp[k,i] = scalar_obs(U, Phi, sp, lp, ymws)
        
        @printf("  MSM %d/%d (kappa: %4.3f):   %s   %6.2e    %20.12e    %20.12e    %20.12e    %20.12e\n",
                j, nms, sp.kap[1], acc[k,i] ? "true " : "false", dh[k,i],
                pl[k,i], rho[k,i], Lphi[k,i], Lalp[k,i])
    end

    println("\n\n")
end

println("## Timming results")
print_timer(linechars = :ascii)


# Save observables in BDIO file
# Uinfo 1:  List of kappa values
# Uinfo 2:  Thermalization plaquette
# Uinfo 3:  Measurements   plaquette
# Uinfo 4:  Thermalization rho2
# Uinfo 5:  Measurements   rho2
# Uinfo 6:  Thermalization Lphi
# Uinfo 8:  Measurements   Lphi
# Uinfo 9:  Thermalization Lalp
# Uinfo 10: Measurements   Lalp
# Uinfo 11: Thermalization dh
# Uinfo 12: Measurement    dh

fb = BDIO_open("scalar_results_nscalar$NSC.bdio", "w",
               "Test of scalar simulations with $NSC scalar fields")

kv = Vector{Float64}()
for i in 1:nsteps
    push!(kv, h*(i-1))
end
BDIO_start_record!(fb, BDIO_BIN_F64LE, 1, true)
BDIO_write!(fb, kv)
BDIO_write_hash!(fb)
    
for i in 1:nsteps
    BDIO_start_record!(fb, BDIO_BIN_F64LE, 2, true)
    BDIO_write!(fb, pl[1:nth,i])
    BDIO_write_hash!(fb)
    BDIO_start_record!(fb, BDIO_BIN_F64LE, 3, true)
    BDIO_write!(fb, pl[nth+1:end,i])
    BDIO_write_hash!(fb)
    
    BDIO_start_record!(fb, BDIO_BIN_F64LE, 4, true)
    BDIO_write!(fb, rho[1:nth,i])
    BDIO_write_hash!(fb)
    BDIO_start_record!(fb, BDIO_BIN_F64LE, 5, true)
    BDIO_write!(fb, rho[nth+1:end,i])
    BDIO_write_hash!(fb)
    
    BDIO_start_record!(fb, BDIO_BIN_F64LE, 6, true)
    BDIO_write!(fb, Lphi[1:nth,i])
    BDIO_write_hash!(fb)
    BDIO_start_record!(fb, BDIO_BIN_F64LE, 8, true)
    BDIO_write!(fb, Lphi[nth+1:end,i])
    BDIO_write_hash!(fb)
    
    BDIO_start_record!(fb, BDIO_BIN_F64LE, 9, true)
    BDIO_write!(fb, Lalp[1:nth,i])
    BDIO_write_hash!(fb)
    BDIO_start_record!(fb, BDIO_BIN_F64LE, 10, true)
    BDIO_write!(fb, Lalp[nth+1:end,i])
    BDIO_write_hash!(fb)
    
    BDIO_start_record!(fb, BDIO_BIN_F64LE, 11, true)
    BDIO_write!(fb, dh[1:nth,i])
    BDIO_write_hash!(fb)
    BDIO_start_record!(fb, BDIO_BIN_F64LE, 12, true)
    BDIO_write!(fb, dh[nth+1:end,i])
    BDIO_write_hash!(fb)
end

BDIO_close!(fb)

println("## END")
# END

