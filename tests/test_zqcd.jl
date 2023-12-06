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
Sigma = scalar_field(PREC,lp)
fill!(Sigma,one(PREC))

Pi = scalar_field(ALG{PREC},lp)
fill!(Pi,zero(ALG{PREC}))



function provaMD!(mom,U, Smom,Sigma, Pmom,Pi, lp::SpaceParm, gp::GaugeParm, zp::ZQCDParm{T}, ymws::YMworkspace{T}, zws::ZQCDworkspace{T}) where {NI, T <: AbstractFloat}
    for i in 1:20
        LatticeGPU.YM.force_gauge(ymws, U, gp.c0, gp, lp)
        zqcd_force(ymws,zws,U,Sigma,Pi,zp,gp,lp)

        println("...$(i).1: momentum upd")
        mom  .= mom  .+ 0.05/2 .* ymws.frc1
        Smom .= Smom .+ 0.05/2 .* zws.frcSigma
        Pmom .= Pmom .+ 0.05/2 .* zws.frcPi

        h = hamiltonian(mom, U, Smom, Pmom, Sigma, Pi, lp, zp, gp, ymws)

        println("...$(i).2: update conf")
        U     .= expm.(U, mom, 0.05)
        Sigma .= Sigma .+ 0.05 .* Smom
        Pi    .= Pi    .+ 0.05 .* Pmom

        h = hamiltonian(mom, U, Smom, Pmom, Sigma, Pi, lp, zp, gp, ymws)
        
        LatticeGPU.YM.force_gauge(ymws, U, gp.c0, gp, lp)
        zqcd_force(ymws,zws,U,Sigma,Pi,zp,gp,lp)

        
        println("...$i.3: mom upd")
        mom  .= mom  .+ 0.05/2 .* ymws.frc1
        Smom .= Smom .+ 0.05/2 .* zws.frcSigma
        Pmom .= Pmom .+ 0.05/2 .* zws.frcPi
        
        h = hamiltonian(mom, U, Smom, Pmom, Sigma, Pi, lp, zp, gp, ymws)
        println("$i:    $h")
    end

    return nothing
end

function provaHMC!(U,Sigma,Pi, lp::SpaceParm, gp::GaugeParm, zp::ZQCDParm, ymws::YMworkspace, zws::ZQCDworkspace; noacc=false)
    @timeit "HMC trajectory" begin
        ymws.U1   .= U
        zws.Sigma .= Sigma
        zws.Pi    .= Pi

        randomize!(ymws.mom,lp,ymws)
        randomize!(zws.momSigma,zws.momPi,lp,ymws)

        println("...1")
        Hin = hamiltonian(ymws.mom, U, zws.momSigma, zws.momPi, Sigma, Pi, lp, zp, gp, ymws)

        provaMD!(ymws.mom,U,zws.momSigma,Sigma,zws.momPi,Pi,lp,gp,zp,ymws,zws)

        ΔH = hamiltonian(ymws.mom, U, zws.momSigma, zws.momPi, Sigma, Pi, lp, zp, gp, ymws) - Hin
        pacc = exp(-ΔH)

        acc = true
        if (noacc)
            return ΔH, acc
        end
        
        if (pacc < 1.0)
            r = rand()
            if (pacc < r) 
                U     .= ymws.U1
                Sigma .= zws.Sigma
                Pi    .= zws.Pi
                acc = false
            end
        end
    end
    return ΔH, acc
end




## ==============================================================
## ========================== TESTS =============================
## ==============================================================
    for i in 1:5
        println("+++++++++++++ $i ++++++++++++++")
        dh,acc = provaHMC!(U,Sigma,Pi,lp,gp,zp,ymws,zws)
    end

## ==============================================================
## ==============================================================
## ==============================================================




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


