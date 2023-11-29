###
### "THE BEER-WARE LICENSE":
### Alberto Ramos wrote this file. As long as you retain this 
### notice you can do whatever you want with this stuff. If we meet some 
### day, and you think this stuff is worth it, you can buy me a beer in 
### return. <alberto.ramos@cern.ch>
###
### author:  pietro.butti.fl@gmail.com
### file:    ZQCDHMC.jl
### created: Sun 17 Sep 2023 12:54:46 CEST
###                               

function hamiltonian(mom,U, smom, pmom, Sigma, Pi, lp, zp, gp, ymws)
    
    @timeit "Computing ZQCD Hamiltonian" begin
        SG = gauge_action(U, lp, gp, ymws)
        SZ = zqcd_action(U, Sigma, Pi, lp, zp, gp, ymws)
        PG = CUDA.mapreduce(norm2, +, mom)/2
        PZ = CUDA.mapreduce(abs2, +, smom)/2 + CUDA.mapreduce(norm2, +, pmom)/2
    end

    println("S[U]=$SG, P[U]=$PG,        S[Z]=$SZ,  P[Z]=$PZ")
        
    return SG+SZ+PG+PZ
end

function MD!(mom,U, Smom,Sigma, Pmom,Pi,   int::IntrScheme{NI, T}, lp::SpaceParm, gp::GaugeParm, zp::ZQCDParm{T}, ymws::YMworkspace{T}, zws::ZQCDworkspace{T}) where {NI, T <: AbstractFloat}
    @timeit "ZQCD MD evolution" begin
        # Evaluate initial forces with (U⁽⁰⁾,Z⁽⁰⁾)
        YM.force_gauge(ymws, U, gp.c0, gp, lp)
        zqcd_force(ymws,zws,U,Sigma,Pi,zp,gp,lp)

        # Evaluate initial momenta (p⁽⁰⁾,π⁽⁰⁾)
        mom  .= mom  .+ (int.r[1]*int.eps) .* ymws.frc1
        Smom .= Smom .+ (int.r[1]*int.eps) .* zws.frcSigma
        Pmom .= Pmom .+ (int.r[1]*int.eps) .* zws.frcPi

        for trajID in 1:int.ns
            k   = 2
            off = 1

            for leap in 1:NI-1
                U     .= expm.(U, mom, int.eps * int.r[k])
                Sigma .= Sigma .+ (int.eps * int.r[k]) .* Smom
                Pi    .= Pi    .+ (int.eps * int.r[k]) .* Pmom # is in the algebra

                if k == NI
                    off = -1
                end
                k += off

                YM.force_gauge(ymws, U, gp.c0, gp, lp)
                zqcd_force(ymws,zws,U,Sigma,Pi,zp,gp,lp)

                if (trajID < int.ns) && (k==1)
                    mom  .= mom  .+ (2. * int.r[k]*int.eps) .* ymws.frc1
                    Smom .= Smom .+ (2. * int.r[k]*int.eps) .* zws.frcSigma
                    Pmom .= Pmom .+ (2. * int.r[k]*int.eps) .* zws.frcPi
                else
                    mom  .= mom  .+ (int.r[k]*int.eps) .* ymws.frc1
                    Smom .= Smom .+ (int.r[k]*int.eps) .* zws.frcSigma
                    Pmom .= Pmom .+ (int.r[k]*int.eps) .* zws.frcPi
                end
                k += off
            end
        end
    end

    return nothing
end


function HMC!(U,Sigma,Pi, int::IntrScheme, lp::SpaceParm, gp::GaugeParm, zp::ZQCDParm, ymws::YMworkspace, zws::ZQCDworkspace; noacc=false)
    @timeit "HMC trajectory" begin
        ymws.U1   .= U
        zws.Sigma .= Sigma
        zws.Pi    .= Pi

        randomize!(ymws.mom,lp,ymws)
        randomize!(zws.momSigma,zws.momPi,lp,ymws)

        Hin = hamiltonian(ymws.mom, U, zws.momSigma, zws.momPi, Sigma, Pi, lp, zp, gp, ymws)

        MD!(ymws.mom,U,zws.momSigma,Sigma,zws.momPi,Pi,int,lp,gp,zp,ymws,zws)

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
HMC!(U,Sigma,Pi, eps,ns, lp::SpaceParm, gp::GaugeParm, zp::ZQCDParm{T}, ymws::YMworkspace, zws::ZQCDworkspace; noacc=false) where T = HMC!(U,Sigma,Pi, omf4(T,eps,ns), lp, gp, zp, ymws, zws, noacc=noacc)
