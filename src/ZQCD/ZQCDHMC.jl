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

function hamiltonian(mom,U, zmom, Sigma, Pi, lp, zp, gp, ymws)
    
    @timeit "Computing ZQCD Hamiltonian" begin
        SG = gauge_action(U, lp, gp, ymws)
        SZ = zqcd_action(U, Sigma, Pi, lp, zp, gp, ymws)
        PG = CUDA.mapreduce(norm2, +, mom)/2
        PZ = CUDA.mapreduce(norm2, +, zmom)/2
    end
        
    return SG+SZ+PG+PZ
end

## NI is the number of intermediate step in the integrator, int.ns the number of trajectories
function MD!(mom,U, Smom,Sigma, Pmom,Pi,   int::IntrScheme{NI, T}, lp::SpaceParm, gp::GaugeParm, sp::ZQCDParm{T}, ymws::YMworkspace{T}, zws::ZQCDWorkspace{T}) where {NI, T <: AbstractFloat}

    @timeit "ZQCD MD evolution" begin
        # Evaluate initial forces with (U⁽⁰⁾,Z⁽⁰⁾)
        YM.force_gauge(ymws, U, gp.c0, gp, lp)
        # zqcd_force(...)

        # Evaluate initial momenta (p⁽⁰⁾,π⁽⁰⁾)
        mom  .= mom  .+ (int.r[1]*int.eps) .* ymws.frc1
        Smom .= Smom .+ (int.r[1]*int.eps) .* zws.frcSigma
        Pmom .= Pmom .+ (int.r[1]*int.eps) .* zws.frcPi

        for trajID in 1:int.ns
            k   = 2 # what are these things?
            off = 1 # what are these things?

            for leap in 1:ND-1
                U     .= expm.(U, mom, int.eps * int.r[k])
                Sigma .= Sigma .+ (int.eps * int.r[k]) .* Smom
                Pi    .= Pi    .+ (int.eps * int.r[k]) .* Pmom # is in the algebra

                if k == NI
                    off = -1
                end
                k += off

                YM.force_gauge(ymws, U, gp.c0, gp, lp)
                # zqcd_force(...)                

                if (i < int.ns) && (k==1)
                    mom  .= mom  .+ (2. * int.r[k]*int.eps) .* ymws.frc1
                    Smom .= Smom .+ (2. * int.r[1]*int.eps) .* zws.frcSigma
                    Pmom .= Pmom .+ (2. * int.r[1]*int.eps) .* zws.frcPi
                else
                    mom .= mom   .+ (int.r[k]*int.eps) .* ymws.frc1
                    Smom .= Smom .+ (int.r[1]*int.eps) .* zws.frcSigma
                    Pmom .= Pmom .+ (int.r[1]*int.eps) .* zws.frcPi
                end
                k += off
            end
        end
    end

    return nothing
end