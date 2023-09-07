###
### "THE BEER-WARE LICENSE":
### Alberto Ramos wrote this file. As long as you retain this 
### notice you can do whatever you want with this stuff. If we meet some 
### day, and you think this stuff is worth it, you can buy me a beer in 
### return. <alberto.ramos@cern.ch>
###
### file:    ScalarHMC.jl
### created: Sun Oct 10 23:40:28 2021
###                               


function hamiltonian(mom, U, pmom, Phi, lp, gp, sp, ymws)
    
    @timeit "Computing Hamiltonian" begin
        SG = gauge_action(U, lp, gp, ymws)
        SS = scalar_action(U, Phi, lp, sp, ymws)
        PG = CUDA.mapreduce(norm2, +, mom)/2
        PS = CUDA.mapreduce(norm2, +, pmom)/2
    end
        
    return SG+SS+PG+PS
end

function HMC!(U, Phi,  int::IntrScheme, lp::SpaceParm, gp::GaugeParm, sp::ScalarParm, ymws::YMworkspace{T}, sws::ScalarWorkspace; noacc=false) where T

    @timeit "HMC trajectory" begin
        ymws.U1 .= U
        sws.Phi .= Phi
        
        randomize!(ymws.mom, lp, ymws)
        randomize!(sws.mom, sp, lp, ymws)
        hini = hamiltonian(ymws.mom, U, sws.mom, Phi, lp, gp, sp, ymws)
        
        MD!(ymws.mom, U, sws.mom, Phi, int, lp, gp, sp, ymws, sws)
        
        dh   = hamiltonian(ymws.mom, U, sws.mom, Phi, lp, gp, sp, ymws) - hini
        pacc = exp(-dh)
        
        acc = true
        if (noacc)
            return dh, acc
        end
        
        if (pacc < 1.0)
            r = rand()
            if (pacc < r) 
                U   .= ymws.U1
                Phi .= sws.Phi
                acc = false
            end
        end
    end
        
    return dh, acc
end
HMC!(U, Phi, eps, ns, lp::SpaceParm, gp::GaugeParm, ymws::YMworkspace{T}; noacc=false) where T = HMC!(U, Phi, omf4(T, eps, ns), lp, gp, ymws; noacc=noacc)

function MD!(mom, U, pmom, Phi, int::IntrScheme{NI, T}, lp::SpaceParm, gp::GaugeParm, sp::ScalarParm, ymws::YMworkspace{T}, sws::ScalarWorkspace) where {NI, T <: AbstractFloat}

    @timeit "MD evolution" begin
        YM.force_gauge(ymws, U, gp.c0, gp, lp)
        force_scalar(ymws, sws, U, Phi, sp, gp, lp)
        
        mom  .= mom  .+ (int.r[1]*int.eps) .* ymws.frc1
        pmom .= pmom .+ (int.r[1]*int.eps) .* sws.frc1
        for i in 1:int.ns
            k   = 2
            off = 1
            for j in 1:NI-1
                U .= expm.(U, mom, int.eps*int.r[k])
                Phi .= Phi .+ (int.eps*int.r[k]).*pmom
                if k == NI
                    off = -1
                end
                k += off
                
                YM.force_gauge(ymws, U, gp.c0, gp, lp)
                force_scalar(ymws, sws, U, Phi, sp, gp, lp)
                if (i < int.ns) && (k == 1)
                    mom  .= mom  .+ (2*int.r[k]*int.eps) .* ymws.frc1
                    pmom .= pmom .+ (2*int.r[k]*int.eps) .* sws.frc1
                else
                    mom  .= mom  .+ (int.r[k]*int.eps) .* ymws.frc1
                    pmom .= pmom .+ (int.r[k]*int.eps) .* sws.frc1
                end
                k += off
            end
        end
    end
    
    return nothing
end
