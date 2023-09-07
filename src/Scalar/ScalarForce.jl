###
### "THE BEER-WARE LICENSE":
### Alberto Ramos wrote this file. As long as you retain this 
### notice you can do whatever you want with this stuff. If we meet some 
### day, and you think this stuff is worth it, you can buy me a beer in 
### return. <alberto.ramos@cern.ch>
###
### file:    ScalarForce.jl
### created: Wed Oct  6 15:39:07 2021
###                               

function force_scalar(ymws::YMworkspace, sws::ScalarWorkspace, U, Phi, sp::ScalarParm, gp::GaugeParm, lp::SpaceParm)

    @timeit "Scalar force" begin
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_force_scalar!(ymws.frc1,sws.frc1,U,Phi,sp,gp,lp)
        end
    end

    
    return nothing
end

function krnl_force_scalar!(fgauge, fscalar, U::AbstractArray{TG}, Phi::AbstractArray{TS}, sp::ScalarParm{NP,T}, gp::GaugeParm, lp::SpaceParm{N,M,B,D}) where {TG,TS,NP,T,N,M,B,D}

    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)

    for id in 1:N
        fgauge[b,id,r] = (gp.beta/gp.ng)*fgauge[b,id,r]
    end
    sync_threads()

    for i in 1:NP
        fscalar[b,i,r] = zero(TS)
        for id in 1:N
            bu, ru = up((b,r), id, lp)
            bd, rd = dw((b,r), id, lp)

            p1 = U[b,id,r]*Phi[bu,i,ru]

            fscalar[b,i,r] += (2*sp.kap[i])*(p1 + dag(U[bd,id,rd])*Phi[bd,i,rd])

            fgauge[b,id,r] -= (2*sp.kap[i])*projalg(p1*dag(Phi[b,i,r]))
        end
        fscalar[b,i,r] -= ( 2 + 4*sp.eta[i]*(dot(Phi[b,i,r],Phi[b,i,r])-1) ) * Phi[b,i,r]
    end
    return nothing
end


function krnl_force_scalar!(fgauge, fscalar, U::AbstractArray{TG}, Phi::AbstractArray{TS}, sp::ScalarParm{2,T}, gp::GaugeParm, lp::SpaceParm{N,M,B,D}) where {TG,TS,T,N,M,B,D}

    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)

    for id in 1:N
        fgauge[b,id,r] = (gp.beta/gp.ng)*fgauge[b,id,r]
    end
    sync_threads()

    for i in 1:2
        fscalar[b,i,r] = zero(TS)
        for id in 1:N
            bu, ru = up((b,r), id, lp)
            bd, rd = dw((b,r), id, lp)

            p1 = U[b,id,r]*Phi[bu,i,ru]

            fscalar[b,i,r] += (2*sp.kap[i])*(p1 + dag(U[bd,id,rd])*Phi[bd,i,rd])

            fgauge[b,id,r] -= (2*sp.kap[i])*projalg(p1*dag(Phi[b,i,r]))
        end
    end
    sdot1 = dot(Phi[b,1,r],Phi[b,1,r])
    sdot2 = dot(Phi[b,2,r],Phi[b,2,r])
    sdot12 = dot(Phi[b,1,r],Phi[b,2,r])
    tr12p3 = real( tr_ipau( dag(Phi[b,1,r])*Phi[b,2,r], Pauli{3} ) )

    fscalar[b,1,r] -= (2 * (1 + 2*sp.eta[1]*(sdot1-1) + sp.xi[1]*sdot2 + 2*sp.xi[4]*sdot12)) * Phi[b,1,r]
    fscalar[b,2,r] -= (2 * (1 + 2*sp.eta[2]*(sdot2-1) + sp.xi[1]*sdot1 + 2*sp.xi[5]*sdot12)) * Phi[b,2,r]

    fscalar[b,1,r] -= (2 * (sp.muh + (sp.xi[2] + sp.xi[3])*sdot12 + sp.xi[4]*sdot1 + sp.xi[5]*sdot2)) * Phi[b,2,r]
    fscalar[b,1,r] -= 2 * tr12p3 * ( sp.xi[2] - sp.xi[3] ) * fundipau(Phi[b,2,r], Pauli{3})
    fscalar[b,2,r] -= (2 * (sp.muh + sdot12*(sp.xi[2] + sp.xi[3]) + sp.xi[4]*sdot1 + sp.xi[5]*sdot2)) * Phi[b,1,r]
    fscalar[b,2,r] -= 2 * tr12p3 * ( -sp.xi[2] + sp.xi[3] ) * fundipau(Phi[b,1,r], Pauli{3})

    return nothing
end


