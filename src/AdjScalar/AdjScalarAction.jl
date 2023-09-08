###
### "THE BEER-WARE LICENSE":
### Alberto Ramos wrote this file. As long as you retain this 
### notice you can do whatever you want with this stuff. If we meet some 
### day, and you think this stuff is worth it, you can buy me a beer in 
### return. <alberto.ramos@cern.ch>
###
### author:  pietro.butti.fl@gmail.com
### file:    AdjScalarAction.jl
### created: Fri  8 Sep 2023 12:22:24 CEST
###                               

function adj_scalar_action(U, Sigma, Pi, lp::SpaceParm, sp::ScalarAdjParm, ymws::YMworkspace{T}) where {T <: AbstractFloat}
    # @timeit "Adjoint scalar action" begin
    #     CUDA.@sync begin
    #         CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_act!(ymws.rm, U, Phi, sp, lp)
    #     end
    # end
        
    # S = CUDA.reduce(+, ymws.rm)
    # return S
end


function krnl_adj_act!(act, U::AbstractArray{TG}, Sigma::AbstractArray{TS}, Pi::AbstractArray{TP}, sp::AdjScalarParm{T}, lp::SpaceParm{N,M,B,D}) where {TG,TS,TP,N,M,B,D}
    # b = Int64(CUDA.threadIdx().x)
    # r = Int64(CUDA.blockIdx().x)

    # S = zero(eltype(act))
    # for id in 1:N
    #     bu, ru = up((b, r), id, lp)
    #     for i in 1:2
    #         S += -2*sp.kap[i]*dot(Phi[b,i,r],U[b,id,r]*Phi[bu,i,ru])
    #     end
    # end

    # sdot1 = dot(Phi[b,1,r],Phi[b,1,r])
    # sdot2 = dot(Phi[b,2,r],Phi[b,2,r])
    # sdot12 = dot(Phi[b,1,r],Phi[b,2,r])
    # tr12p3 = real( tr_ipau( dag(Phi[b,1,r])*Phi[b,2,r], Pauli{3} ) )

    # S +=sdot1 + sdot2 + sp.eta[1]*(sdot1 - 1)^2 + sp.eta[2]*(sdot2 - 1)^2
    # S += 2*sp.muh*sdot12 + sp.xi[1]*sdot1*sdot2 + sp.xi[2]*(sdot12^2+tr12p3^2) + sp.xi[3]*(sdot12^2-tr12p3^2)
    # S += 2 * (sp.xi[4]*sdot12*sdot1 + sp.xi[5]*sdot12*sdot2)

    # #OLD
    # # S +=sdot1 + sdot2 + sp.eta[1]*(sdot1 - 1)^2 + sp.eta[2]*(sdot2 - 1)^2
    # # S += 2*sp.muh*sdot12 + sp.xi[1]*sdot1*sdot2 + sp.xi[2]*sdot12^2
    # # S += 2 * (sp.xi[3]*sdot12*sdot1 + sp.xi[4]*sdot12*sdot2)


    # I = point_coord((b,r), lp)
    # act[I] = S
    # return nothing
end

