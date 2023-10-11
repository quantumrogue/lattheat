###
### "THE BEER-WARE LICENSE":
### Alberto Ramos wrote this file. As long as you retain this 
### notice you can do whatever you want with this stuff. If we meet some 
### day, and you think this stuff is worth it, you can buy me a beer in 
### return. Alessandro Conigli wrote this file, based on a work by Alberto Ramos.
### <aconigli@uni-mainz.de> <alberto.ramos@cern.ch>
###
### file:    HotSU2Action.jl
### created: Tue Oct  5 11:53:49 2021
###    
function hotSU2_action(U, Sigma, Pi, lp::SpaceParm, sp::HotSU2Parm, gp::GaugeParm, ymws::YMworkspace{T}) where {T <: AbstractFloat}
    @timeit "HotSU2 action" begin
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_action!(ymws.rm, gp.beta, U, Sigma, Pi, sp, lp)
        end
    end

    S = CUDA.reduce(+, ymws.rm)
    return S
end

function krnl_action!(act, beta,  U::AbstractArray{TG}, Sigma::AbstractArray{TS}, Pi::AbstractArray{TP}, sp::HotSU2Parm{T}, lp::SpaceParm{N,M,B,D}) where {TG,TS,TP,T,N,M,B,D}

    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)

    S = zero(eltype(act))

    # kinetic term Σ
    Sig2 = Sigma[b,r] * Sigma[b,r] 
    S += Sig2
    for id in 1:N
        bu, ru = up((b,r), id, lp)
        S -= Sigma[bu,ru] * Sigma[b,r]
    end

    # kinetic term Π
    pi2 = dot(Pi[b,r], Pi[b,r])
    for id in 1:N
        bu, ru = up((b,r), id, lp)
        S += 2* tr(pi2 - Pi[b,r] * U[b,dir,r] * Pi[bu,ru] * dag(U[b,dir,r]) )
    end

    S *= (4/beta)

    # potential 
    S += (4/beta)^3 * 
        (sp.b1 * Sig2 + 
        sp.b2 * pi2 + 
        sp.c1 * Sig2 * Sig2 + 
        sp.c2 * pi2 * pi2 +
        sp.c3 * pi2 * Sig2)

    I = point_coord((b,r), lp)
    act[I] = S
    return nothing
end