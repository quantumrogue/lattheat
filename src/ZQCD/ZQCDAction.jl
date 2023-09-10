###
### "THE BEER-WARE LICENSE":
### Alberto Ramos wrote this file. As long as you retain this 
### notice you can do whatever you want with this stuff. If we meet some 
### day, and you think this stuff is worth it, you can buy me a beer in 
### return. <alberto.ramos@cern.ch>
###
### author:  pietro.butti.fl@gmail.com
### file:    ZQCDAction.jl
### created: Fri  8 Sep 2023 12:22:24 CEST
###                               

function zqcd_action(U, Sigma, Pi, lp::SpaceParm, sp::ZQCDParm, ymws::YMworkspace{T}) where {T <: AbstractFloat}
    @timeit "Adjoint scalar action" begin
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_zqcd_act!(ymws.rm, U, Phi, sp, lp)
        end
    end
        
    S = CUDA.reduce(+, ymws.rm)
    return S
end


function krnl_zqcd_act!(act, U::AbstractArray{TG}, Sigma::AbstractArray{TS}, Pi::AbstractArray{TP}, sp::ZQCDParm{T}, lp::SpaceParm{N,M,B,D}) where {TG,TS,TP,N,M,B,D}

    # Square mapping to CUDA block
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)

    S = zero(eltype(act))

    # Calculate kinetic action for Σ
    S = - 8. / sp.beta * Sigma[b,r]
    for dir in 1:N
        b_up, r_up = up((b, r), id, lp)
        S *= Sigma[b_up,r_up]
    end
    S += 8. / sp.beta * 3. * Sigma[b,r]*Sigma[b,r]

    I = point_coord((b,r), lp)
    act[I] = S
    return nothing
end

