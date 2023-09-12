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

function zqcd_action(U, Sigma, Pi, lp::SpaceParm, sp::ZQCDParm, gp::GaugeParm, ymws::YMworkspace{T}) where T<:AbstractFloat
    @timeit "ZQCD action" begin
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_zqcd_act!(ymws.rm, gp.beta, U, Sigma, Pi, sp, lp)
        end
    end
        
    S = CUDA.reduce(+, ymws.rm)
    return S
end


function krnl_zqcd_act!(act, beta, U::AbstractArray{TG}, Sigma::AbstractArray{TS}, Pi::AbstractArray{TP}, sp::ZQCDParm{T}, lp::SpaceParm{N,M,B,D}) where {T,TG,TS,TP,N,M,B,D}

    # Square mapping to CUDA block
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)

    S = zero(eltype(act))

    # Calculate kinetic action for Σ
        sigma2 = Sigma[b,r]*Sigma[b,r]
        S -= - 2. * Sigma[b,r]
        for dir in 1:N
            b_up, r_up = up((b, r), dir, lp)
            S *= Sigma[b_up,r_up]
        end
        S += 3. * sigma2

    # Calculate kinetic action for Π
        pi2 = norm2(Pi[b,r])
        S += 3. * 2. * pi2
        for dir in 1:N
            b_up, r_up = up((b, r), dir, lp)
            S -= tr( Pi[b,r] * U[b,dir,r] * Pi[b_up,r_up] / U[b,dir,r])  ## what about inline allocation?
        end

    # # Calculate potential
    #     S += sp.b1 * sigma2 + 
    #         sp.b2 * pi2 + 
    #         sp.c1 * sigma2 * sigma2 + 
    #         sp.c2 * pi2 * pi2 +
    #         sp.c3 * pi2 * sigma2
    # S *= 4. / beta

    # Calculate gauge action
        I = point_coord((b,r), lp)
        # it = I[N]

        # for id1 in N:-1:1
        #     bu1, ru1 = up((b, r), id1, lp)
        #     ipl = ipl + 1

        #     for id2 = 1:id1-1
        #         bu2, ru2 = up((b, r), id2, lp)
        #         ipl = ipl + 1

        #         gt1 = U[bu1,id2,ru1]
        #         S -= beta * tr(U[b,id1,r]*gt1 / (U[b,id2,r]*U[bu2,id1,ru2])) / 2.
        #     end
        # end

    act[I] = S

    return nothing
end

