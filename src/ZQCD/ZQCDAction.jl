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
        for dir in 1:N
            b_up, r_up = up((b, r), dir, lp)
            S -= Sigma[b_up,r_up]
        end
        S *= Sigma[b,r]
        sigma2 = Sigma[b,r]*Sigma[b,r]
        S += sigma2

    # Calculate kinetic action for Π
        pi2 = norm2(Pi[b,r])
        S += 2. * 3. * (- pi2/2.)
        for dir in 1:N
            b_up, r_up = up((b, r), dir, lp)
            S -= 2. * tr(Pi[b,r] * U[b,dir,r] * Pi[b_up,r_up] / U[b,dir,r])
        end

    # Calculate potential
        S += sp.b1 * sigma2 + 
            sp.b2 * pi2 + 
            sp.c1 * sigma2 * sigma2 + 
            sp.c2 * pi2 * pi2 +
            sp.c3 * pi2 * sigma2

    S *= 4. / beta

    # Calculate gauge action
        for I in N:-1:1
            b_u1, r_u1 = up((b, r), I, lp)
            for J in 1:I-1
                    b_u2, r_u2 = up((b,r), J, lp)
                    S -= beta/2. * tr( U[b,I,r] * U[b_u1,J,r_u1] / U[b_u2,I,r_u2] / U[b,J,r]) 
            end
        end

        
    I = point_coord((b,r), lp)
    act[I] = S

    return nothing
end

