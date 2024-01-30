###
### "THE BEER-WARE LICENSE":
### Alberto Ramos wrote this file. As long as you retain this 
### notice you can do whatever you want with this stuff. If we meet some 
### day, and you think this stuff is worth it, you can buy me a beer in 
### return. <alberto.ramos@cern.ch>
###
### author:  pietro.butti.fl@gmail.com
### file:    ZQCDMetropolis.jl
### created: Tue 30 Jan 2024 14:05:14 CET
###             


function sweep_gauge!(U, lp::SpaceParm, ymws::YMworkspace, ϵ, cooler)
    @timeit "Sweep for links" begin
        R    = CUDA.rand(ymws.PRC, lp.bsz, lp.ndim, 4, lp.rsz) #.- convert(ymws.PRC,0.5)
        mask = CUDA.rand(ymws.PRC, lp.bsz, lp.ndim, lp.rsz)
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_sweep_gauge!(U,R,mask,lp,convert(ymws.PRC,ϵ),cooler)          
        end
    end
    return nothing
end

function krnl_sweep_gauge!(U,R::AbstractArray{T},mask::AbstractArray{T},lp::SpaceParm,ϵ::T,cooler) where T<:AbstractFloat
    b, r = CUDA.threadIdx().x, CUDA.blockIdx().x

    for id in 1:lp.ndim
        if mask[b,id,r]<cooler 
            r0 = R[b,id,4,r]
            r1 = R[b,id,1,r]
            r2 = R[b,id,2,r]
            r3 = R[b,id,3,r]
            normx = sqrt(r1*r1 + r2*r2 + r3*r3)

            x0 = sign(r0)*sqrt(1. -ϵ*ϵ)
            x1 = ϵ*r1/normx
            x2 = ϵ*r2/normx
            x3 = ϵ*r3/normx

            XU = SU2(complex(x0,x3),complex(x2,x1)) * U[b,id,r]
            U[b,id,r] = XU
        end
    end
    return nothing
end


# function sweep_Z!(Sigma,Pi, ϵ, cooler, lp::SpaceParm, zws::ZQCDworkspace)
#     @timeit "Sweep for Z field" begin
#         R = CUDA.rand(zws.PRC, lp.bsz, 4, lp.rsz) .- convert(zws.PRC,0.5)
#         CUDA.@sync begin
#             CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_sweep_Z!(Sigma,Pi,R,ϵ, cooler)          
#         end
#     end
#     return nothing
# end

# function krnl_sweep_Z!(Sigma,Pi,R,ϵ, cooler)
#     b, r = CUDA.threadIdx().x, CUDA.blockIdx().x

#     if rand()<cooler
#         return nothing
#     else

#     x0 = sign(R[b,4,r])*sqrt(1-ϵ*ϵ)

#     normx = sqrt(R[b,1,r]*R[b,1,r] + R[b,2,r]*R[b,2,r] + R[b,3,r]*R[b,3,r])
#     x1 = ϵ*R[b,1,r]/normx
#     x2 = ϵ*R[b,2,r]/normx
#     x3 = ϵ*R[b,3,r]/normx

#     # Extract components of Π
#     π1 = Pi[b,r].t1
#     π2 = Pi[b,r].t2
#     π3 = Pi[b,r].t3

#     Sigma[b,r] = x0*Sigma[b,r] + x1*π1 + x2*π2 + x3*π3
#     Pi1   = x0*π1 +  Sigma[b,r]*x1 + x2*π3 - x3*π2  
#     Pi2   = x0*π2 +  Sigma[b,r]*x2 + x3*π1 - x1*π3  
#     Pi3   = x0*π3 +  Sigma[b,r]*x3 + x1*π2 - x2*π1  

#     Pi[b,r] = SU2alg(Pi1,Pi2,Pi3)

#     return nothing
# end


function MetropolisUpdate!(U,Sigma,Pi,ϵ,cooler,lp,gp,zp,ymws,zws,noacc=false)
    @timeit "Metropolis update" begin
        ymws.U1   .= U
        zws.Sigma .= Sigma
        zws.Pi    .= Pi

        # Compute initial action
        Sin = gauge_action(U,lp,gp,ymws)

        # Propose a change
        sweep_gauge!(U,lp,ymws,ϵ,cooler)
        # sweep_Z!(Sigma,Pi,ϵ,lp,ymws)


        # Compute action difference
        ΔS = gauge_action(U,lp,gp,ymws) - Sin
        
        # Acc/rej step
        acc = true
        if noacc
            return ΔS, acc
        end

        if rand()>exp(-ΔS)
            U     .= ymws.U1
            Sigma .= zws.Sigma
            Pi    .= zws.Pi
            acc    = false
        end

    end
    return ΔS, acc
end