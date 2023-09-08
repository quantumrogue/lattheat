###
### "THE BEER-WARE LICENSE":
### Alberto Ramos wrote this file. As long as you retain this 
### notice you can do whatever you want with this stuff. If we meet some 
### day, and you think this stuff is worth it, you can buy me aa beer in 
### return. <alberto.ramos@cern.ch>
###
### author  pietro.butti.fl@gmail.com
### file:    AdjScalarForce.jl
### created: Wed Oct  6 15:39:07 2021
###                               

function force_adj_scalar(ymws::YMworkspace, sws::AdjScalarWorkspace, U, Sigma, Pi, sp::AdjScalarParm, gp::GaugeParm, lp::SpaceParm)

    # @timeit "Scalar force" begin
    #     CUDA.@sync begin
    #         CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_force_scalar!(ymws.frc1,sws.frc1,U,Phi,sp,gp,lp)
    #     end
    # end

    return nothing
end


function krnl_force_adj_scalar!(fgauge,fZ, U::AbstractArray{TG}, Sigma::AbstractArray{TS}, Pi::AbstractArray{TP}, sp::AdjScalarParm{T}, gp::GaugeParm, lp::SpaceParm{N,M,B,D}) where {TG,TS,TP,T,N,M,B,D}

    # Square mapping to CUDA block
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)

    Pi2 = 4. * norm2(Pi[b,r])
    Pi3 = 8. * (Pi[b,r].t1^3 + Pi[b,r].t2^3 + Pi[b,r].t3^3)

    # Compute force for Σ -------------------------------------------------------
        fZ[b,1,r] = (6. + sp.b1 + sp.c3 * Pi2*Pi2)*Sigma[b,r] + 
            2. * sp.c1 * Sigma[b,r]*Sigma[b,r]*Sigma[b,r]

        for dir in 1:N
            # Fetch the coordinates of point after and before in direction dir
            up_b, up_r, dw_b, dw_r = updw((b,r),dir,lp)

            fZ[b,1,r] -= Sigma[up_b,up_r] - Sigma[dw_b,dw_r]
        end
        fZ[b,1,r] *= 8. /gp.beta

    # Compute force for Π ---------------------------------------------------------
    for aa in 1:3
        # Fetch the field component
        Pia = 2. * (aa==1 ? Pi[b,r].t1 : (aa==2 ? Pi[b,r].t2 : Pi[b,r].t3)) #(maybe better with symbolic notation?)
        
        fZ[b,aa,r] = 2. * sp.b2 * Pia + 
            2. * sp.c3 * Sigma[b,r] * Pi2 +
            4. * sp.c2 * Pi3

        for bb in 1:3
            if bb!=aa
                Pib = 2. * (bb==1 ? Pi[b,r].t1 : (bb==2 ? Pi[b,r].t2 : Pi[b,r].t3))
                fZ[b,aa,r] += Pia*Pib*Pib
            end
        end

        algipau(Pauli{aa},Pi[b,r]) + algipau(Pi[b,r],Pauli{aa})
        
        fZ[b,aa,r] *= 4. / gp.beta
    end



end