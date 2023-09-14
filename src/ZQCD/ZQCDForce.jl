        ###
### "THE BEER-WARE LICENSE":
### Alberto Ramos wrote this file. As long as you retain this 
### notice you can do whatever you want with this stuff. If we meet some 
### day, and you think this stuff is worth it, you can buy me aa beer in 
### return. <alberto.ramos@cern.ch>
###
### author  pietro.butti.fl@gmail.com
### file:    ZQCDForce.jl
### created: Wed Oct  6 15:39:07 2021
###                               

function force_zqcd(ymws::YMworkspace, zws::ZQCDWorkspace, U, Sigma, Pi, zp::ZQCDParm, gp::GaugeParm, lp::SpaceParm)

    @timeit "ZQCD force" begin
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_force_zqcd!(ymws.frc1,zws.frc,U,Sigma,Pi,zp,gp,lp)
        end
    end

    return nothing
end


function krnl_force_zqcd!(fgauge,fZ, U::AbstractArray{TG}, Sigma::AbstractArray{TS}, Pi::AbstractArray{TP}, zp::ZQCDParm{T}, gp::GaugeParm, lp::SpaceParm{N,M,B,D}) where {TG,TS,TP,T,N,M,B,D}

    # Square mapping to CUDA block
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)

    Pi2 = 4. * norm2(Pi[b,r])
    Pi3 = 8. * (Pi[b,r].t1^3 + Pi[b,r].t2^3 + Pi[b,r].t3^3)

    # Compute force for Σ -------------------------------------------------------
        fZ[b,1,r] = (6. + 2. *zp.b1 + 2. *zp.c3 * Pi2)*Sigma[b,r] + 
            4. * zp.c1 * Sigma[b,r]*Sigma[b,r]*Sigma[b,r]

        for dir in 1:N
            # Fetch the coordinates of point after and before in direction dir
            up_b, up_r, dw_b, dw_r = updw((b,r),dir,lp)

            fZ[b,1,r] -= Sigma[up_b,up_r] - Sigma[dw_b,dw_r]
        end
        fZ[b,1,r] *= 4. /gp.beta

    # Compute force for Π ---------------------------------------------------------
    for aa in 1:3
        # Fetch the field component
        Pia = 2. * (aa==1 ? Pi[b,r].t1 : (aa==2 ? Pi[b,r].t2 : Pi[b,r].t3)) #(maybe better with symbolic notation?)
        
        # fZ[b,aa,r] = 2. * zp.b2 * Pia + 
        #     2. * zp.c3 * Sigma[b,r] * Pi2 +
        #     4. * zp.c2 * Pi3

        # for bb in 1:3
        #     if bb!=aa
        #         Pib = 2. * (bb==1 ? Pi[b,r].t1 : (bb==2 ? Pi[b,r].t2 : Pi[b,r].t3))
        #         fZ[b,aa,r] += Pia*Pib*Pib
        #     end
        # end

        # algipau(Pauli{aa},Pi[b,r]) + algipau(Pi[b,r],Pauli{aa})
        
        # fZ[b,aa,r] *= 4. / gp.beta



        # Potential parameters
        FF = 0 ## da cambiare di brutto
        for bb in 1:3
            if bb!=aa
                Pib = 2. * (bb==1 ? Pi[b,r].t1 : (bb==2 ? Pi[b,r].t2 : Pi[b,r].t3))
                FF += Pib^2
            end
        end
        FF *= Pia 
        FF += Pia^2
        FF *= 4. * zp.c2 *
        FF += 4. / gp.beta * ( 2. * (zp.b2 + zp.c3*Sigma[b,r]) * Pia )

        #...

        return nothing
    end



end