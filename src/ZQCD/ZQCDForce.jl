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

function zqcd_force(ymws::YMworkspace, zws::ZQCDWorkspace, U, Sigma, Pi, zp::ZQCDParm, gp::GaugeParm, lp::SpaceParm)

    @timeit "ZQCD force" begin
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_zqcd_force!(ymws.frc1,zws.frc,U,Sigma,Pi,zp,gp,lp)
        end
    end

    return nothing
end


function krnl_zqcd_force!(fgauge,fSigma,fPi, U::AbstractArray{TG}, Sigma::AbstractArray{TZ}, Pi::AbstractArray{TZ}, zp::ZQCDParm{T}, gp::GaugeParm, lp::SpaceParm{N,M,B,D}) where {TG,TZ,T,N,M,B,D}

    # Square mapping to CUDA block
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    
    # Compute gauge force ----------------------------------------------------------
        for id in 1:N
            fgauge[b,id,r] = (gp.beta/gp.ng)*fgauge[b,id,r]
        end
        sync_threads()

        # ZQCD gauge force
        for dir in 1:N
            b_up, r_up = up((b, r), dir, lp)
            UPiU = U[b,dir,r] * Pi[b_up,r_up] / U[b,dir,r]
            fgauge[b,dir,r] -=  8. / gp.beta * projalg(UPiU * Pi[b,r])
            fgauge[b,dir,r] +=  8. / gp.beta * projalg(Pi[b,r] * UPiU)
        end
    # -----------------------------------------------------------------------------


    Pi2 = norm2(Pi[b,r])

    # Compute force for Σ -------------------------------------------------------
        fSigma = zero(TZ)
        fSigma = (6. + 2. *zp.b1 + 2. *zp.c3 * Pi2)*Sigma[b,r] + 
            4. * zp.c1 * Sigma[b,r]*Sigma[b,r]*Sigma[b,r]

        for dir in 1:N
            # Fetch the coordinates of point after and before in direction dir
            up_b, up_r, dw_b, dw_r = updw((b,r),dir,lp)

            fSigma -= Sigma[up_b,up_r] - Sigma[dw_b,dw_r]
        end
        fSigma *= 4. /gp.beta
    # -----------------------------------------------------------------------------

    # Compute force for Π ---------------------------------------------------------
        aux = zeros(TZ,3)
        for aa in 1:3
            # Fetch the field component
            Pia = 2. * (aa==1 ? Pi[b,r].t1 : (aa==2 ? Pi[b,r].t2 : Pi[b,r].t3)) #(maybe better with symbolic notation?)
            
            # Potential term
            for bb in 1:3
                if bb!=aa
                    Pib = 2. * (bb==1 ? Pi[b,r].t1 : (bb==2 ? Pi[b,r].t2 : Pi[b,r].t3))
                    aux[aa] += Pib^2
                end
            end
            aux[aa] *= Pia 
            aux[aa] += Pia^3
            aux[aa] *= 4. * zp.c2
            aux[aa] += 2. * (zp.b2 + zp.c3*Sigma[b,r]) * Pia
            aux[aa] *= 4. / gp.beta

            # Kinetic term
            for dir in 1:N
                up_b, up_r, dw_b, dw_r = updw((b,r),dir,lp)
                aux[aa] -= 8. / gp.beta * tr_ipau(        U[b,dir,r]  * Pi[up_b,dir,up_r] / U[Pi[b,dir,r]],Pauli{aa})
                aux[aa] -= 8. / gp.beta * tr_ipau(inverse(U[b,dir,r]) * Pi[dw_b,dir,dw_r] * U[Pi[up_b,dir,up_r]],Pauli{aa})
            end
        end
        fPi[b,r] += SU2alg{T}(aux...)
    # -----------------------------------------------------------------------------

    return nothing
end