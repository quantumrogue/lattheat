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

function zqcd_force(ymws::YMworkspace, zws::ZQCDworkspace, U, Sigma, Pi, zp::ZQCDParm, gp::GaugeParm, lp::SpaceParm)

    @timeit "ZQCD force" begin
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_zqcd_force!(ymws.frc1,zws.frcSigma,zws.frcPi,U,Sigma,Pi,zp,gp,lp)
        end
    end

    return nothing
end


function krnl_zqcd_force!(fgauge,fSigma,fPi, U::AbstractArray{TG}, Sigma::AbstractArray{TS}, Pi::AbstractArray{TP}, zp::ZQCDParm{T}, gp::GaugeParm, lp::SpaceParm{N,M,B,D}) where {TG,TS,TP,T,N,M,B,D}
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
            UPiU = U[b,dir,r] * Pi[b_up,r_up] * dag(U[b,dir,r])

            fgauge[b,dir,r] +=  projalg(8. * im / gp.beta, UPiU * Pi[b,r])
            fgauge[b,dir,r] -=  projalg(8. * im / gp.beta, Pi[b,r] * UPiU)
        end
    # -----------------------------------------------------------------------------


    Pi2 = norm2(Pi[b,r])

    # Compute force for Σ -------------------------------------------------------
        fSigma[b,r] = zero(TS)
        fSigma[b,r] = (6. + 2. *zp.b1 + 2. *zp.c3 * Pi2)*Sigma[b,r] + 
            4. * zp.c1 * Sigma[b,r]*Sigma[b,r]*Sigma[b,r]

        for dir in 1:N
            up_b, up_r, dw_b, dw_r = updw((b,r),dir,lp)

            fSigma[b,r] -= Sigma[up_b,up_r] - Sigma[dw_b,dw_r]
        end
        fSigma[b,r] *= 4. /gp.beta
    # -----------------------------------------------------------------------------


    # Compute force for Π ---------------------------------------------------------
        for dir in 1:N
            up_b, up_r, dw_b, dw_r = updw((b,r),dir,lp)
            fPi[b,r] -= 8. ./ gp.beta * projalg(
                U[b,dir,r]*Pi[up_b,up_r]*dag(U[b,dir,r])  +  dag(U[b,dir,r])*Pi[dw_b,dw_r]*U[b,dir,r]
            )
        end
       fPi[b,r] -= 16. / gp.beta * zp.c2 * (-Pi2/4.) * ((zp.b2 + zp.c3*Sigma[b,r])/zp.c2 - 8. * (-Pi2/4.)) * Pi[b,r]
    # -----------------------------------------------------------------------------

    return nothing
end