###
### "THE BEER-WARE LICENSE":
### Alberto Ramos wrote this file. As long as you retain this
### notice you can do whatever you want with this stuff. If we meet some
### day, and you think this stuff is worth it, you can buy me a beer in
### return. <>
###
### file:    YMact.jl
### created: Mon Jul 12 18:31:19 2021
###


"""
    computes global observables by calling krnl_obs! and summing
    for all lattice points
"""

function scalar_obs(U, Phi, sp::ScalarParm{NP,T}, lp::SpaceParm, ymws::YMworkspace) where {NP,T}

    @timeit "Scalar observables" begin
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_obs!(ymws.rm, ymws.cm, U, Phi, sp, lp)
        end
        
        V = prod(lp.iL)
        #summation of global observables
        rho2   = CUDA.mapreduce(norm2, +, Phi)/(V*NP)
        lphi   = CUDA.reduce(+, ymws.rm)/(lp.ndim*V*NP)
        lalpha = CUDA.mapreduce(real, +, ymws.cm)/(lp.ndim*V*NP)
    end
    
    return rho2, lphi, lalpha
end


function scalar_obs(U, Phi, isc1::Int64, isc2::Int64, sp::ScalarParm{NP,T}, lp::SpaceParm, ymws::YMworkspace) where {NP,T}

    @timeit "Scalar observables" begin
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_obs!(ymws.rm, ymws.cm, U, Phi, isc1, isc2, sp, lp)
        end

        V = prod(lp.iL)
        #summation of global observables
        rho2   = CUDA.mapreduce(imag, +, ymws.cm)/V
        lphi   = CUDA.reduce(+, ymws.rm)/(lp.ndim*V)
        lalpha = CUDA.mapreduce(real, +, ymws.cm)/(lp.ndim*V)
    end
    
    return rho2, lphi, lalpha
end
scalar_obs(U, Phi, isc::Int64, sp::ScalarParm{NP,T}, lp::SpaceParm, ymws::YMworkspace) where {NP,T} = scalar_obs(U, Phi, isc, isc, sp, lp, ymws)

"""
    CUDA function to compute the observables defined in the Obs struct
    for each lattice point
"""

function krnl_obs!(rm, cm, U::AbstractArray{TG}, Phi::AbstractArray{TS}, sp::ScalarParm{NP,T}, lp::SpaceParm{N,M,B,D}) where {TG,TS,NP,T,N,M,B,D}

    #thread/block coordinate
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)

    IX = point_coord((b,r), lp)

    rm[IX] = zero(eltype(rm))
    cm[IX] = zero(eltype(cm))
    #compute obs
    for i in 1:NP
        psq = norm( Phi[b,i,r] )
        for id in 1:N
            bu, ru = up((b, r), id, lp)

            rm[IX] += dot( Phi[b,i,r], U[b,id,r]*Phi[bu,i,ru] )
            cm[IX] += complex(rm[IX])/(psq*norm(Phi[bu,i,ru]))
        end
    end
    return nothing
end

function krnl_obs!(rm, cm, U::AbstractArray{TG}, Phi::AbstractArray{TS}, isc1::Int64, isc2::Int64, sp::ScalarParm{NP,T}, lp::SpaceParm{N,M,B,D}) where {TG,TS,NP,T,N,M,B,D}

    #thread/block coordinate
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)

    IX = point_coord((b,r), lp)

    rm[IX] = zero(eltype(rm))
    cm[IX] = zero(eltype(cm))
    #compute obs
    psq  = norm( Phi[b,isc1,r] )
    for id in 1:N
        bu, ru = up((b, r), id, lp)
        # dot introduces a factor 2 compared to the trace
        # dot( A, B) = 2 Tr(AB)
        rm[IX] += dot( Phi[b,isc1,r], U[b,id,r]*Phi[bu,isc2,ru] )
        cm[IX] += complex(rm[IX]/(psq*norm(Phi[bu,isc2,ru])))
    end
    cm[IX] += complex( zero(T), dot(Phi[b,isc1,r], Phi[b,isc2,r]) )
    return nothing
end
krnl_obs!(rm, cm, U::AbstractArray{TG}, Phi::AbstractArray{TS}, isc::Int64, sp::ScalarParm{NP,T}, lp::SpaceParm{N,M,B,D}) where {TG,TS,NP,T,N,M,B,D} = krnl_obs!(rm, cm, U, Phi, isc, isc, sp, lp)


"""
    Flavour symmetric Correlation functions
    Higgs phi_i * U * phi_i
    W-boson phi_i * U * phi_i * tau

"""

function scalar_corr(U, Phi, isc::Int64, sp::ScalarParm{NP,T}, lp::SpaceParm, ymws::YMworkspace, gp::GaugeParm, sws::ScalarWorkspace) where {NP,T}

    @timeit "Scalar correlation" begin

        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_corr!(ymws.rm, U, Phi, isc, sp, lp)
        end

        tp = (1,2,3)
        V3 = prod(lp.iL[1:end-1])
        #summation of spatial lattice - momentum interpolator
        h2 = Vector{T}(undef, lp.iL[end])
        #normalization for spatial directions only
        h2 = reshape(Array(CUDA.reduce(+, ymws.rm;dims=tp)), lp.iL[end])/(V3*(lp.ndim-1))

        w1 = Array{T,3}(undef, 3, 3, lp.iL[end])
        # vector boson
        for mu in 1:3
            for i in 1:3
                CUDA.@sync begin
                    CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_corr!(ymws.cm, mu, Pauli{i}, U, Phi, isc, isc, sp, lp)
                end
                #su(2) trace is always real
                w1[mu, i, :] .= reshape(Array(CUDA.mapreduce(real, +, ymws.cm;dims=tp)), lp.iL[end]) ./ V3
            end
        end
    end

    return h2,w1
end

"""
    Flavour mixed Correlation functions - W-boson only
    W-boson phi_1 * U * phi_2 * tau

"""


function mixed_corr(U, Phi, sp::ScalarParm{NP,T}, lp::SpaceParm, ymws::YMworkspace, gp::GaugeParm, sws::ScalarWorkspace) where {NP,T}

    @timeit "Scalar correlation" begin

        tp = (1,2,3)
        V3 = prod(lp.iL[1:end-1])

        w12 = Array{T,3}(undef, 3, 3, lp.iL[end])
        # vector boson
        for mu in 1:3
            for i in 1:3
                CUDA.@sync begin
                    CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_corr!(ymws.cm, mu, Pauli{i}, U, Phi, 1, 2, sp, lp)
                end
                #su(2) trace is always real
                w12[mu, i, :] .= reshape(Array(CUDA.mapreduce(real, +, ymws.cm;dims=tp)), lp.iL[end]) ./ V3
            end
        end
    end

    return w12

end


function smearing(U, Phi, smear::smr, sp::ScalarParm{NP,T}, lp::SpaceParm, ymws::YMworkspace, gp::GaugeParm, sws::ScalarWorkspace) where {NP,T}

    #smear gauge fields
    flw(U, smear.flwint, smear.sus, gp, lp, ymws)

    #smear scalar fields
    for s in 1:smear.n
        # compute Laplacian
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_laplacian!(sws.frc1,U,Phi,sp,gp,lp)
        end
        # update scalar fields

        for i in 1:NP
            Phi[:,i,:] .= Phi[:,i,:] + smear.r .* sws.frc1[:,i,:]
        end
    end
end

function krnl_laplacian!(fscalar, U::AbstractArray{TG}, Phi::AbstractArray{TS}, sp::ScalarParm{NP,T}, gp::GaugeParm, lp::SpaceParm{N,M,B,D}) where {TG,TS,NP,T,N,M,B,D}

    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    sync_threads()

    #cycle scalar flavour
    for i in 1:NP
        #compute laplacian
        fscalar[b,i,r] = -2.0*(N-1)*Phi[b,i,r]
        #spatial directions only
        for id in 1:N-1
            bu, ru = up((b,r), id, lp)
            bd, rd = dw((b,r), id, lp)

            fscalar[b,i,r] += U[b,id,r]*Phi[bu,i,ru] + dag(U[bd,id,rd])*Phi[bd,i,rd]
        end
    end
    return nothing
end


"""
    Higgs
    Computes the zero momentum interpolator for an observable

         Questions:
            - How to generalize kernel to work for any observable?
            - maybe compute all correlators in this function (?)
"""
function krnl_corr!(rm, U::AbstractArray{TG}, Phi::AbstractArray{TS}, isc::Int64, sp::ScalarParm{NP,T}, lp::SpaceParm{N,M,B,D}) where {TG,TS,NP,T,N,M,B,D}

    #thread/block coordinate
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    IX = point_coord((b,r), lp)

    rm[IX] = zero(eltype(rm))
    #
    #compute interpolator for each point
    for id in 1:N-1
        bu, ru = up((b, r), id, lp)
        # dot introduces a factor 1/2 compared to the trace
        # dot( A, B) = 2 Tr(A*B)
        rm[IX] += dot( Phi[b,isc,r], U[b,id,r]*Phi[bu,isc,ru] )
    end
    return nothing
end


"""
    Vector W-Boson
"""
function krnl_corr!(cm, mu::Int64, pli::Type{Pauli{k}}, U::AbstractArray{TG}, Phi::AbstractArray{TS}, isc1::Int64, isc2::Int64, sp::ScalarParm{NP,T}, lp::SpaceParm{N,M,B,D}) where {TG,TS,NP,T,N,M,B,D,k}
    #mu: spatial direction interpolator
    #pli: pauli matrix type for multiplication
    #TG, TS: Types - fields
    #D: nr sites per block
    #N: dimensionality

    #thread/block coordinate
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    IX = point_coord((b,r), lp)
    cm[IX] = zero(eltype(cm))

    #compute interpolator for each point
    for s in 1:1
        bu, ru = up((b, r), mu, lp)
        cm[IX] += tr( fundXpauli( (Phi[b,isc1,r]\U[b,mu,r]) * Phi[bu,isc2,ru], pli) )
    end
    return nothing
end
