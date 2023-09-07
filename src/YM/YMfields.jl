###
### "THE BEER-WARE LICENSE":
### Alberto Ramos wrote this file. As long as you retain this 
### notice you can do whatever you want with this stuff. If we meet some 
### day, and you think this stuff is worth it, you can buy me a beer in 
### return. <alberto.ramos@cern.ch>
###
### file:    YMfields.jl
### created: Thu Jul 15 15:16:47 2021
###                               

function randomize!(f, lp::SpaceParm, ymws::YMworkspace)

    if ymws.ALG == SU2alg
        @timeit "Randomize SU(2) algebra field" begin
            m = CUDA.randn(ymws.PRC, lp.bsz,lp.ndim,3,lp.rsz)
            CUDA.@sync begin
                CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_assign_SU2!(f,m,lp)
            end
        end
        return nothing
    end

    if ymws.ALG == SU3alg
        @timeit "Randomize SU(3) algebra field" begin
            m = CUDA.randn(ymws.PRC, lp.bsz,lp.ndim,8,lp.rsz)
            CUDA.@sync begin
                CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_assign_SU3!(f,m,lp)
            end
        end
        return nothing
    end

    return nothing
end

function krnl_assign_SU3!(frc::AbstractArray{T}, m, lp::SpaceParm{N,M,BC_PERIODIC,D}) where {T,N,M,D}

    b, r = CUDA.threadIdx().x, CUDA.blockIdx().x
    for id in 1:lp.ndim
        frc[b,id,r] = SU3alg(m[b,id,1,r], m[b,id,2,r], m[b,id,3,r],
                             m[b,id,4,r], m[b,id,5,r], m[b,id,6,r],
                             m[b,id,7,r], m[b,id,8,r])
    end
    return nothing
end

function krnl_assign_SU3!(frc::AbstractArray{T}, m, lp::SpaceParm{N,M,B,D}) where {T,N,M,B,D}

    b, r = CUDA.threadIdx().x, CUDA.blockIdx().x
    it = point_time((b,r), lp)

    if ((B==BC_SF_AFWB)||(B==BC_SF_ORBI))
        if it == 1
            for id in 1:lp.ndim-1
                frc[b,id,r] = zero(T)
            end
            frc[b,N,r] = SU3alg(m[b,N,1,r], m[b,N,2,r], m[b,N,3,r],
                                m[b,N,4,r], m[b,N,5,r], m[b,N,6,r],
                                m[b,N,7,r], m[b,N,8,r])
        else
            for id in 1:lp.ndim
                frc[b,id,r] = SU3alg(m[b,id,1,r], m[b,id,2,r], m[b,id,3,r],
                                     m[b,id,4,r], m[b,id,5,r], m[b,id,6,r],
                                     m[b,id,7,r], m[b,id,8,r])
            end
        end
    end

    return nothing
end

function krnl_assign_SU2!(frc, m, lp::SpaceParm{N,M,BC_PERIODIC,D}) where {N,M,D}

    b, r = CUDA.threadIdx().x, CUDA.blockIdx().x
    for id in 1:lp.ndim
        frc[b,id,r] = SU2alg(m[b,id,1,r], m[b,id,2,r], m[b,id,3,r])
    end
    return nothing
end
