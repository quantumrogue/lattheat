###
### "THE BEER-WARE LICENSE":
### Alberto Ramos wrote this file. As long as you retain this 
### notice you can do whatever you want with this stuff. If we meet some 
### day, and you think this stuff is worth it, you can buy me a beer in 
### return. <alberto.ramos@cern.ch>
###
### author:  pietro.butti.fl@gmail.com
### file:    ZQCDFields.jl
### created: Tue 12 Sep 12:15:09 CEST 2023
###             


function unitarize!(SIGMA,PI, lp::SpaceParm)
    @timeit "reunitarize Z field" begin
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_unitarize_Z!(SIGMA,PI)
        end        
    end
    return nothing
end

function krnl_unitarize_Z!(SIGMA::AbstractArray{T},PI::AbstractArray{PT}) where {T,PT}
    b, r = CUDA.threadIdx().x, CUDA.blockIdx().x
    SIGMA[b,r] = sqrt(convert(T,4.) - norm2(PI[b,r]))
    return nothing
end




function randomize!(SIGMA,PI, lp::SpaceParm, ymws::YMworkspace)

    @timeit "Randomize ZQCD field" begin
        m = CUDA.randn(ymws.PRC, lp.bsz, 4, lp.rsz)
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_assign_ZQCD!(SIGMA,PI,m)
        end
    end

    return nothing
end

function krnl_assign_ZQCD!(Σ::AbstractArray{T},Π,m::AbstractArray{T}) where T 
    b, r = CUDA.threadIdx().x, CUDA.blockIdx().x

    Σ[b,r] = m[b,1,r]
    Π[b,r] = SU2alg(m[b,2,r],m[b,3,r],m[b,4,r])

    return nothing
end
