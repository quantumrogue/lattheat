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

function randomize!(f, lp::SpaceParm, ymws::YMworkspace, zws::ZQCDworkspace)
    
    @timeit "Randomize gauge field" begin
        m = CUDA.randn(ymws.PRC, lp.bsz, 3, lp.rsz)
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_assign_SU2!(f,m,lp)
        end
    end

    @timeit "Randomize ZQCD field" begin
        m = CUDA.randn(ymws.PRC, lp.bsz, 4, lp.rsz)
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_assign_SU2!(f,m,sp,lp)
        end
    end

    return nothing
end

function krnl_assign_ZQCD!(Z::AbstractArray{T}, m, sp::ScalarParm{NS}, lp::SpaceParm) where {T, NS}

    # Think about precision here
    SR2::eltype(sp.kap) = 1.4142135623730951
    
    b, r = CUDA.threadIdx().x, CUDA.blockIdx().x
    for i in 1:NS
        f[b,i,r] = SU2fund(complex(m[b,1,i,r]*SR2, m[b,2,i,r]*SR2),
                           complex(m[b,3,i,r]*SR2, m[b,4,i,r]*SR2))
    end

    return nothing
end
