###
### "THE BEER-WARE LICENSE":
### Alberto Ramos wrote this file. As long as you retain this 
### notice you can do whatever you want with this stuff. If we meet some 
### day, and you think this stuff is worth it, you can buy me a beer in 
### return. <alberto.ramos@cern.ch>
###
### file:    YM.jl
### created: Mon Jul 12 16:23:51 2021
###                               


module YM

using CUDA, Random, StructArrays, TimerOutputs, BDIO
using ..Space
using ..Groups
using ..Fields
using ..MD

import Base.show

struct GaugeParm{T,G,N}
    beta::T
    c0::T
    cG::NTuple{2,T}
    ng::Int64

    Ubnd::NTuple{N, G}

    GaugeParm{T1,T2,T3}(a,b,c,d,e) where {T1,T2,T3} = new{T1,T2,T3}(a,b,c,d,e)
    function GaugeParm{T}(::Type{G}, bt, c0, cG, phi, iL) where {T,G}

        degree(::Type{SU2{T}}) where T <: AbstractFloat = 2
        degree(::Type{SU3{T}}) where T <: AbstractFloat = 3
        ng = degree(G)
        nsd = length(iL)

        return new{T,G,nsd}(bt, c0, cG, ng, ntuple(id->bndfield(phi[1], phi[2], iL[id]), nsd))
    end
    function GaugeParm{T}(::Type{G}, bt, c0) where {T,G}

        degree(::Type{SU2{T}}) where T <: AbstractFloat = 2
        degree(::Type{SU3{T}}) where T <: AbstractFloat = 3
        ng = degree(G)

        return new{T,G,0}(bt, c0, (0.0,0.0), ng, ())
    end
end
export GaugeParm
function Base.show(io::IO, gp::GaugeParm{T, G, N}) where {T,G,N}

    println(io, "Group:  ", G)
    println(io, " - beta:              ", gp.beta)
    println(io, " - c0:                ", gp.c0)
    println(io, " - cG:                ", gp.cG)
    if (N > 0)
        for i in 1:N
            println(io, "   - Boundary link:     ", gp.Ubnd[i])
        end
    end

    return nothing
end

struct YMworkspace{T}
    GRP
    ALG
    PRC
    frc1
    frc2
    mom
    U1
    cm # complex of volume
    rm # float   of volume
    function YMworkspace(::Type{G}, ::Type{T}, lp::SpaceParm) where {G <: Group, T <: AbstractFloat}
        
        @timeit "Allocating YMWorkspace" begin
            if (G == SU2)
                GRP = SU2
                ALG = SU2alg
                f1 = vector_field(SU2alg{T}, lp)
                f2 = vector_field(SU2alg{T}, lp)
                mm = vector_field(SU2alg{T}, lp)
                u1 = vector_field(SU2{T},    lp)
            end
            
            if (G == SU3)
                GRP = SU3
                ALG = SU3alg
                f1 = vector_field(SU3alg{T}, lp)
                f2 = vector_field(SU3alg{T}, lp)
                mm = vector_field(SU3alg{T}, lp)
                u1 = vector_field(SU3{T},    lp)
            end
            cs = scalar_field_point(Complex{T}, lp)
            rs = scalar_field_point(T, lp)
        end
            
        return new{T}(GRP,ALG,T,f1, f2, mm, u1, cs, rs)
    end
end
export YMworkspace
function Base.show(io::IO, ymws::YMworkspace)
    
    println(io, "Workspace for Group:   ", ymws.GRP)
    println(io, "              Algebra: ", ymws.ALG)
    println(io, "Precision:             ", ymws.PRC)
    if ymws.fpln == nothing
        println(io, "  - Running in memory efficient mode")
    else
        println(io, "  - Running in computing efficient mode")
    end
    return nothing
end


function ztwist(gp::GaugeParm{T,G}, lp::SpaceParm{N,M,B,D}) where {T,G,N,M,B,D}

    function plnf(ipl)
        id1, id2 = lp.plidx[ipl]
        return convert(Complex{T},exp(2im * pi * lp.ntw[ipl]/(lp.iL[id1]*lp.iL[id2]*gp.ng)))
    end

    return ntuple(i->plnf(i), M)
end

function ztwist(gp::GaugeParm{T,G}, lp::SpaceParm{N,M,B,D}, ipl::Int) where {T,G,N,M,B,D}

    id1, id2 = lp.plidx[ipl]
    return convert(Complex{T},exp(2im * pi * lp.ntw[ipl]/(lp.iL[id1]*lp.iL[id2]*gp.ng)))
end
export ztwist

include("YMfields.jl")
export randomize!, zero!, norm2

include("YMact.jl")
export krnl_plaq!, force0_wilson!

include("YMhmc.jl")
export gauge_action, hamiltonian, plaquette, HMC!, OMF4!

include("YMflow.jl")
export FlowIntr, flw, flw_adapt
export Eoft_clover, Eoft_plaq, Qtop, dEoft_clover, dEoft_plaq
export FlowIntr, wfl_euler, zfl_euler, wfl_rk2, zfl_rk2, wfl_rk3, zfl_rk3

include("YMsf.jl")
export sfcoupling, bndfield, setbndfield

include("YMio.jl")
export import_lex64, import_cern64, save_cnfg, read_cnfg

end
