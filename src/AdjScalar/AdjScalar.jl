###
### "THE BEER-WARE LICENSE":
### Alberto Ramos wrote this file. As long as you retain this 
### notice you can do whatever you want with this stuff. If we meet some 
### day, and you think this stuff is worth it, you can buy me a beer in 
### return. <alberto.ramos@cern.ch>
###
### author:  pietro.butti.fl@gmail.com
### file:    AdjScalar.jl
### created: Fri  8 Sep 2023 10:53:35 CEST
###                               

module Scalar

using CUDA, Random, TimerOutputs
using ..Space
using ..Groups
using ..Fields
using ..MD
using ..YM
import ..YM: HMC!, randomize!, MD!, force_gauge, FlowIntr

import Base.show

struct ScalarAdjParm{T}
    b1::T
    b2::T
    c1::T
    c2::T
    c3::T
end

"""
    ScalarAdjParm{T}(r2::T,g2::T) where T

2-argument (overridden) constructor for `ScalarAdjParm`. `c1`, `c1`, `c1` are set to their numerical values given by formulas (3.10-3.12) of arXiv:0801.1566v2, while `b1` and `b2` are functions of the gauge coupling β given by formulas (C.9) and (C.10)
"""
function ScalarAdjParm{T}(r2::T,g2::T,beta::T) where T
    # _c1 =
    # _c2 =
    # _c3 =
    # _b1 = 2.38193365/4/π * ( 2*_c1 + _c3 ) / beta ...
    # _b2 = ...
    # 
    # return new{T}(_b1,_b2,_c1,_c2,_c3)
end

function Base.show(io::IO, sp::ScalarAdjParm{T}) where T
    # ...
end
export ScalarAdjParm

struct ScalarAdjWorkspace{T}
    frc
    mom
    Sigma
    Pi
    ScalarAdjWorkspace(::Type{T}, lp::SpaceParm) where {T<:AbstractFloat} = new{T}(
        nscalar_field(T,4,lp)      # F_Π : 3 scalar fields
        nscalar_field(T,4,lp)      # mom_α : 4 scalar field
        scalar_field(T,lp),        # Σ : a real field
        scalar_field(SU2alg{T},lp) # Π : (3 scalar field ∼) SU2 algebra matrix (Π = i Πₐ⋅σₐ)
    )
end
export ScalarAdjWorkspace


include("AdjScalarAction.jl")
    export adj_scalar_action

end
