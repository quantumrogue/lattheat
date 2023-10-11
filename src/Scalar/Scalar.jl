###
### "THE BEER-WARE LICENSE":
### Alberto Ramos wrote this file. As long as you retain this 
### notice you can do whatever you want with this stuff. If we meet some 
### day, and you think this stuff is worth it, you can buy me a beer in 
### return. <alberto.ramos@cern.ch>
###
### file:    Scalar.jl
### created: Tue Oct  5 11:53:31 2021
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

struct ScalarParm{N,T}
    kap::NTuple{N,T}
    eta::NTuple{N,T}
    muh::T
    xi::NTuple{5,T}
    
    function ScalarParm(kv::NTuple{2,T}, ev::NTuple{2,T}, mu::T, xi::NTuple{5,T}) where T


        return new{2,T}(kv,ev,mu, xi)
    end
    function ScalarParm(kv::NTuple{N,T}, ev::NTuple{N,T}) where {N,T}


        return new{N,T}(kv,ev,0.0,ntuple(i->0.0,4))
    end
end

function Base.show(io::IO, sp::ScalarParm{N,T}) where {N,T}

    println(io, "Number of scalar fields: ", N)
    print(io, " - Kappas: ")
    for i in 1:N
        print(io, " ", sp.kap[i])
    end
    print("\n - etas:   ")
    for i in 1:N
        print(io, " ", sp.eta[i])
    end
    if N == 2
        print("\n - mu12:   ")
        print(io, " ", sp.muh)
        print("\n - xi:    ")
        print(io, " ", sp.xi)
    end
    println("\n")
    

    
end
export ScalarParm

struct ScalarWorkspace{T}
    frc1
    mom
    Phi
    function ScalarWorkspace(::Type{T}, n, lp::SpaceParm) where {T <: AbstractFloat}
        return new{T}(nscalar_field(SU2fund{T}, n, lp),
                      nscalar_field(SU2fund{T}, n, lp),
                      nscalar_field(SU2fund{T}, n, lp))
    end
end
export ScalarWorkspace

# Smearing info
struct smr{T}
    #link smearing
    sus::Int64 #gradient flow steps
    dt::T #flow integration step-size
    flwint::FlowIntr
    #scalar smearing
    n::Int64 #smearing steps
    r::T # smearing weight

    smr{T1}(a,b,c,d,e) where {T1} = new{T1}(a,b,c,d,e)

    function smr{T}(a, b, d, e) where {T}
        return new{T}(a, b, wfl_rk3(T, b, 1.0E-6), d, e)
    end
end
export smr

include("ScalarAction.jl")
export scalar_action

include("ScalarForce.jl")
export force_scalar

include("ScalarFields.jl")
export randomize!

include("ScalarHMC.jl")
export HMC!

include("ScalarObs.jl")
export scalar_obs, scalar_corr, mixed_corr, smearing

end
