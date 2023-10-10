module HotSU2

using CUDA, Random, TimerOutputs
using ..Space 
using ..Groups
using ..Fields 
using ..MD
using ..YM
import ..YM: HMC!, randomize!, MD!, force_gauge, FlowIntr

import Base.show

struct HotSU2Param{T}
    c1::T
    c2::T
    c3::T
    b1::T
    b2::T
end
function Base.show(io::IO, hp::HotSU2Param{T}) where {T}

end

function HotSU2Param(r::T, g2::T, beta::T) where {T}
    c1 = 0.0311994*r^2 + 0.0135415*g2
    c2 = 0.0311994*r^2 + 0.008443432*g2
    c3 = 0.0623987*r^2
    b1 =  -0.25*r^2/g2^2 - (2.38193365/(4*pi))*(2*c1+c3)*beta +
            (1/(16*pi^2)) * ((48*c1^2 + 12*c3^2 - 12*c3)*(log(1.5*beta) + 0.08849) - 6.9537*c3)
    b2 = -0.25*r^2/g2^2 + 0.441841/g2 - (0.7939779/(4*pi))*(10*c2+c3+2)*beta +
            (1/(16*pi^2)) * ((80*c2^2+4*c3^2-40*c2)*(log(1.5*beta)+0.08849) - 23.17895*c2 -8.66687)
    return HotSU2Param{T}(c1, c2, c3, b1, b2)
end

export HotSU2Param

struct HotSU2Workspace{T}
    Sigma
    Pi
    frcSigma
    frcPi
    momSigma
    momPi
    function HotSU2Workspace(::Type{T}, lp::SpaceParm) where {T <: AbstractFloat}
        
        return new{T}(scalar_field(T, lp),          # -> Σ scalar field
                      scalar_field(SU2alg{T}, lp),  # -> Π SU(2) matrix 
                      scalar_field(T, lp),          # -> Σ forces
                      scalar_field(SU2alg{T}, lp),  # -> Π forces
                      scalar_field(T, lp),          # -> Σ mom
                      scalar_field(SU2alg{T}, lp))  # -> Π mom    
    end
end
export HotSU2Workspace

include("HotSU2Action.jl")
export hotSU2_action

end