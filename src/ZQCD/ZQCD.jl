###
### "THE BEER-WARE LICENSE":
### Alberto Ramos wrote this file. As long as you retain this 
### notice you can do whatever you want with this stuff. If we meet some 
### day, and you think this stuff is worth it, you can buy me a beer in 
### return. <alberto.ramos@cern.ch>
###
### author:  pietro.butti.fl@gmail.com
### file:    ZQCD.jl
### created: Fri  8 Sep 2023 10:53:35 CEST
###                               

module ZQCD

    using CUDA, Random, TimerOutputs
    using ..Space
    using ..Groups
    using ..Fields
    using ..MD
    using ..YM
    import ..YM: HMC!, randomize!, MD!, force_gauge, FlowIntr

    import Base.show

    struct ZQCDParm{T}
        b1::T
        b2::T
        c1::T
        c2::T
        c3::T
    end

    """
        ZQCDParm{T}(r2::T,g2::T) where T

    2-argument (overridden) constructor for `ZQCDParm`. `c1`, `c1`, `c1` are set to their numerical values given by formulas (3.10-3.12) of arXiv:0801.1566v2, while `b1` and `b2` are functions of the gauge coupling β given by formulas (C.9) and (C.10)
    """
    function ZQCDParm{T}(r2::T, g2::T, β::T) where T
        _c1 = 0.0311994 * r2 + 0.0135415 * g2 
        _c2 = 0.0311994 * r2 + 0.008443432 * g2 
        _c3 = 0.0623987 * r2
        _b1 =  -r2/(2. * g2)^2 - 2.38193365/4. /π * (2. *_c1 + _c3 ) * β + 1. / 16. / π^2 * ((48. *_c1^2 + 12. *_c3^2 - 12. *_c3 )*(log(1.5*β) + 0.08849) - 6.9537*_c3)
        _b2 =  -r2/(2. * g2)^2 + 0.441841/g2 - 0.7939779/4. / π * (10. * _c2 + _c3 + 2) * β +  1. / 16. / π^2 * ((80. *_c2^2 + 4. *_c3^2 - 40. *_c3 )*(log(1.5*β) + 0.08849) - 23.17895*_c2 - 8.66687)

        return ZQCDParm{T}(_b1,_b2,_c1,_c2,_c3)
    end

    function Base.show(io::IO, sp::ZQCDParm{T}) where T
        println(io, " ZQCD initialized with: ")
        println(io, "b₁ = $(sp.b1)")
        println(io, "b₂ = $(sp.b2)")
        println(io, "c₁ = $(sp.c1)")
        println(io, "c₂ = $(sp.c2)")
        println(io, "c₃ = $(sp.c3)")

    end
    export ZQCDParm

    struct ZQCDworkspace{T}
        frcSigma
        frcPi
        momSigma
        momPi
        Sigma
        Pi
        ZQCDworkspace(::Type{T}, lp::SpaceParm) where {T<:AbstractFloat} = new{T}(
            scalar_field(T,lp),        # F_Σ : scalar field
            scalar_field(SU2alg{T},lp),# F_Π : SU2 algebra fields
            scalar_field(T,lp),        # mom_Σ : scalar field
            scalar_field(SU2alg{T},lp),# mom_Π :  SU2 algebra field
            scalar_field(T,lp),        # Σ : a real field
            scalar_field(SU2alg{T},lp) # Π : (3 scalar field ∼) SU2 algebra matrix (Π = i Πₐ/2⋅σₐ)
        )
    end
    export ZQCDworkspace



    include("ZQCDAction.jl")
        export zqcd_action

    include("ZQCDForce.jl")
        export zqcd_force

    include("ZQCDFields.jl")
        export randomize!

    include("ZQCDHMC.jl")
        export hamiltonian



end
