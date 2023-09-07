###
### "THE BEER-WARE LICENSE":
### Alberto Ramos wrote this file. As long as you retain this 
### notice you can do whatever you want with this stuff. If we meet some 
### day, and you think this stuff is worth it, you can buy me a beer in 
### return. <alberto.ramos@cern.ch>
###
### file:    SU3Types.jl
### created: Sun Oct  3 09:05:23 2021
###                               

#
# Use memory efficient representation: Only store
# first two rows. Third row constructed on the fly.
#
# a.u31 = conj(a.u12*a.u23 - a.u13*a.u22)
# a.u32 = conj(a.u13*a.u21 - a.u11*a.u23)
# a.u33 = conj(a.u11*a.u22 - a.u12*a.u21)
#

struct SU3{T} <: Group
    u11::Complex{T}
    u12::Complex{T}
    u13::Complex{T}
    u21::Complex{T}
    u22::Complex{T}
    u23::Complex{T}
end
Base.one(::Type{SU3{T}}) where T <: AbstractFloat = SU3{T}(one(T),zero(T),zero(T),zero(T),one(T),zero(T))

struct M3x3{T} 
    u11::Complex{T}
    u12::Complex{T}
    u13::Complex{T}
    u21::Complex{T}
    u22::Complex{T}
    u23::Complex{T}
    u31::Complex{T}
    u32::Complex{T}
    u33::Complex{T}
end
Base.one(::Type{M3x3{T}}) where T <: AbstractFloat = M3x3{T}(one(T),zero(T),zero(T),zero(T),one(T),zero(T),zero(T),zero(T),one(T))
Base.zero(::Type{M3x3{T}}) where T <: AbstractFloat = M3x3{T}(zero(T),zero(T),zero(T),zero(T),zero(T),zero(T),zero(T),zero(T),zero(T))

struct SU3alg{T} <: Algebra
    t1::T
    t2::T
    t3::T
    t4::T
    t5::T
    t6::T
    t7::T
    t8::T
end


Random.rand(rng::AbstractRNG, ::Random.SamplerType{SU3{T}}) where T <: AbstractFloat = exp(SU3alg{T}(randn(rng,T),randn(rng,T),randn(rng,T),randn(rng,T),randn(rng,T),randn(rng,T),randn(rng,T),randn(rng,T)))
Random.rand(rng::AbstractRNG, ::Random.SamplerType{SU3alg{T}}) where T <: AbstractFloat = SU3alg{T}(randn(rng,T),randn(rng,T),randn(rng,T),randn(rng,T),randn(rng,T),randn(rng,T),randn(rng,T),randn(rng,T))


Base.convert(::Type{M3x3{T}}, a::SU3alg{T}) where T = alg2mat(a)
Base.convert(::Type{M3x3{T}}, a::SU3{T}) where T = M3x3{T}(a.u11,a.u12,a.u13,
                                                        a.u21,a.u22,a.u23,
                                                        conj(a.u12*a.u23 - a.u13*a.u22),
                                                        conj(a.u13*a.u21 - a.u11*a.u23),
                                                        conj(a.u11*a.u22 - a.u12*a.u21))

struct SU3fund{T}
    t1::Complex{T}
    t2::Complex{T}
    t3::Complex{T}
end
Base.zero(::Type{SU3fund{T}}) where T <: AbstractFloat = SU3fund{T}(zero(T),zero(T),zero(T))
Random.rand(rng::AbstractRNG, ::Random.SamplerType{SU3fund{T}}) where T <: AbstractFloat = SU3fund{T}(complex(randn(rng,T),randn(rng,T)),
                                                                                                      complex(randn(rng,T),randn(rng,T)),
                                                                                                      complex(randn(rng,T),randn(rng,T)))
