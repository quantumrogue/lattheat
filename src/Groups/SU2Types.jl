###
### "THE BEER-WARE LICENSE":
### Alberto Ramos wrote this file. As long as you retain this 
### notice you can do whatever you want with this stuff. If we meet some 
### day, and you think this stuff is worth it, you can buy me a beer in 
### return. <alberto.ramos@cern.ch>
###
### file:    SU2Types.jl
### created: Sun Oct  3 09:22:48 2021
###                               

struct SU2{T} <: Group
    t1::Complex{T}
    t2::Complex{T}
end

struct M2x2{T}
    u11::Complex{T}
    u12::Complex{T}
    u21::Complex{T}
    u22::Complex{T}
end

struct SU2alg{T} <: Algebra
    t1::T
    t2::T
    t3::T
end

struct SU2fund{T}
    t1::Complex{T}
    t2::Complex{T}
end    

Base.zero(::Type{SU2fund{T}}) where T <: AbstractFloat = SU2fund{T}(zero(T),zero(T))
Base.zero(::Type{SU2alg{T}})  where T <: AbstractFloat = SU2alg{T}(zero(T),zero(T),zero(T))
Base.zero(::Type{M2x2{T}})    where T <: AbstractFloat = M2x2{T}(zero(T),zero(T),zero(T),zero(T))
Base.one(::Type{SU2{T}})      where T <: AbstractFloat = SU2{T}(one(T),zero(T))
Base.one(::Type{M2x2{T}})     where T <: AbstractFloat = M2x2{T}(one(T),zero(T),zero(T),one(T))
Base.one(::Type{SU2fund{T}})  where T <: AbstractFloat = SU2fund{T}(2*one(T),zero(T))

Base.convert(::Type{M2x2{T}}, a::SU2alg{T}) where T = alg2mat(a)
Base.convert(::Type{M2x2{T}}, a::SU2{T}) where T = M2x2{T}(a.t1,a.t2,-conj(a.t2), conj(a.t1))

Random.rand(rng::AbstractRNG, ::Random.SamplerType{SU2fund{T}}) where T <: AbstractFloat = SU2fund{T}(complex(randn(rng,T),randn(rng,T)),complex(randn(rng,T),randn(rng,T)))
Random.rand(rng::AbstractRNG, ::Random.SamplerType{SU2alg{T}}) where T <: AbstractFloat = SU2alg{T}(randn(rng,T),randn(rng,T),randn(rng,T))
Random.rand(rng::AbstractRNG, ::Random.SamplerType{SU2{T}})    where T <: AbstractFloat = exp(SU2alg{T}(randn(rng,T),randn(rng,T),randn(rng,T)))
