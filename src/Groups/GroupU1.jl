###
### "THE BEER-WARE LICENSE":
### Alberto Ramos wrote this file. As long as you retain this 
### notice you can do whatever you want with this stuff. If we meet some 
### day, and you think this stuff is worth it, you can buy me a beer in 
### return. <alberto.ramos@cern.ch>
###
### file:    GroupU1.jl
### created: Tue Sep 21 09:33:44 2021
###                               

using CUDA, Random

import Base.:*, Base.:+, Base.:-,Base.:/,Base.:\,Base.exp,Base.zero,Base.one
import Random.rand
struct U1{T} <: Group
    t1::T
    t2::T
end
U1(a::T)          where T <: AbstractFloat = U1{T}(a,zero(T))
inverse(b::U1{T}) where T <: AbstractFloat = U1{T}(b.t1,-b.t2)
dag(a::U1{T})    where T <: AbstractFloat = inverse(a)
norm(a::U1{T})    where T <: AbstractFloat = sqrt(a.t1^2+a.t2^2)
norm2(a::U1{T})   where T <: AbstractFloat = a.t1^2+a.t2^2
tr(g::U1{T})      where T <: AbstractFloat = complex(a.t1)
Base.one(::Type{U1{T}}) where T <: AbstractFloat = U1{T}(one(T), zero(T))
function Random.rand(rng::AbstractRNG, ::Random.SamplerType{U1{T}}) where T <: AbstractFloat
    r = randn(rng,T)
    return U1{T}(CUDA.cos(r),CUDA.sin(r))
end

"""
    function normalize(a::U1)

Return a normalized element of `SU(2)`
"""
function normalize(a::U1{T}) where T <: AbstractFloat
    dr = norm(a)
    return U1{T}(a.t1/dr, a.t2/dr)
end

Base.:*(a::U1{T},b::U1{T}) where T <: AbstractFloat = U1{T}(a.t1*b.t1-a.t2*b.t2, a.t1*b.t2+a.t2*b.t1)
Base.:/(a::U1{T},b::U1{T}) where T <: AbstractFloat = U1{T}(a.t1*b.t1+a.t2*b.t2, -a.t1*b.t2+a.t2*b.t1)
Base.:\(a::U1{T},b::U1{T}) where T <: AbstractFloat = U1{T}(a.t1*b.t1+a.t2*b.t2, a.t1*b.t2-a.t2*b.t1)

struct U1alg{T} <: Algebra
    t::T
end
projalg(g::U1{T})             where T <: AbstractFloat = U1alg{T}(g.t2)
dot(a::U1alg{T}, b::U1alg{T}) where T <: AbstractFloat = a.t*b.t
norm(a::U1alg{T})             where T <: AbstractFloat = abs(a.t)
norm2(a::U1alg{T})            where T <: AbstractFloat = a.t^2
Base.zero(::Type{U1alg{T}})   where T <: AbstractFloat = U1alg{T}(zero(T))
Random.rand(rng::AbstractRNG, ::Random.SamplerType{U1alg{T}}) where T <: AbstractFloat = U1alg{T}(randn(rng,T))

Base.:+(a::U1alg{T})             where T <: AbstractFloat = U1alg{T}(a.t)
Base.:-(a::U1alg{T})             where T <: AbstractFloat = U1alg{T}(-a.t)
Base.:+(a::U1alg{T},b::U1alg{T}) where T <: AbstractFloat = U1alg{T}(a.t+b.t)
Base.:-(a::U1alg{T},b::U1alg{T}) where T <: AbstractFloat = U1alg{T}(a.t-b.t)

Base.:*(a::U1alg{T},b::Number)   where T <: AbstractFloat = U1alg{T}(a.t*b)
Base.:*(b::Number,a::U1alg{T})   where T <: AbstractFloat = U1alg{T}(a.t*b)
Base.:/(a::U1alg{T},b::Number)   where T <: AbstractFloat = U1alg{T}(a.t/b)

isgroup(a::U1{T}) where T <: AbstractFloat = (abs(a.t) -1.0) < 1.0E-10

"""
    function Base.exp(a::U1alg, t::Number=1)

Computes `exp(a)`
"""
Base.exp(a::U1alg{T}) where T <: AbstractFloat = U1{T}(CUDA.cos(a.t), CUDA.sin(a.t))
Base.exp(a::U1alg{T}, t::T) where T <: AbstractFloat  = U1{T}(CUDA.cos(t*a.t), CUDA.sin(t*a.t))

"""
    function expm(g::U1, a::U1alg; t=1)

Computes `exp(a)*g`

"""
expm(g::U1{T}, a::U1alg{T}) where T <: AbstractFloat = U1{T}(CUDA.cos(a.t), CUDA.sin(a.t))*g
expm(g::U1{T}, a::U1alg{T}, t::T) where T <: AbstractFloat = U1{T}(CUDA.cos(t*a.t), CUDA.sin(t*a.t))*g

export U1, U1alg, inverse, dag, tr, projalg, expm, exp, norm, norm2, isgroup
