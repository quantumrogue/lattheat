###
### "THE BEER-WARE LICENSE":
### Alberto Ramos wrote this file. As long as you retain this 
### notice you can do whatever you want with this stuff. If we meet some 
### day, and you think this stuff is worth it, you can buy me a beer in 
### return. <alberto.ramos@cern.ch>
###
### file:    GroupSU2.jl
### created: Sun Jul 11 17:23:12 2021
###                               

#
# SU(2) group elements represented trough Cayley-Dickson
#       construction
# https://en.wikipedia.org/wiki/Cayley%E2%80%93Dickson_construction
using CUDA, Random

SU2(a::T, b::T)          where T <: AbstractFloat = SU2{T}(complex(a), complex(b))
inverse(b::SU2{T})       where T <: AbstractFloat = SU2{T}(conj(b.t1), -b.t2)
dag(a::SU2{T})           where T <: AbstractFloat = inverse(a)
norm(a::SU2{T})          where T <: AbstractFloat = sqrt(abs2(a.t1) + abs2(a.t2))
norm2(a::SU2{T})         where T <: AbstractFloat = abs2(a.t1) + abs2(a.t2)
tr(g::SU2{T})            where T <: AbstractFloat = complex(2.0*real(g.t1), 0.0)
dev_one(g::SU2{T})       where T <: AbstractFloat = sqrt(( abs2(g.t1 - one(T)) + abs2(g.t2))/2)


# =============== PIETRO ================
#Tr(g*i*PauliMatrix)
tr_ipau(g::SU2{T}, ::Type{Pauli{1}}) where T <: AbstractFloat = complex(-2. * imag(g.t2), 0.0)
tr_ipau(g::SU2{T}, ::Type{Pauli{2}}) where T <: AbstractFloat = complex(-2. * real(g.t2), 0.0)
tr_ipau(g::SU2{T}, ::Type{Pauli{3}}) where T <: AbstractFloat = complex(-2. * imag(g.t1), 0.0)
# +++++++++++++++++++++++++++++++++++++++


"""
    function unitarize(a::T) where {T <: Group}

Return a unitarized element of the group.
"""
function unitarize(a::SU2{T}) where T <: AbstractFloat
    dr = sqrt(abs2(a.t1) + abs2(a.t2))
    if (dr == 0.0)
        return SU2{T}(0.0,0.0)
    end
    return SU2{T}(a.t1/dr,a.t2/dr)
end

Base.:*(a::SU2{T},b::SU2{T}) where T <: AbstractFloat = SU2{T}(a.t1*b.t1-a.t2*conj(b.t2),a.t1*b.t2+a.t2*conj(b.t1))
Base.:/(a::SU2{T},b::SU2{T}) where T <: AbstractFloat = SU2{T}(a.t1*conj(b.t1)+a.t2*conj(b.t2),-a.t1*b.t2+a.t2*b.t1)
Base.:\(a::SU2{T},b::SU2{T}) where T <: AbstractFloat = SU2{T}(conj(a.t1)*b.t1+a.t2*conj(b.t2),conj(a.t1)*b.t2-a.t2*conj(b.t1))

function isgroup(a::SU2{T}) where T <: AbstractFloat
    tol = 1.0E-10
    if (abs2(a.t1) + abs2(a.t2) - 1.0 < 1.0E-10)
        return true
    else
        return false
    end
end

