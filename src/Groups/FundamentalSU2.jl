###
### "THE BEER-WARE LICENSE":
### Alberto Ramos wrote this file. As long as you retain this 
### notice you can do whatever you want with this stuff. If we meet some 
### day, and you think this stuff is worth it, you can buy me a beer in 
### return. <alberto.ramos@cern.ch>
###
### file:    FundamentalSU2.jl
### created: Tue Oct  5 10:31:09 2021
###                               

SU2fund(a::T, b::T)                where T <: AbstractFloat = SU2fund{T}(complex(a), complex(b))
dag(a::SU2fund{T})                 where T <: AbstractFloat = SU2fund{T}(conj(a.t1), -a.t2)
norm(a::SU2fund{T})                where T <: AbstractFloat = sqrt((abs2(a.t1) + abs2(a.t2))/2)
norm2(a::SU2fund{T})               where T <: AbstractFloat = (abs2(a.t1) + abs2(a.t2))/2
tr(g::SU2fund{T})                  where T <: AbstractFloat = complex(real(g.t1), 0.0)
dot(g1::SU2fund{T},g2::SU2fund{T}) where T <: AbstractFloat = real(conj(g1.t1)*g2.t1+g1.t2*conj(g2.t2))/2
projalg(g::SU2fund{T})             where T <: AbstractFloat = SU2alg{T}(imag(g.t2)/2, real(g.t2)/2, imag(g.t1)/2)

Base.:*(a::SU2fund{T},b::SU2fund{T}) where T <: AbstractFloat = SU2fund{T}((a.t1*b.t1-a.t2*conj(b.t2))/2,(a.t1*b.t2+a.t2*conj(b.t1))/2)
Base.:*(a::SU2fund{T},b::SU2{T})     where T <: AbstractFloat = SU2fund{T}( a.t1*b.t1-a.t2*conj(b.t2)   , a.t1*b.t2+a.t2*conj(b.t1))
Base.:*(a::SU2{T},b::SU2fund{T})     where T <: AbstractFloat = SU2fund{T}( a.t1*b.t1-a.t2*conj(b.t2)   , a.t1*b.t2+a.t2*conj(b.t1))
Base.:/(a::SU2fund{T},b::SU2{T})     where T <: AbstractFloat = SU2fund{T}(a.t1*conj(b.t1)+a.t2*conj(b.t2),-a.t1*b.t2+a.t2*b.t1)
Base.:/(a::SU2{T},b::SU2fund{T})     where T <: AbstractFloat = SU2fund{T}(a.t1*conj(b.t1)+a.t2*conj(b.t2),-a.t1*b.t2+a.t2*b.t1)
Base.:\(a::SU2{T},b::SU2fund{T})     where T <: AbstractFloat = SU2fund{T}(conj(a.t1)*b.t1+a.t2*conj(b.t2),conj(a.t1)*b.t2-a.t2*conj(b.t1))
Base.:\(a::SU2fund{T},b::SU2{T})     where T <: AbstractFloat = SU2fund{T}(conj(a.t1)*b.t1+a.t2*conj(b.t2),conj(a.t1)*b.t2-a.t2*conj(b.t1))

Base.:+(a::SU2fund{T},b::SU2fund{T}) where T <: AbstractFloat = SU2fund{T}(a.t1+b.t1,a.t2+b.t2)
Base.:-(a::SU2fund{T},b::SU2fund{T}) where T <: AbstractFloat = SU2fund{T}(a.t1-b.t1,a.t2-b.t2)

# Operations with numbers
Base.:*(a::SU2fund{T},b::Number) where T <: AbstractFloat = SU2fund{T}(b*a.t1,b*a.t2)
Base.:*(b::Number,a::SU2fund{T}) where T <: AbstractFloat = SU2fund{T}(b*a.t1,b*a.t2)
Base.:/(a::SU2fund{T},b::Number) where T <: AbstractFloat = SU2fund{T}(a.t1/b,a.t2/b)


#Multiplication by i*(Pauli Matrix)
fundipau(a::SU2fund{T}, ::Type{Pauli{1}}) where T <: AbstractFloat = SU2fund{T}(complex(0.0,1.0)*a.t2, complex(0.0,1.0)*a.t1)
fundipau(a::SU2fund{T}, ::Type{Pauli{2}}) where T <: AbstractFloat = SU2fund{T}(-a.t2, a.t1)
fundipau(a::SU2fund{T}, ::Type{Pauli{3}}) where T <: AbstractFloat = SU2fund{T}(complex(0.0,1.0)*a.t1, complex(0.0,-1.0)*a.t2)

#Tr(g*i*PauliMatrix)
tr_ipau(g::SU2fund{T}, ::Type{Pauli{1}}) where T <: AbstractFloat = complex(-imag(g.t2), 0.0)
tr_ipau(g::SU2fund{T}, ::Type{Pauli{2}}) where T <: AbstractFloat = complex(-real(g.t2), 0.0)
tr_ipau(g::SU2fund{T}, ::Type{Pauli{3}}) where T <: AbstractFloat = complex(-imag(g.t1), 0.0)
