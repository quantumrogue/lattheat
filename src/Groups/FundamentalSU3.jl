###
### "THE BEER-WARE LICENSE":
### Alberto Ramos wrote this file. As long as you retain this 
### notice you can do whatever you want with this stuff. If we meet some 
### day, and you think this stuff is worth it, you can buy me a beer in 
### return. <alberto.ramos@cern.ch>
###
### file:    FundamentalSU3.jl
### created: Tue Nov 16 15:07:00 2021
###                               


SU3fund(a::T, b::T, c::T)          where T <: AbstractFloat = SU3fund{T}(complex(a), complex(b), complex(c))

"""
    norm(a::SU3fund{T})

Returns the conjugate of a fundamental element. 
"""
dag(a::SU3fund{T})                 where T <: AbstractFloat = SU3fund{T}(conj(a.t1), conj(a.t2), conj(a.t3))

"""
    norm2(a::SU3fund{T})

Returns the norm squared of a fundamental element. Same result as dot(a,a).
"""
norm(a::SU3fund{T})                where T <: AbstractFloat = sqrt((abs2(a.t1) + abs2(a.t2) + abs2(a.t1)))

"""
    norm(a::SU3fund{T})

Returns the norm of a fundamental element. Same result as sqrt(dot(a,a)).
"""
norm2(a::SU3fund{T})               where T <: AbstractFloat = (abs2(a.t1) + abs2(a.t2) + abs2(a.t1))

"""
    dot(a::SU3fund{T},b::SU3fund{T})

Returns the scalar product of two fundamental elements. The convention is for the product to the linear in the second argument, and anti-linear in the first argument. 
"""
dot(g1::SU3fund{T},g2::SU3fund{T}) where T <: AbstractFloat = conj(g1.t1)*g2.t1+g1.t2*conj(g2.t2)+g1.t3*conj(g2.t3)

"""
    *(g::SU3{T},b::SU3fund{T})

Returns ga
"""
Base.:*(g::SU3{T},b::SU3fund{T})     where T <: AbstractFloat = SU3fund{T}(g.u11*b.t1 + g.u12*b.t2 + g.u13*b.t3,
                                                                           g.u21*b.t1 + g.u22*b.t2 + g.u23*b.t3,
                                                                           conj(g.u12*g.u23 - g.u13*g.u22)*b.t1 +
                                                                               conj(g.u13*g.u21 - g.u11*g.u23)*b.t2 +
                                                                               conj(g.u11*g.u22 - g.u12*g.u21)*b.t3)

"""
    \\(g::SU3{T},b::SU3fund{T})

Returns g^dag b
"""
Base.:\(g::SU3{T},b::SU3fund{T})     where T <: AbstractFloat = SU3fund{T}(conj(g.u11)*b.t1 + conj(g.u21)*b.t2 + (g.u12*g.u23 - g.u13*g.u22)*b.t3,
                                                                           conj(g.u12)*b.t1 + conj(g.u22)*b.t2 + (g.u13*g.u21 - g.u11*g.u23)*b.t3,
                                                                           conj(g.u13)*b.t1 + conj(g.u23)*b.t2 + (g.u11*g.u22 - g.u12*g.u21)*b.t3)

Base.:+(a::SU3fund{T},b::SU3fund{T}) where T <: AbstractFloat = SU3fund{T}(a.t1+b.t1,a.t2+b.t2,a.t3+b.t3)
Base.:-(a::SU3fund{T},b::SU3fund{T}) where T <: AbstractFloat = SU3fund{T}(a.t1-b.t1,a.t2-b.t2,a.t3-b.t3)
Base.:+(a::SU3fund{T})               where T <: AbstractFloat = SU3fund{T}(a.t1,a.t2,a.t3)
Base.:-(a::SU3fund{T})               where T <: AbstractFloat = SU3fund{T}(-a.t1,-a.t2,-a.t3)
imm(a::SU3fund{T})                   where T <: AbstractFloat = SU3fund{T}(complex(-imag(a.t1),real(a.t1)),
                                                                           complex(-imag(a.t2),real(a.t2)),
                                                                           complex(-imag(a.t3),real(a.t3)))
mimm(a::SU3fund{T})                  where T <: AbstractFloat = SU3fund{T}(complex(imag(a.t1),-real(a.t1)),
                                                                           complex(imag(a.t2),-real(a.t2)),
                                                                           complex(imag(a.t3),-real(a.t3)))

# Operations with numbers
Base.:*(a::SU3fund{T},b::Number) where T <: AbstractFloat = SU3fund{T}(b*a.t1,b*a.t2,b*a.t3)
Base.:*(b::Number,a::SU3fund{T}) where T <: AbstractFloat = SU3fund{T}(b*a.t1,b*a.t2,b*a.t3)
Base.:/(a::SU3fund{T},b::Number) where T <: AbstractFloat = SU3fund{T}(a.t1/b,a.t2/b,a.t3/b)

