###
### "THE BEER-WARE LICENSE":
### Alberto Ramos wrote this file. As long as you retain this 
### notice you can do whatever you want with this stuff. If we meet some 
### day, and you think this stuff is worth it, you can buy me a beer in 
### return. <alberto.ramos@cern.ch>
###
### file:    AlgebraSU2.jl
### created: Sun Oct  3 09:24:25 2021
###                               

SU2alg(x::T)                       where T <: AbstractFloat = SU2alg{T}(x,0.0,0.0)
SU2alg(v::Vector{T})               where T <: AbstractFloat = SU2alg{T}(v[1],v[2],v[3])
projalg(g::SU2{T})                 where T <: AbstractFloat = SU2alg{T}(imag(g.t2), real(g.t2), imag(g.t1))
projalg(z::Complex{T}, g::SU2{T})  where T <: AbstractFloat = SU2alg{T}(imag(z*g.t2), real(z*g.t2), imag(z*g.t1))
dot(a::SU2alg{T}, b::SU2alg{T})    where T <: AbstractFloat = a.t1*b.t1 + a.t2*b.t2 + a.t3*b.t3
norm(a::SU2alg{T})             where T <: AbstractFloat = sqrt(a.t1^2 + a.t2^2 + a.t3^2)
norm2(a::SU2alg{T})            where T <: AbstractFloat = a.t1^2 + a.t2^2 + a.t3^2

Base.:+(a::SU2alg{T})              where T <: AbstractFloat = SU2alg{T}(a.t1,a.t2,a.t3)
Base.:-(a::SU2alg{T})              where T <: AbstractFloat = SU2alg{T}(-a.t1,-a.t2,-a.t3)
Base.:+(a::SU2alg{T},b::SU2alg{T}) where T <: AbstractFloat = SU2alg{T}(a.t1+b.t1,a.t2+b.t2,a.t3+b.t3)
Base.:-(a::SU2alg{T},b::SU2alg{T}) where T <: AbstractFloat = SU2alg{T}(a.t1-b.t1,a.t2-b.t2,a.t3-b.t3)

Base.:*(a::SU2alg{T},b::Number)    where T <: AbstractFloat = SU2alg{T}(a.t1*b,a.t2*b,a.t3*b)
Base.:*(b::Number,a::SU2alg{T})    where T <: AbstractFloat = SU2alg{T}(a.t1*b,a.t2*b,a.t3*b)
Base.:/(a::SU2alg{T},b::Number)    where T <: AbstractFloat = SU2alg{T}(a.t1/b,a.t2/b,a.t3/b)

function alg2mat(a::SU2alg{T}) where T <: AbstractFloat

    u11::Complex{T} = complex(0.0, a.t3)/2
    u22::Complex{T} = conj(u11)
    u12::Complex{T} = complex(a.t2,a.t1)/2
    u21::Complex{T} = -conj(u12)
    
    return M2x2{T}(u11,u12,u21,u22)
end

Base.:*(a::SU2alg,b::SU2) = alg2mat(a)*b
Base.:*(a::SU2,b::SU2alg) = a*alg2mat(b)
Base.:/(a::SU2alg,b::SU2) = alg2mat(a)/b
Base.:\(a::SU2,b::SU2alg) = a\alg2mat(b)

## ================ PIETRO ===============
Base.:*(a::SU2alg,b::M2x2) = alg2mat(a)*b
Base.:*(a::M2x2,b::SU2alg) = a*alg2mat(b)
## ========================================


"""
    function Base.exp(a::T, t::Number=1) where {T <: Algebra}

Computes `exp(a)`
"""
function Base.exp(a::SU2alg{T}) where T <: AbstractFloat
    
    rm = sqrt( a.t1^2+a.t2^2+a.t3^2 )/2.0
    if (abs(rm) < 0.05)
        rms = rm^2/2.0
        ca = 1.0 - rms    *(1.0 - (rms/6.0 )*(1.0 - rms/15.0))
        sa = 0.5 - rms/6.0*(1.0 - (rms/10.0)*(1.0 - rms/21.0))
    else
        ca = CUDA.cos(rm)
	sa = CUDA.sin(rm)/(2.0*rm)
    end

    t1 = complex(ca,sa*a.t3)
    t2 = complex(sa*a.t2,sa*a.t1)
    return SU2{T}(t1,t2)
end

function Base.exp(a::SU2alg{T}, t::T) where T <: AbstractFloat
    
    rm = t*sqrt( a.t1^2+a.t2^2+a.t3^2 )/2.0
    if (abs(rm) < 0.05)
        rms = rm^2/2.0
        ca = 1.0 - rms    *(1.0 - (rms/6.0 )*(1.0 - rms/15.0))
        sa = t*(0.5 - rms/6.0*(1.0 - (rms/10.0)*(1.0 - rms/21.0)))
    else
        ca = CUDA.cos(rm)
	sa = t*CUDA.sin(rm)/(2.0*rm)
    end

    t1 = complex(ca,sa*a.t3)
    t2 = complex(sa*a.t2,sa*a.t1)
    return SU2{T}(t1,t2)
end


"""
    function expm(g::G, a::A) where {G <: Algebra, A <: Algebra}

Computes `exp(a)*g`

"""
function expm(g::SU2{T}, a::SU2alg{T}) where T <: AbstractFloat
    
    rm = sqrt( a.t1^2+a.t2^2+a.t3^2 )/2.0
    if (abs(rm) < 0.05)
        rms = rm^2/2.0
        ca = 1.0 - rms    *(1.0 - (rms/6.0 )*(1.0 - rms/15.0))
        sa = 0.5 - rms/6.0*(1.0 - (rms/10.0)*(1.0 - rms/21.0))
    else
        ca = CUDA.cos(rm)
	sa = CUDA.sin(rm)/(2.0*rm)
    end

    t1 = complex(ca,sa*a.t3)*g.t1-complex(sa*a.t2,sa*a.t1)*conj(g.t2)
    t2 = complex(ca,sa*a.t3)*g.t2+complex(sa*a.t2,sa*a.t1)*conj(g.t1)
    return SU2{T}(t1,t2)
end

"""
    function expm(g::SU2, a::SU2alg, t::Float64)

Computes `exp(t*a)*g`

"""
function expm(g::SU2{T}, a::SU2alg{T}, t::T) where T <: AbstractFloat
    
    rm = t*sqrt( a.t1^2+a.t2^2+a.t3^2 )/2.0
    if (abs(rm) < 0.05)
        rms = rm^2/2.0
        ca = 1.0 - rms    *(1.0 - (rms/6.0 )*(1.0 - rms/15.0))
        sa = t*(0.5 - rms/6.0*(1.0 - (rms/10.0)*(1.0 - rms/21.0)))
    else
        ca = CUDA.cos(rm)
	sa = t*CUDA.sin(rm)/(2.0*rm)
    end

    t1 = complex(ca,sa*a.t3)*g.t1-complex(sa*a.t2,sa*a.t1)*conj(g.t2)
    t2 = complex(ca,sa*a.t3)*g.t2+complex(sa*a.t2,sa*a.t1)*conj(g.t1)
    return SU2{T}(t1,t2)
               
end

"""
    function adjaction(g::SU2, a::SU2alg)

Computes the adjoint action of g on the algebra.
For matrix representations it is: `g^{-1}*a*g`

"""
function adjaction(g::SU2{T}, a::SU2alg{T}) where T <: AbstractFloat

    x = g.t1
    y = g.t2
    c11 = (x^2 + conj(x)^2 - y^2 - conj(y)^2 )/2
    c12 = -1im*(x^2 - conj(x)^2 + y^2 - conj(y)^2 )/2
    c13 = -x*y - conj(x)*conj(y)

    c21 = 1im*(x^2 - conj(x)^2 - y^2 + conj(y)^2 )/2
    c22 = (x^2 + conj(x)^2 + y^2 + conj(y)^2 )/2
    c23 = -1im*(x*y - conj(x)*conj(y))

    c31 = conj(x)*y + x*conj(y)
    c32 = 1im*(conj(x)*y - x*conj(y))
    c33 = abs(x)^2 - abs(y)^2

    f1 = real(a.t1 * c11 + a.t2 * c21 + a.t3 * c31)
    f2 = real(a.t1 * c12 + a.t2 * c22 + a.t3 * c32)
    f3 = real(a.t1 * c13 + a.t2 * c23 + a.t3 * c33)

    return SU2alg{T}(f1,f2,f3)

end



# #multiplication by pauli matrices
# algXpauli(g::SU2alg{T}, ::Type{Pauli{1}}) where T <: AbstractFloat = M2x2{T}(complex(-g.t2,g.t2),complex(-g.t3),complex(),complex())
# algXpauli(g::SU2alg{T}, ::Type{Pauli{2}}) where T <: AbstractFloat = SU2{T}()
# algXpauli(g::SU2alg{T}, ::Type{Pauli{3}}) where T <: AbstractFloat = M2x2{T}(complex(0,1)*g.t1, -complex(0,1)*g.t2, -complex(0,1)*conj(g.t2), -complex(0,1)*conj(g.t1))
