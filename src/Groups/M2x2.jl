###
### "THE BEER-WARE LICENSE":
### Alberto Ramos wrote this file. As long as you retain this 
### notice you can do whatever you want with this stuff. If we meet some 
### day, and you think this stuff is worth it, you can buy me a beer in 
### return. <alberto.ramos@cern.ch>
###
### file:    M3x3.jl
### created: Sun Oct  3 09:03:34 2021
###                               


tr(a::M2x2{T})            where T <: AbstractFloat = a.u11+a.u22

Base.:*(a::M2x2{T},b::M2x2{T}) where T <: AbstractFloat = M2x2{T}(a.u11*b.u11 + a.u12*b.u21,
                                                                  a.u11*b.u12 + a.u12*b.u22, 
                                                                  a.u21*b.u11 + a.u22*b.u21, 
                                                                  a.u21*b.u12 + a.u22*b.u22)

Base.:*(a::SU2{T},b::M2x2{T}) where T <: AbstractFloat = M2x2{T}(a.t1*b.u11+a.t2*b.u21,
                                                                 a.t1*b.u12+a.t2*b.u22,
                                                                 -conj(a.t2)*b.u11+conj(a.t1)*b.u21,
                                                                 -conj(a.t2)*b.u12+conj(a.t1)*b.u22)
    
Base.:*(a::M2x2{T},b::SU2{T}) where T <: AbstractFloat = M2x2{T}(a.u11*b.t1-a.u12*conj(b.t2),
                                                                 a.u11*b.t2+a.u12*conj(b.t1),
                                                                 a.u21*b.t1-a.u22*conj(b.t2),
                                                                 a.u21*b.t2+a.u22*conj(b.t1))

Base.:/(a::M2x2{T},b::SU2{T}) where T <: AbstractFloat = M2x2{T}(a.u11*conj(b.t1)-a.u12*conj(b.t2),
                                                                 a.u11*b.t2+a.u12*b.t1,
                                                                 a.u21*conj(b.t1)-a.u22*conj(b.t2),
                                                                 -a.u21*b.t2+a.u22*b.t1)

Base.:\(a::SU2{T},b::M2x2{T}) where T <: AbstractFloat = M2x2{T}(conj(a.t1)*b.u11+a.t2*b.u21,
                                                                 conj(a.t1)*b.u12+a.t2*b.u22,
                                                                 -conj(a.t2)*b.u11+a.t1*b.u21,
                                                                 -conj(a.t2)*b.u12+a.t1*b.u22)

Base.:*(a::Number,b::M2x2{T}) where T <: AbstractFloat  = M2x2{T}(a*b.u11, a*b.u12,
                                                                  a*b.u21, a*b.u22)

Base.:*(b::M2x2{T},a::Number) where T <: AbstractFloat  = M2x2{T}(a*b.u11, a*b.u12,
                                                                  a*b.u21, a*b.u22)

Base.:+(a::M2x2{T},b::M2x2{T}) where T <: AbstractFloat = M2x2{T}(a.u11+b.u11, a.u12+b.u12,
                                                                  a.u21+b.u21, a.u22+b.u22)

Base.:-(a::M2x2{T},b::M2x2{T}) where T <: AbstractFloat = M2x2{T}(a.u11-b.u11, a.u12-b.u12,
                                                                  a.u21-b.u21, a.u22-b.u22)

Base.:-(b::M2x2{T}) where T <: AbstractFloat            = M2x2{T}(-b.u11, -b.u12,
                                                                  -b.u21, -b.u22)

Base.:+(b::M2x2{T}) where T <: AbstractFloat            = M2x2{T}(b.u11, b.u12,
                                                                  b.u21, b.u22)

function projalg(a::M2x2{T}) where T <: AbstractFloat

    m12 = (a.u12 - conj(a.u21))/2

    return SU2alg{T}(imag( m12 ), real( m12 ), (imag(a.u11) - imag(a.u22))/2)
end

function projalg(z::Complex{T}, a::M2x2{T}) where T <: AbstractFloat

    zu11 = z*a.u11
    zu12 = z*a.u12
    zu21 = z*a.u21
    zu22 = z*a.u22

    m12 = (zu12 - conj(zu21))/2

    return SU2alg{T}(imag( m12 ), real( m12 ), (imag(zu11) - imag(zu22))/2)
end


#project SU2fund to M2x2
fundtoMat(g::SU2fund{T})    where T <: AbstractFloat = M2x2{T}(g.t1, g.t2, -conj(g.t2), conj(g.t1))

# dummy structs for dispatch:
# Basis of \\Gamma_n
struct Pauli{N}
end

#multiplication by pauli matrices
fundXpauli(g::SU2fund{T}, ::Type{Pauli{1}}) where T <: AbstractFloat = M2x2{T}(complex(0,1)*g.t2, complex(0,1)*g.t1, complex(0,1)*conj(g.t1),-complex(0,1)*conj(g.t2))
fundXpauli(g::SU2fund{T}, ::Type{Pauli{2}}) where T <: AbstractFloat = M2x2{T}(-g.t2, g.t1, -conj(g.t1), -conj(g.t2))
fundXpauli(g::SU2fund{T}, ::Type{Pauli{3}}) where T <: AbstractFloat = M2x2{T}(complex(0,1)*g.t1, -complex(0,1)*g.t2, -complex(0,1)*conj(g.t2), -complex(0,1)*conj(g.t1))

# =============== PIETRO ================
#Tr(g*i*PauliMatrix)
tr_ipau(g::SU2{T}, ::Type{Pauli{1}}) where T <: AbstractFloat = complex(-2. * imag(g.t2), 0.0)
tr_ipau(g::SU2{T}, ::Type{Pauli{2}}) where T <: AbstractFloat = complex(-2. * real(g.t2), 0.0)
tr_ipau(g::SU2{T}, ::Type{Pauli{3}}) where T <: AbstractFloat = complex(-2. * imag(g.t1), 0.0)
# ========================================