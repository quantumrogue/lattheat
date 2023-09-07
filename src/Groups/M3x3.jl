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

tr(a::M3x3{T})            where T <: AbstractFloat = a.u11+a.u22+a.u33

Base.:*(a::M3x3{T},b::M3x3{T}) where T <: AbstractFloat = M3x3{T}(a.u11*b.u11 + a.u12*b.u21 + a.u13*b.u31,
                                                                  a.u11*b.u12 + a.u12*b.u22 + a.u13*b.u32, 
                                                                  a.u11*b.u13 + a.u12*b.u23 + a.u13*b.u33, 
                                                                  a.u21*b.u11 + a.u22*b.u21 + a.u23*b.u31, 
                                                                  a.u21*b.u12 + a.u22*b.u22 + a.u23*b.u32,
                                                                  a.u21*b.u13 + a.u22*b.u23 + a.u23*b.u33,
                                                                  a.u31*b.u11 + a.u32*b.u21 + a.u33*b.u31, 
                                                                  a.u31*b.u12 + a.u32*b.u22 + a.u33*b.u32,
                                                                  a.u31*b.u13 + a.u32*b.u23 + a.u33*b.u33)

function Base.:*(a::SU3{T},b::M3x3{T}) where T <: AbstractFloat
    
    au31 = conj(a.u12*a.u23 - a.u13*a.u22)
    au32 = conj(a.u13*a.u21 - a.u11*a.u23)
    au33 = conj(a.u11*a.u22 - a.u12*a.u21)

    return M3x3{T}(a.u11*b.u11 + a.u12*b.u21 + a.u13*b.u31,
                   a.u11*b.u12 + a.u12*b.u22 + a.u13*b.u32, 
                   a.u11*b.u13 + a.u12*b.u23 + a.u13*b.u33, 
                   a.u21*b.u11 + a.u22*b.u21 + a.u23*b.u31, 
                   a.u21*b.u12 + a.u22*b.u22 + a.u23*b.u32,
                   a.u21*b.u13 + a.u22*b.u23 + a.u23*b.u33,
                   au31*b.u11  + au32*b.u21  + au33*b.u31, 
                   au31*b.u12  + au32*b.u22  + au33*b.u32,
                   au31*b.u13  + au32*b.u23  + au33*b.u33)
end

    
function Base.:*(a::M3x3{T},b::SU3{T}) where T <: AbstractFloat

    bu31 = conj(b.u12*b.u23 - b.u13*b.u22)
    bu32 = conj(b.u13*b.u21 - b.u11*b.u23)
    bu33 = conj(b.u11*b.u22 - b.u12*b.u21)

    return M3x3{T}(a.u11*b.u11 + a.u12*b.u21 + a.u13*bu31,
                   a.u11*b.u12 + a.u12*b.u22 + a.u13*bu32, 
                   a.u11*b.u13 + a.u12*b.u23 + a.u13*bu33, 
                   a.u21*b.u11 + a.u22*b.u21 + a.u23*bu31, 
                   a.u21*b.u12 + a.u22*b.u22 + a.u23*bu32,
                   a.u21*b.u13 + a.u22*b.u23 + a.u23*bu33,
                   a.u31*b.u11 + a.u32*b.u21 + a.u33*bu31, 
                   a.u31*b.u12 + a.u32*b.u22 + a.u33*bu32,
                   a.u31*b.u13 + a.u32*b.u23 + a.u33*bu33)
end

function Base.:/(a::M3x3{T},b::SU3{T}) where T <: AbstractFloat

    bu31 = (b.u12*b.u23 - b.u13*b.u22)
    bu32 = (b.u13*b.u21 - b.u11*b.u23)
    bu33 = (b.u11*b.u22 - b.u12*b.u21)

    return M3x3{T}(a.u11*conj(b.u11) + a.u12*conj(b.u12) + a.u13*conj(b.u13),
                   a.u11*conj(b.u21) + a.u12*conj(b.u22) + a.u13*conj(b.u23), 
                   a.u11*(bu31) + a.u12*(bu32) + a.u13*(bu33), 
                   a.u21*conj(b.u11) + a.u22*conj(b.u12) + a.u23*conj(b.u13), 
                   a.u21*conj(b.u21) + a.u22*conj(b.u22) + a.u23*conj(b.u23),
                   a.u21*(bu31) + a.u22*(bu32) + a.u23*(bu33),
                   a.u31*conj(b.u11) + a.u32*conj(b.u12) + a.u33*conj(b.u13), 
                   a.u31*conj(b.u21) + a.u32*conj(b.u22) + a.u33*conj(b.u23),
                   a.u31*(bu31) + a.u32*(bu32) + a.u33*(bu33))
end

function Base.:\(a::SU3{T},b::M3x3{T}) where T <: AbstractFloat

    au31 = (a.u12*a.u23 - a.u13*a.u22)
    au32 = (a.u13*a.u21 - a.u11*a.u23)
    au33 = (a.u11*a.u22 - a.u12*a.u21)

    return M3x3{T}(conj(a.u11)*b.u11 + conj(a.u21)*b.u21 + (au31)*b.u31,
                   conj(a.u11)*b.u12 + conj(a.u21)*b.u22 + (au31)*b.u32, 
                   conj(a.u11)*b.u13 + conj(a.u21)*b.u23 + (au31)*b.u33, 
                   conj(a.u12)*b.u11 + conj(a.u22)*b.u21 + (au32)*b.u31, 
                   conj(a.u12)*b.u12 + conj(a.u22)*b.u22 + (au32)*b.u32,
                   conj(a.u12)*b.u13 + conj(a.u22)*b.u23 + (au32)*b.u33,
                   conj(a.u13)*b.u11 + conj(a.u23)*b.u21 + (au33)*b.u31, 
                   conj(a.u13)*b.u12 + conj(a.u23)*b.u22 + (au33)*b.u32,
                   conj(a.u13)*b.u13 + conj(a.u23)*b.u23 + (au33)*b.u33)

end

Base.:*(a::Number,b::M3x3{T}) where T <: AbstractFloat  = M3x3{T}(a*b.u11, a*b.u12, a*bu13,
                                                                  a*b.u21, a*b.u22, a*bu23,
                                                                  a*b.u31, a*b.u32, a*bu33)

Base.:*(b::M3x3{T},a::Number) where T <: AbstractFloat  = M3x3{T}(a*b.u11, a*b.u12, a*bu13,
                                                                  a*b.u21, a*b.u22, a*bu23,
                                                                  a*b.u31, a*b.u32, a*bu33)

Base.:+(a::M3x3{T},b::M3x3{T}) where T <: AbstractFloat = M3x3{T}(a.u11+b.u11, a.u12+b.u12, a.u13+bu13,
                                                                  a.u21+b.u21, a.u22+b.u22, a.u23+bu23,
                                                                  a.u31+b.u31, a.u32+b.u32, a.u33+bu33)

Base.:-(a::M3x3{T},b::M3x3{T}) where T <: AbstractFloat = M3x3{T}(a.u11-b.u11, a.u12-b.u12, a.u13-bu13,
                                                                  a.u21-b.u21, a.u22-b.u22, a.u23-bu23,
                                                                  a.u31-b.u31, a.u32-b.u32, a.u33-bu33)

Base.:-(b::M3x3{T}) where T <: AbstractFloat            = M3x3{T}(-b.u11, -b.u12, -bu13,
                                                                  -b.u21, -b.u22, -bu23,
                                                                  -b.u31, -b.u32, -bu33)

Base.:+(b::M3x3{T}) where T <: AbstractFloat            = M3x3{T}(b.u11, b.u12, bu13,
                                                                  b.u21, b.u22, bu23,
                                                                  b.u31, b.u32, bu33)
function projalg(a::M3x3{T}) where T <: AbstractFloat

    sr3ov2::T = 0.866025403784438646763723170752

    ditr = ( imag(a.u11) + imag(a.u22) - 2.0*imag(a.u33) )/3.0
    m12 = (a.u12 - conj(a.u21))/2.0
    m13 = (a.u13 - conj(a.u31))/2.0
    m23 = (a.u23 - conj(a.u32))/2.0

    return SU3alg{T}(imag( m12 ), imag( m13 ), imag( m23 ),
                     real( m12 ), real( m13 ), real( m23 ),
                     (imag(a.u11)-imag(a.u22))/2.0,
                     sr3ov2*(ditr))
end

function projalg(z::Complex{T}, a::M3x3{T}) where T <: AbstractFloat

    sr3ov2::T = 0.866025403784438646763723170752

    zu11 = z*a.u11
    zu12 = z*a.u12
    zu13 = z*a.u13
    zu21 = z*a.u21
    zu22 = z*a.u22
    zu23 = z*a.u23
    zu31 = z*a.u31
    zu32 = z*a.u32
    zu33 = z*a.u33

    ditr = ( imag(zu11) + imag(zu22) - 2.0*imag(zu33) )/3.0
    m12 = (zu12 - conj(zu21))/2.0
    m13 = (zu13 - conj(zu31))/2.0
    m23 = (zu23 - conj(zu32))/2.0

    return SU3alg{T}(imag( m12 ), imag( m13 ), imag( m23 ),
                     real( m12 ), real( m13 ), real( m23 ),
                     (imag(zu11)-imag(zu22))/2.0,
                     sr3ov2*(ditr))
end
