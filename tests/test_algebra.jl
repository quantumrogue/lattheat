using Symbolics, LinearAlgebra

function SU2alg(v1,v2,v3) 
    u11 = complex(0.,v3)/2
    u22 = conj(u11)
    u12 = complex(v2,v1)/2
    u21 = -conj(u12)

    return [u11 u12; u21 u22]
end

function projalg(M::Matrix)
    w = (M[1,1] - M[2,2])/2
    v = (M[1,2] - conj(M[2,1]))/2

    return SU2alg(imag(v),real(v),imag(w))
end


iσ₁ = im*[0 1; 1 0]
iσ₂ = im*[0 -im; im 0]
iσ₃ = im*[1 0; 0 -1]
function triσ(M::Matrix)
    return tr(iσ₁*M)*iσ₁ + tr(iσ₂*M)*iσ₂ + tr(iσ₃*M)*iσ₃  
end




@variables a₁,a₂, b₁,b₂, c₁,c₂, d₁,d₂
M = [
    complex(a₁,a₂) complex(b₁,b₂);
    complex(c₁,c₂) complex(d₁,d₂)
]

# projalg does project everything over the LinearAlgebra
tr(projalg(M))
projalg(M) + projalg(M)'

# For a generic matrix triσ(M)∉su(2)
tr(triσ(M))
triσ(M)+triσ(M)'  # this is ==0 true only if M∈SU(2) 

# => for a generic matrix, `projalg` and `triσ` are **NOT** the same thing.
# ==========================

@variables u₁, u₂, u₃, u₀
U = SU2alg(u₁, u₂, u₃) + I*u₀/2

triσ(U)+triσ(U)'

# triσ(U)-projalg(U)'