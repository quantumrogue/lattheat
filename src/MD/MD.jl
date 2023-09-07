###
### "THE BEER-WARE LICENSE":
### Alberto Ramos wrote this file. As long as you retain this 
### notice you can do whatever you want with this stuff. If we meet some 
### day, and you think this stuff is worth it, you can buy me a beer in 
### return. <alberto.ramos@cern.ch>
###
### file:    MD.jl
### created: Fri Oct  8 21:53:14 2021
###                               

module MD

# Dalla Brida / Luscher coefficients of
# OMF integrator
const r1omf4 =  0.08398315262876693
const r2omf4 =  0.25397851084105950
const r3omf4 =  0.68223653357190910
const r4omf4 = -0.03230286765269967
const r5omf4 =  0.5-r1omf4-r3omf4
const r6omf4 =  1.0-2.0*(r2omf4+r4omf4)

const r1omf2 =  0.1931833275037836
const r2omf2 =  0.5
const r3omf2 =  1 - 2*r1omf2

struct IntrScheme{N, T}
    r::NTuple{N, T}
    eps::T
    ns::Int64    
end


omf2(::Type{T}, eps, ns) where T = IntrScheme{3,T}((r1omf2,r2omf2,r3omf2), eps, ns)
omf4(::Type{T}, eps, ns) where T = IntrScheme{6,T}((r1omf4,r2omf4,r3omf4,r4omf4,r5omf4,r6omf4), eps, ns)
leapfrog(::Type{T}, eps, ns) where T = IntrScheme{2,T}((0.5,1.0), eps, ns)


import Base.show
function Base.show(io::IO, int::IntrScheme{N,T}) where {N,T}

    if N == 2
        println(io, "LEAPFROG integration scheme")
    elseif N == 3
        println(io, "OMF2 integration scheme")
    elseif N == 6
        println(io, "OMF4 integration scheme")
    end
    println(io, "  - eps: ", int.eps)
    println(io, "  - ns:  ", int.ns)
    
    return nothing
end
    
export IntrScheme, omf4, leapfrog, omf2


end
