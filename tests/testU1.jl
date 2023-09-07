using LinearAlgebra, Random

import Pkg
#Pkg.activate("/lhome/ific/a/alramos/s.images/julia/workspace/LatticeGPU")
Pkg.activate("/home/alberto/code/julia/LatticeGPU")
using LatticeGPU


T = Float32

b = rand(U1{T})
println(b)

ba = rand(U1alg{T})
println("Ba:        ", ba)
b = exp(ba)
println("B:         ", b)
c = exp(ba, convert(T,-1))
println(typeof(norm2(ba)))
d = b*c
println("Test:      ", d)

c = inverse(b)
println("Inverse B: ", c)

d = b*c
println("Test:      ", d)

println("B:         ", b)
println("Ba:        ", ba)
b = expm(b, ba, convert(T,-1))
println("Test:      ", b)


Ma = Array{U1{T}}(undef, 2)
rand!(Ma)
println(Ma)

fill!(Ma, one(eltype(Ma)))
println(Ma)
