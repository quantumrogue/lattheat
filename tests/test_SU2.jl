using LinearAlgebra, Random

import Pkg
#Pkg.activate("/lhome/ific/a/alramos/s.images/julia/workspace/LatticeGPU")
Pkg.activate("/home/alberto/code/julia/LatticeGPU")
using LatticeGPU


T = Float64

b = rand(SU2{T})
println(b)

ba = rand(SU2alg{T})
println("Ba:        ", ba)
b = exp(ba)
println("B:         ", b)
println(typeof(norm2(ba)))

c = inverse(b)
println("Inverse B: ", c)

d = b*c
println("Test:      ", d)

c = exp(ba, -1.0)
println("Inverse B: ", c)

d = b*c
println("Test:      ", d)

Ma = Array{SU2{T}}(undef, 2)
rand!(Ma)
println(Ma)

fill!(Ma, one(eltype(Ma)))
println(Ma)

println("## Aqui test M2x2")
ba = rand(SU2alg{T})
ga = exp(ba)
println("Matrix: ", alg2mat(ba))
println("Exp:    ", ga)


mo = one(M2x2{T})
println(mo)
mp = mo*ga
println(mp)
println(projalg(mp))
println(projalg(ga))

println("## HERE test SU(2) fundamental")
a = rand(SU2fund{T})
println(a)

ft(x) = LatticeGPU.dot(x,x)
f2(x) = LatticeGPU.tr(dag(x)*x)
f3(x) = LatticeGPU.norm2(x)

S = ft(a)
println("Check 1: ", S)
S = f2(a)
println("Check 2: ", S)
S = f3(a)
println("Check 3: ", S)

h = 0.000001
xh = SU2fund{T}(a.t1+h, a.t2)
Sp = ft(xh)
xh = SU2fund{T}(a.t1-h, a.t2)
Sm = ft(xh)
println("Numerical derivative: ", (Sp-Sm)/(2*h))
xh = SU2fund{T}(a.t1, a.t2+h)
Sp = ft(xh)
xh = SU2fund{T}(a.t1, a.t2-h)
Sm = ft(xh)
println("Numerical derivative: ", (Sp-Sm)/(2*h))
h = 0.000001im
xh = SU2fund{T}(a.t1+h, a.t2)
Sp = ft(xh)
xh = SU2fund{T}(a.t1-h, a.t2)
Sm = ft(xh)
println("Numerical derivative: ", 1im * (Sp-Sm)/(2*h))
xh = SU2fund{T}(a.t1, a.t2+h)
Sp = ft(xh)
xh = SU2fund{T}(a.t1, a.t2-h)
Sm = ft(xh)
println("Numerical derivative: ", 1im * (Sp-Sm)/(2*h))

ad = a
println("Exact derivative:     ", ad)



println("## Hamiltonian dynamics")
println(" # Mass terms")
H(p,a) = LatticeGPU.norm2(p)/2 + LatticeGPU.norm2(a)
function MD!(p, a, ns, ee)
    for i in 1:ns
        p = p - ee*a
        a = a + ee*p
        p = p - ee*a
    end
    return p, a
end

a = rand(SU2fund{T})
p = rand(SU2fund{T})
println(H(p,a))

ee = 0.0001
ns = 10000
p, a = MD!(p, a, ns, ee)

println(H(p,a))

println(" # Interaction terms")
H(p,a, M,kap) = LatticeGPU.norm2(p)/2 - (2*kap)*LatticeGPU.dot(a, M)
function MD!(p, a, M, kap, ns, ee)
    for i in 1:ns
        p = p + 0.5*( (2*kap)*ee*M )
        a = a + ee*p
        p = p + 0.5*( (2*kap)*ee*M )
    end
    return p, a
end

a = rand(SU2fund{T})
p = rand(SU2fund{T})
M = rand(SU2fund{T})
kap = 0.5
println(H(p,a, M,kap))

ee = 0.0001
ns = 10000
p, a = MD!(p, a, M,kap, ns, ee)

println(H(p,a, M,kap))

println(" # Gauge Interaction terms")
H(p,a, M,kap) = LatticeGPU.norm2(p)/2 - (2*kap)*LatticeGPU.tr(a*M)
function MD!(p, a, M, kap, ns, ee)
    for i in 1:ns
        p = p + 0.5*( - (2*kap)*ee*projalg(a*M) )
        a = expm(a, p, ee)
        p = p + 0.5*( - (2*kap)*ee*projalg(a*M) )
    end
    return p, a
end

a = rand(SU2{T})
p = rand(SU2alg{T})
M = rand(SU2fund{T})
#M = SU2fund{T}(0.0+rand()im, rand()+rand()im)
kap = 1.1
println(H(p,a, M,kap))

ee = 0.0001
ns = 10000
p, a = MD!(p, a, M,kap, ns, ee)

println(H(p,a, M,kap))

println(" # Gauge Interaction terms")
H(p,a, M1,M2,kap) = LatticeGPU.norm2(p)/2 - (2*kap)*LatticeGPU.dot(M1,a*M2)
#H(p,a, M1,M2,kap) = LatticeGPU.norm2(p)/2 - (2*kap)*LatticeGPU.tr(a*(M2*dag(M1)))
function MD!(p, a, M1, M2, kap, ns, ee)
    for i in 1:ns
        p = p + 0.5*( - (2*kap)*ee*projalg(a*M2*dag(M1)) )
        a = expm(a, p, ee)
        p = p + 0.5*( - (2*kap)*ee*projalg(a*M2*dag(M1)) )
    end
    return p, a
end

a = rand(SU2{T})
p = rand(SU2alg{T})
M1 = rand(SU2fund{T})
M2 = rand(SU2fund{T})
#M = SU2fund{T}(0.0+rand()im, rand()+rand()im)
kap = 1.1
println(H(p,a, M1,M2,kap))

ee = 0.0001
ns = 10000
p, a = MD!(p, a, M1,M2,kap, ns, ee)

println(H(p,a, M1,M2,kap))


