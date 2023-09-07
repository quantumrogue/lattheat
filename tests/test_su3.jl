using CUDA, LinearAlgebra

import Pkg
#Pkg.activate("/lhome/ific/a/alramos/s.images/julia/workspace/LatticeGPU")
Pkg.activate("/home/alberto/code/julia/LatticeGPU")
using LatticeGPU

function g2mat(g::SU3)

    M = Array{ComplexF64, 2}(undef, 3,3)

    M[1,1] = g.u11
    M[1,2] = g.u12
    M[1,3] = g.u13

    M[2,1] = g.u21
    M[2,2] = g.u22
    M[2,3] = g.u23

    M[3,1] = conj(g.u12*g.u23 - g.u13*g.u22)
    M[3,2] = conj(g.u13*g.u21 - g.u11*g.u23)
    M[3,3] = conj(g.u11*g.u22 - g.u12*g.u21)

    return M
end
    


# 0.40284714488721746 + 0.2704272209422031im -0.029482825024553627 - 0.8247329455356851im 0.28771631112777535 + 0.027366985901323956im; -0.08478364480998268 + 0.8226014762207954im -0.4790638417896126 + 0.24301903735299646im -0.022591091614522323 + 0.16452285690920823im; 0.28083864951126214 + 0.04302898862961919im 0.0066864552013863165 - 0.17418727240313508im -0.939634663641523 + 0.07732362776719631im

T  = Float64
a = rand(SU3alg{T})
println("Random algebra: ", a)
g1 = exp(a, 0.2)
g2 = exp(a, -0.2)

g = expm(g1, a, -0.2)
println(g)





M = [0.40284714488721746 + 0.2704272209422031im -0.029482825024553627 - 0.8247329455356851im 0.28771631112777535 + 0.027366985901323956im; -0.08478364480998268 + 0.8226014762207954im -0.4790638417896126 + 0.24301903735299646im -0.022591091614522323 + 0.16452285690920823im; 0.28083864951126214 + 0.04302898862961919im 0.0066864552013863165 - 0.17418727240313508im -0.939634663641523 + 0.07732362776719631im]

println(det(M))


g1 = SU3(M[1,1],M[1,2],M[1,3], M[2,1],M[2,2],M[2,3])
println("dir: ", g1)
g2 = exp(a)
println("exp: ", g2)
println("dif: ", g2mat(g1)-g2mat(g2))

g3 = g1/g2
println(g3)

println("END")

ftest(g::Group) = LatticeGPU.tr(g)

#println(" ## SU(2)")
#asu2 = SU2alg(0.23, 1.23, -0.34)
#gsu2 = exp(asu2) 
#
#eps = 0.001
#h   = SU2alg(eps,0.0,0.0)
#fp = ftest(exp(h)*gsu2)
#fm = ftest(exp(h,-1.0)*gsu2)
#println("Numerical derivative: ", (fp-fm)/(2.0*eps))
#h   = SU2alg(0.0,eps,0.0)
#fp = ftest(exp(h)*gsu2)
#fm = ftest(exp(h,-1.0)*gsu2)
#println("Numerical derivative: ", (fp-fm)/(2.0*eps))
#h   = SU2alg(0.0,0.0,eps)
#fp = ftest(exp(h)*gsu2)
#fm = ftest(exp(h,-1.0)*gsu2)
#println("Numerical derivative: ", (fp-fm)/(2.0*eps))
#println("Exact     derivative: ", -projalg(gsu2))



println("\n\n ## SU(3)")
asu3 = SU3alg{T}(0.23, 1.23, -0.34, 2.34, -0.23, 0.23, -1.34, 1.34)
gsu3 = exp(asu3) 

eps = 0.001
h   = SU3alg{T}(eps,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
fp = ftest(exp(h)*gsu3)
fm = ftest(exp(h,-1.0)*gsu3)
println("Numerical derivative: ", (fp-fm)/(2.0*eps))
h   = SU3alg{T}(0.0,eps,0.0,0.0,0.0,0.0,0.0,0.0)
fp = ftest(exp(h)*gsu3)
fm = ftest(exp(h,-1.0)*gsu3)
println("Numerical derivative: ", (fp-fm)/(2.0*eps))
h   = SU3alg{T}(0.0,0.0,eps,0.0,0.0,0.0,0.0,0.0)
fp = ftest(exp(h)*gsu3)
fm = ftest(exp(h,-1.0)*gsu3)
println("Numerical derivative: ", (fp-fm)/(2.0*eps))
h   = SU3alg{T}(0.0,0.0,0.0,eps,0.0,0.0,0.0,0.0)
fp = ftest(exp(h)*gsu3)
fm = ftest(exp(h,-1.0)*gsu3)
println("Numerical derivative: ", (fp-fm)/(2.0*eps))
h   = SU3alg{T}(0.0,0.0,0.0,0.0,eps,0.0,0.0,0.0)
fp = ftest(exp(h)*gsu3)
fm = ftest(exp(h,-1.0)*gsu3)
println("Numerical derivative: ", (fp-fm)/(2.0*eps))
h   = SU3alg{T}(0.0,0.0,0.0,0.0,0.0,eps,0.0,0.0)
fp = ftest(exp(h)*gsu3)
fm = ftest(exp(h,-1.0)*gsu3)
println("Numerical derivative: ", (fp-fm)/(2.0*eps))
h   = SU3alg{T}(0.0,0.0,0.0,0.0,0.0,0.0,eps,0.0)
fp = ftest(exp(h)*gsu3)
fm = ftest(exp(h,-1.0)*gsu3)
println("Numerical derivative: ", (fp-fm)/(2.0*eps))
h   = SU3alg{T}(0.0,0.0,0.0,0.0,0.0,0.0,0.0,eps)
fp = ftest(exp(h)*gsu3)
fm = ftest(exp(h,-1.0)*gsu3)
println("Numerical derivative: ", (fp-fm)/(2.0*eps))
println("Exact     derivative: ", -projalg(gsu3))

println("\n # Mutiplications")

g1 = exp(SU3alg{T}(0.23, 1.23, -0.34, 2.34, -0.23, 0.23, -1.34, 1.34))
g2 = exp(SU3alg{T}(1.23, -0.23, -0.14, 0.4, -1.23, -0.8, -0.34, 0.34))

a = g1/(g2*g1)
b = g2*a
println("b is one: ", b)


println("## Aqui test M3x3")
ba = rand(SU3alg{T})
ga = exp(ba)
println("Matrix: ", alg2mat(ba))
println("Exp:    ", ga)
