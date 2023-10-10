using CUDA, Logging, StructArrays, Random, TimerOutputs, Statistics

CUDA.allowscalar(false)
import Pkg
Pkg.activate("../LatticeGPU")
using LatticeGPU

lp = SpaceParm{4}((32,32,32,32), (4,4,4,4))
ϕp = Phi4Parm(.05,.3)
PRC = Float64

ϕws = Phi4workspace(PRC,lp)
ϕ = scalar_field(PRC,lp)

int  = leapfrog(PRC,.05,20)

println("## Thermalization & production")
for i in 1:10000
    dh, acc = LatticeGPU.Phi4.HMC!(ϕ,int,lp,ϕp,ϕws)
    println(
        dh,"  ",
        mean(abs2.(ϕ)),"  ",
        mean(abs2.(ϕ).^2),"  ",
        phi4_action(ϕ,lp,ϕp,ϕws)/prod(lp.iL),"   ",
        hopping(ϕ,lp,ϕws)/prod(lp.iL)
    )
end
