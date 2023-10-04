using CUDA, Logging, StructArrays, Random, TimerOutputs, Statistics

CUDA.allowscalar(true)
import Pkg
Pkg.activate("/LatticeGPU")
using LatticeGPU

lp = SpaceParm{4}((16,16,16,16), (4,4,4,4))
ϕp = Phi4Parm(.5,.6325)
PRC = Float64

ϕws = Phi4workspace(PRC,lp)
ϕ = scalar_field(PRC,lp)




dt  = 0.05
ns  = 20

int = leapfrog(PRC,dt,ns)

println("## Thermalization")
for _ in 1:10
    @time dh, acc = LatticeGPU.Phi4.HMC!(ϕ,int,lp,ϕp,ϕws,noacc=true)
    println("# HMC: ", acc, " ", exp(-dh))
    println("# ⟨ϕ²⟩ = ",mean(abs2.(ϕ)),"\n")
end


println("## Production")
for _ in 1:10
    @time dh, acc = LatticeGPU.Phi4.HMC!(ϕ,int,lp,ϕp,ϕws,noacc=true)
    println("# HMC: ", acc, " ", exp(-dh))
    println("# ⟨ϕ²⟩ = ",mean(abs2.(ϕ)),"\n")
end