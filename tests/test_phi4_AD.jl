using CUDA, Logging, StructArrays, Random, TimerOutputs, Statistics

CUDA.allowscalar(false)
import Pkg
Pkg.activate("../LatticeGPU")
using LatticeGPU

lp = SpaceParm{4}((32,32,32,32), (4,4,4,4))
ϕp = Phi4Parm(.05,.3)
PRC = Float64

ϕws = Phi4workspace(PRC,lp)
PHI = scalar_field(PRC,lp)
ϕ   = scalar_field(PRC,lp)

int = omf2(Float64,0.1,10)

# ==============================================================
import LatticeGPU.Phi4.randomize!
randomize!(ϕ,ϕws,ϕp,lp)



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








# import LatticeGPU.Phi4.randomize!
# import LatticeGPU.Phi4.hamiltonian


# ϕ .= PHI.*0.
# Δτ = 0.1
# # LOG() = println(hamiltonian(ϕws.mom,ϕ,lp,ϕp,ϕws))

# randomize!(ϕws.mom,ϕws,ϕp,lp)
# randomize!(ϕ,ϕws,ϕp,lp)

# Hin = hamiltonian(ϕws.mom,ϕ,lp,ϕp,ϕws)

# for i in 1:10
#     Hin = hamiltonian(ϕws.mom,ϕ,lp,ϕp,ϕws)

#     phi4_force(ϕws,ϕ,ϕp,lp)
#     ϕws.mom .= ϕws.mom .- Δτ/2 .* ϕws.frc
#     ϕ .= ϕ .+ Δτ .* ϕws.mom
#     phi4_force(ϕws,ϕ,ϕp,lp)
#     ϕws.mom .= ϕws.mom .- Δτ/2 .* ϕws.frc
#     println(hamiltonian(ϕws.mom,ϕ,lp,ϕp,ϕws) - Hin)
# end







