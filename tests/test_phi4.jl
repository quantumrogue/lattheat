using CUDA, Logging, StructArrays, Random, TimerOutputs, Statistics

CUDA.allowscalar(true)
import Pkg
Pkg.activate("/LatticeGPU")
using LatticeGPU

lp = SpaceParm{4}((16,16,16,16), (4,4,4,4))
ϕp = Phi4Parm(0.185825,1.1689)
PRC = Float64

ϕws = Phi4workspace(PRC,lp)
ϕ = scalar_field(PRC,lp)

int  = leapfrog(PRC,.02,10)


println("## Production")
for i in 1:10000
    dh, acc = LatticeGPU.Phi4.HMC!(ϕ,int,lp,ϕp,ϕws)
    println(
        dh,"  ",
        mean(ϕ),"  ",
        mean(abs2.(ϕ)),"  ",
        phi4_action(ϕ,lp,ϕp,ϕws)/prod(lp.iL)
    )
end





# const Nsteps = 10
# const dt = .02


# for tMC in 1:1000
#     # Copy gauge fields
#     ϕws.ϕ .= ϕ

#     # Initialize momenta
#     LatticeGPU.Phi4.randomize!(ϕws.mom,ϕws,ϕp,lp)

#     # Calculate initial hamiltonian
#     Hin = LatticeGPU.Phi4.hamiltonian(ϕws.mom,ϕ,lp,ϕp,ϕws)

#     # Perform molecular dynamics step
#     for _ in 1:Nsteps
#         md!(ϕws.mom,ϕ,dt,lp,ϕp,ϕws)
#     end

#     # Perform metropolis
#     ΔH = LatticeGPU.Phi4.hamiltonian(ϕws.mom,ϕ,lp,ϕp,ϕws) - Hin
#     println("# ΔH = $ΔH")
#     if ΔH>0
#         if exp(-ΔH)<rand()
#             ϕ .= ϕws.ϕ
#         else
#             println(",     accepted")
#         end 
#     else
#         println(",    accepted")
#     end
# end