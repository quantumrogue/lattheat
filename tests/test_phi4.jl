using CUDA, Logging, StructArrays, Random, TimerOutputs

CUDA.allowscalar(true)
import Pkg
Pkg.activate("/lhome/ific/a/alramos/s.images/julia/workspace/LatticeGPU")
#Pkg.activate("/home/alberto/code/julia/LatticeGPU")
using LatticeGPU

lp = SpaceParm{4}((16,16,16,16), (4,4,4,4))
ϕp = Phi4Parm(.5,.6325)
PRC = Float64

ϕws = Phi4workspace(PREC,lp)