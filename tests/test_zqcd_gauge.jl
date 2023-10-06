using CUDA, Logging, StructArrays, Random, TimerOutputs, ADerrors, Printf, BDIO
using LatticeGPU

lp = SpaceParm{3}((8,8,8), (4,4,4))
gp = GaugeParm{Float64}(SU2{Float64}, 12, 0.)
ymws = YMworkspace(SU2, Float64, lp)

U = vector_field(SU2{Float64}, lp)
fill!(U, one(SU2{Float64}))

gauge_action(U,lp,gp,ymws)