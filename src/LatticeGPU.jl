###
### "THE BEER-WARE LICENSE":
### Alberto Ramos wrote this file. As long as you retain this 
### notice you can do whatever you want with this stuff. If we meet some 
### day, and you think this stuff is worth it, you can buy me a beer in 
### return. <alberto.ramos@cern.ch>
###
### file:    LatticeGPU.jl
### created: Sat Jul 17 17:19:58 2021
###                               


module LatticeGPU

include("Groups/Groups.jl")

using .Groups
export Group, Algebra
export SU2, SU2alg, SU2fund, SU3, SU3alg, M3x3, M2x2, U1, U1alg, SU3fund, fundtoMat, pauli, Pauli, fundXpauli,fundipau,tr_ipau
export dot, expm, exp, dag, unitarize, inverse, tr, projalg, norm, norm2, isgroup, alg2mat, dev_one, adjaction

include("Space/Space.jl")

using .Space
export SpaceParm
export up, dw, updw, point_coord, point_index, point_time
export BC_PERIODIC, BC_OPEN, BC_SF_AFWB, BC_SF_ORBI

include("Fields/Fields.jl")
using .Fields
export vector_field, scalar_field, nscalar_field, scalar_field_point

include("MD/MD.jl")
using .MD
export IntrScheme
export omf4, leapfrog, omf2

include("YM/YM.jl")

using .YM
export ztwist
export YMworkspace, GaugeParm, force0_wilson!, field, field_pln, randomize!, zero!, norm2
export gauge_action, hamiltonian, plaquette, HMC!, OMF4!
export wfl_euler, wfl_rk3, zfl_euler, zfl_rk3, Eoft_clover, Eoft_plaq, Qtop, dEoft_clover, dEoft_plaq
export FlowIntr, wfl_euler, zfl_euler, wfl_rk2, zfl_rk2, wfl_rk3, zfl_rk3
export flw, flw_adapt
export sfcoupling, bndfield, setbndfield
export import_lex64, import_cern64, save_cnfg, read_cnfg

include("Scalar/Scalar.jl")

using .Scalar
export ScalarParm, ScalarWorkspace, smr
export scalar_action, force_scalar
export HMC_Scalar!, randomize!
export scalar_obs, scalar_corr, mixed_corr, smearing


include("ZQCD/ZQCD.jl")

using .ZQCD
export ZQCDParm, ZQCDworkspace
export randomize!
export zqcd_action
export zqcd_force
export hamiltonian, MD!, HMC!


# include("Phi4/Phi4.jl")

# using .Phi4
# export Phi4Parm, Phi4ParmM2L, Phi4workspace
# export randomize!
# export phi4_action, hopping
# export phi4_force
# export hamiltonian, MD!, HMC!

end # module
