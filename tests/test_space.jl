
import Pkg
#Pkg.activate("/lhome/ific/a/alramos/s.images/julia/workspace/LatticeGPU")
Pkg.activate("/home/alberto/code/julia/LatticeGPU")
using LatticeGPU, BenchmarkTools


lp = SpaceParm{4}((8,12,6,6), (8,2,2,3))

function test_point(pt::NTuple{2,Int64}, lp::SpaceParm)
    ok = true
    println("Global point: ", global_point(pt, lp))
    for id in 1:lp.ndim
        ua, ub = up(pt, id, lp)
        println("  - UP in id $id: ", global_point((ua,ub), lp))
        
        da, db = dw(pt, id, lp)
        println("  - DW in id $id: ", global_point((da,db), lp), "\n")
        
        ua2, ub2, da2, db2 = updw(pt, id, lp)
        ok = ok && (ua == ua2)
        ok = ok && (ub == ub2)
        ok = ok && (da == da2)
        ok = ok && (db == db2)
    end
    return ok
end

global ok = true
for i in 1:lp.bsz, j in 1:lp.rsz
    global ok = ok && test_point((i,j), lp)
end

if ok
    println("ALL tests passed")
else
    println("ERROR in test")
end
    
println(lp)

