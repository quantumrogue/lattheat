using CUDA, Logging, StructArrays, Random, TimerOutputs
using ArgParse

CUDA.allowscalar(true)
import Pkg
Pkg.activate("/home/pbutti/lattheat/LatticeGPU")
using LatticeGPU

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--time", "-T"
            help = "Time extent"
            required = true
            arg_type = Int
        "--space", "-L"
            help = "Space extent"
            required = true
            arg_type = Int
            default = 16
        "--beta"
            help = "Gauge coupling"
            required = true
            arg_type = Float64
        "--dims"
            help = "Number of total dimension"
            required = true
            arg_type = Int
        # "arg1"
        #     help = "Output file name"
        #     required = true
    end

    return parse_args(s)
end


function gaugeheater!(f, lp::SpaceParm, ymws::YMworkspace)
    @timeit "Randomize SU(2) gauge field" begin
        m = CUDA.randn(ymws.PRC, lp.bsz,lp.ndim,4,lp.rsz)
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_gaugeheater!(f,m,lp)
        end
        f .= unitarize.(f)
    end
    return nothing
end
function krnl_gaugeheater!(f, m, lp::SpaceParm{N,M,BC_PERIODIC,D}) where {N,M,D}
    b, r = CUDA.threadIdx().x, CUDA.blockIdx().x
    for id in 1:lp.ndim
        f[b,id,r] = SU2(complex(m[b,id,1,r], m[b,id,2,r]), complex(m[b,id,3,r],m[b,id,4,r]))
    end
    return nothing
end




function main()
    parsed_args = parse_commandline()

    println("# Parsed args:")
    for (arg,val) in parsed_args
        println("#  $arg  =>  $val")
    end

    L = parsed_args["space"]
    T = parsed_args["time"]
    dim = parsed_args["dims"]
    β = parsed_args["beta"]

    if L%4!=0
        error("Space extent must be multiple of 4")
    end

    lp = 0
    if dim==4
        lp = SpaceParm{4}((T,L,L,L), (1,4,4))
    elseif dim==3
        lp = SpaceParm{3}((T,L,L), (1,4,4))
    end

    GRP  = SU2
    PREC = Float64
    println("# Precision:         ", PREC)

    # gp = GaugeParm(β, 1.0, (0.0,0.0), 2)
    gp = GaugeParm{PREC}(GRP{PREC},β,0.)



    println("# Allocating YM workspace")
    ymws = YMworkspace(GRP, PREC, lp)

    println("# Seeding CURAND...")
    Random.seed!(CURAND.default_rng(), 1234)
    Random.seed!(1234)

    println("# Allocating gauge field")
    U = vector_field(GRP{PREC}, lp)
    fill!(U, one(GRP{PREC}))
    
    dt  = 0.05
    ns  = 20
    
    # pl = Vector{Float64}()
    gaugeheater!(U,lp,ymws)

    
    println("## ======================== Production ====================")
    for i in 1:10000
        @time dh, acc = HMC!(U, dt,ns,lp, gp, ymws)
        # println("# HMC: ", acc, " ", dh)
        # push!(pl, plaquette(U,lp, gp, ymws))
        println("# Plaquette( $i): ", plaquette(U,lp, gp, ymws), "\n")

    end
end

main()