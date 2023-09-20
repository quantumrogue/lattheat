"""
    2 Higgs
    - Full or Adaptive GF
    - Mass measurements w/ mixed W-boson interpolator
    - saves configuration
    - may start from gauge field configuration
"""

# Measurements
#
# Uinfo 1: Simulation parameters Int64 - [nth, niter,nr_flwtimes,flw_int,flw_iter]
# Uinfo 2: Simulations parameters Float64 - [beta,flw_dt,k2,et1,et2,mu,xi1,xi2,xi3,xi4]
# uid 3:  Plaquette
# uid 4:  rho1, rho2
# uid 5:  Lphi1, Lphi2
# uid 6:  Lalp1, Lalp2
# uid 7:  hash
# uid 8:  dh
# uid 9:  (flwtime) not saved
# uid 10: (Plaquette GF) not saved
# uid 11: (Clover GF) not saved
# uid 12: Correlator H: 11, 22
# uid 13: Correlator W: 11, 22, 12

import Pkg
Pkg.activate("/home/gtelo/PhD/LatGPU/SU2proj/code/runs_su2higgs/latticegpu_dev.jl")
Pkg.instantiate()
Pkg.status()

using CUDA, Logging, StructArrays, Random, TimerOutputs, ADerrors, Printf, BDIO,LatticeGPU, ArgParse, TOML
CUDA.versioninfo()
CUDA.allowscalar(false)

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "-i"
        help = "input file"
        required = true
        arg_type = String
    end

    return parse_args(s)
end


function read_options(fname)

    s = TOML.parsefile(fname)

    BDIO_set_user(s["Run"]["user"])
    BDIO_set_host(s["Run"]["host"])

    # Lattice Parameters
    BCS = BC_PERIODIC
    lp = SpaceParm{4}(Tuple(x for x in s["Simulation"]["size"]),
                        Tuple(x for x in s["Simulation"]["blocks"]),
                        BCS, (0,0,0,0,0,0))

    # Gauge Parameters
    GRP = SU2
    T   = Float64
    gp  = GaugeParm{T}(GRP{T},
                       s["Simulation"]["beta"],
                       s["Simulation"]["c0"])

    # Scalar Parameters
    kappas = s["Simulation"]["kappa"]
    etas = s["Simulation"]["eta"]
    xis = s["Simulation"]["xi"]
    mu = s["Simulation"]["mu"]
    sp = ScalarParm(Tuple(kappas[i] for i in 1:length(kappas)),
                    Tuple(etas[i] for i in 1:length(etas)),
                    mu,
                    Tuple(xis[i] for i in 1:length(xis)))


    hmc_dt = s["HMC"]["eps"]
    nsteps = s["HMC"]["ns"]
    # HMC parameters
    if s["HMC"]["integrator"] == "OMF4"
        int = omf4(T, hmc_dt, nsteps)
    elseif s["HMC"]["integrator"] == "OMF2"
        int = omf2(T, hmc_dt, nsteps)
    elseif s["HMC"]["integrator"] == "LEAPFROG"
        int = leapfrog(T, hmc_dt, nsteps)
    else
        error("Unknown integrator")
    end

    dtr_chk  = s["Checkpoints"]["dtr"]
    cnfg_dir = s["Checkpoints"]["cnfg_dir"]


    flow_bool = s["Flow"]["flow"] #0: no GF; 1: Full GF; 2: adapt step size GF
    flw_type  = s["Flow"]["type"] #Wilson, Zeuthen, Both
    wflwint   = wfl_rk3(T, s["Flow"]["dt"], s["Flow"]["tol"])
    zflwint   = zfl_rk3(T, s["Flow"]["dt"], s["Flow"]["tol"])
    flw_dtr   = s["Flow"]["dtr"] # measure flow every dtr Markov update
    flw_dt = s["Flow"]["dt"] # integration stepsize
    #FULL GF
    flw_steps = s["Flow"]["steps"]
    flw_s     = s["Flow"]["s"] # measure every s*dt after starting time
    #ADAPT STEP SIZE GF
    flw_times    = s["Flow"]["times"] # starting measuring times
    flw_iters    = s["Flow"]["iters"] # iterations

    flwtime = Vector{Float64}()
    if flow_bool == 1
        flwtime = Vector{Float64}(undef, flw_steps)
        flwtime[1] = 0.0
        #compute flow times
        for i in 2:flw_steps
            flwtime[i] = (i-1)*flw_dt*flw_s
        end
    elseif flow_bool == 2
        flwtime = Vector{Float64}(undef, sum(flw_iters)+1+length(flw_iters))
        k = 1
        for i in 1:length(flw_iters)
            for j in 1:flw_iters[i]+1
                k = k + 1
                flwtime[k] = flw_times[i] + (j-1)*flw_dt
            end
        end
    end

    # Smearing
    corr = s["Smearing"]["corr"]
    sus = s["Smearing"]["sus"]
    sdt = s["Smearing"]["sdt"]
    sss = s["Smearing"]["sss"]
    srs = s["Smearing"]["srs"]

    println(" #  [Enviroment Info]")
    println("User: ", s["Run"]["user"])
    println("Host: ", s["Run"]["host"])
    println(" ")

    println(" ")

    Pkg.status()
    println(" ")
    CUDA.versioninfo()
    println(" # END [Enviroment Info]")

    if haskey(s["Run"], "seed")
        rngs = s["Run"]["seed"]
    else
        rngs = convert(UInt64, round(10000*time()))
    end
    println(" # Using seed: ", rngs)
    println("\n")

    Random.seed!(CURAND.default_rng(), rngs)
    Random.seed!(rngs)

    nth  = s["HMC"]["nthm"]
    niter    = s["HMC"]["ntraj"]

    run_dir_name = s["Run"]["run_dir"]*s["Run"]["name"]

    return gp, sp, lp, int, wflwint, zflwint, nth, niter, hmc_dt, nsteps, dtr_chk, cnfg_dir, flow_bool, flw_steps, flw_s, flw_times, flw_iters, flw_dtr, flw_dt, flwtime, flw_type, corr, sus, sdt, sss, srs, run_dir_name
end

function measure_flow!(epl, ecl, u, flwtime, steps, s, flwint, jflw, gp, lp, ymws, flog)

    #Computes Gradient flow for 'steps' steps of length  s*dt

    # epl[jflw,1] = Eoft_plaq(u, gp, lp, ymws)
    ecl[jflw,1] = Eoft_clover(u, gp, lp, ymws)


    if flwint.add_zth
        println(flog, "## integrating zeuthen flow equations")
    else
        println(flog, "## integrating wilson  flow equations")
    end

    for f in 2:steps
        #step flows
        flw(u, wflwint, s, gp, lp, ymws)
        # E(f)
        #epl[jflw,f] = Eoft_plaq(u, gp, lp, ymws)
        ecl[jflw,f] = Eoft_clover(u, gp, lp, ymws)
    end

    if flwint.add_zth
        println(flog, "## zeuthen flow equations integrated in $(steps) steps")
    else
        println(flog, "## wilson  flow equations integrated in $(steps) steps")
    end

    return nothing

end


function measure_flow!(epl, ecl, u, flwtime, flwint, jflw, gp, lp, ymws, flog)

    # flwtime    - flow times
    # flwint     - flow integrator
    # jflw       - flow measurement jflw=(1,flw_iter)

    epl[jflw,1] = Eoft_plaq(u, gp, lp, ymws)
    ecl[jflw,1] = Eoft_clover(u, gp, lp, ymws)

    if flwint.add_zth
        println(flog, "## integrating zeuthen flow equations")
    else
        println(flog, "## integrating wilson  flow equations")
    end

    ns = 0
    t  = 0.0
    for k in 2:length(flwtime)
        if (k-2)%div(length(flwtime),10) == 0
            @printf(flog, "     flow   t=%8.4f:  %20.12e \n",
                    t, t^2*epl[jflw,k-1])
        end

        if isapprox(flwtime[k]-t, flwint.eps) # measure every eps
            flw(u, wflwint, 1, gp, lp, ymws)
            ns = ns + 1
        else                                  # adapt. integration
            nstep, eps = flw_adapt(u, flwint, flwtime[k]-t, gp, lp, ymws)
            ns = ns + nstep
        end
        epl[jflw,k] = Eoft_plaq(u, gp, lp, ymws)
        ecl[jflw,k] = Eoft_clover(u, gp, lp, ymws)

        t = flwtime[k]
    end

    if flwint.add_zth
        println(flog, "## zeuthen flow equations integrated in $ns steps")
    else
        println(flog, "## wilson  flow equations integrated in $ns steps")
    end

    return nothing
end


################ SIMULATION #################

parsed_args = parse_commandline()
infile = parsed_args["i"]

# Set group and precision
GRP  = SU2
ALG  = SU2alg
SCL  = SU2fund
PREC = Float64
NSC = 2
println("Precision: ", PREC)

gp, sp, lp, int, wflwint, zflwint, nth, niter, hmc_dt, nsteps, dtr_chk, cnfg_dir, flow_bool, flw_steps, flw_s, flw_times, flw_iters, flw_dtr, flw_dt, flwtime, flw_type, corr, sus, sdt, sss, srs, run_dir_name = read_options(infile)

#Gauge
println("Gauge parameters: ", gp)
beta = gp.beta
println("Allocating YM workspace")
ymws = YMworkspace(GRP, PREC, lp)

println("Allocating gauge field")
U = vector_field(GRP{PREC}, lp)
fill!(U, one(GRP{PREC}))
println("Time to take the configuration to memory: ")
@time Ucpu = Array(U)

#Scalar
println("Scalar parameters: ", sp)
println("Allocating Scalar workspace")
sws  = ScalarWorkspace(PREC, NSC, lp)
# Scalar fields φ=0
println("Allocating scalar field")
# Phi includes all scalar fields
Phi = nscalar_field(SCL{PREC}, NSC, lp)
# starting value
fill!(Phi, zero(SCL{PREC}))
sclr_s = "_k1$(sp.kap[1])_k2$(sp.kap[2])_etas$(sp.eta[1])_$(sp.eta[2])_mu$(sp.muh)_xis$(sp.xi[1])_$(sp.xi[2])_$(sp.xi[3])_$(sp.xi[4])_$(sp.xi[5])"

#MD integrator
println(int)

#Observables
pl      = Vector{Float64}(undef, niter+nth)
rho1    = Vector{Float64}(undef, niter+nth)
rho2    = Vector{Float64}(undef, niter+nth)
rho12    = Vector{Float64}(undef, niter+nth)
Lphi1   = Vector{Float64}(undef, niter+nth)
Lphi2   = Vector{Float64}(undef, niter+nth)
Lphi12   = Vector{Float64}(undef, niter+nth)
Lalp1   = Vector{Float64}(undef, niter+nth)
Lalp2   = Vector{Float64}(undef, niter+nth)
Lalp12   = Vector{Float64}(undef, niter+nth)
#hmc
dh      = Vector{Float64}(undef, niter+nth)
acc     = Vector{Bool}(undef, niter+nth)
#correlators
#Higgs - 2 fields + mixed interpolator, 4 i*pauli matrices (4th is identity), time, niter
h2      = Array{Float64,4}(undef,NSC+1, 4, lp.iL[end], niter)
#W-Boson - 2 fields + mixed interpolator; 3 spatial directions; 4 i*pauli matrices (4th is identity); time; niter
w1      = Array{Float64,5}(undef,NSC+1,3,4, lp.iL[end], niter)

# FLOW PARAMETERS
Uflw = vector_field(GRP{PREC}, lp)
gfinfo=""
flwint = wflwint #use zflwint for zeuthen flow
if flow_bool == 0
    gfinfo = ""
elseif flow_bool ==1
    gfinfo = string("_GF_stps",flw_steps,"_s",flw_s,"_int",flw_dtr,"_dt",flw_dt)
elseif flow_bool ==2
    str_times = "times_"
    str_iters = "iters_"
    for n in 1:length(flw_times)
        global str_times *= string(flw_times[n])*"_"
        global str_iters *= string(flw_iters[n])*"_"
    end
    gfinfo = string("_GF_",str_times,"_",str_iters,"_int",flw_dtr,"_dt",flw_dt)
    println(flwint)
end

flw_iter  = div(niter,flw_dtr) #total number of flow measurements
nr_flwtimes = length(flwtime) #number of flow time measurements
E       = Array{Float64, 2}(undef, flw_iter, nr_flwtimes)
Ecl     = Array{Float64, 2}(undef, flw_iter, nr_flwtimes)
dE      = Array{Float64, 2}(undef, flw_iter, nr_flwtimes)

# Correlation function - Smearing
smear = smr{PREC}(sus, sdt, sss, srs)
if (sus == 0 && sss ==0 ) || corr == 0
    sm = false
    sminfo = ""
    if corr ==0
        sminfo = "nocorr"
    end
else
    sm = true
    sminfo = string("_SM_us_",sus,"_",sdt,"_ss_",sss,"_",srs)
end
Phismear = nscalar_field(SCL{PREC}, NSC, lp)

# Lattice dimensions
global dm = ""
for i in 1:lp.ndim-1
    global dm *= string(lp.iL[i])*"x"
end
dm *= string(lp.iL[end])



#################################### SIMULATION ###################################

for sim in 1:1

    filename = string(run_dir_name,NSC,"_",dm,"_beta",beta,sclr_s,"_niter", niter,"_eps",hmc_dt,"_nsteps", nsteps,gfinfo,sminfo,".bdio")
    fb = BDIO_open(filename, "d",
                "$(NSC) Scalar Higgs Simulations - Correlation functions")
    #log file
    flog = open(filename*".log", "w+")

    # Simulation param
    iv = [nth, niter, NSC, nr_flwtimes, flw_dtr, flw_iter, sus, sss]
    BDIO_start_record!(fb, BDIO_BIN_INT64LE, 1, true)
    BDIO_write!(fb, iv)
    BDIO_write_hash!(fb)

    BDIO_start_record!(fb, BDIO_BIN_F64LE, 2, true)
    s_parms = [beta, sp.kap[1], sp.kap[2],
               sp.eta[1], sp.eta[2], sp.muh,
               sp.xi[1], sp.xi[2], sp.xi[3], sp.xi[4], sp.xi[5],
               flw_dt, sdt, srs]
    BDIO_write!(fb, s_parms)
    BDIO_write_hash!(fb)

    k = 0
    for i in 1:1
        HMC!(U,Phi, int,lp, gp, sp, ymws, sws; noacc=true)
        # Thermalization
        for j in 1:nth
            k += 1
            dh[k], acc[k] = HMC!(U,Phi,int,lp, gp, sp, ymws, sws)
            pl[k]  = plaquette(U,lp, gp, ymws)
            # observables φ1
            rho1[k],Lphi1[k],Lalp1[k] = scalar_obs(U, Phi, 1, sp, lp, ymws)
            rho2[k],Lphi2[k],Lalp2[k] = scalar_obs(U, Phi, 2, sp, lp, ymws)
            rho12[k],Lphi12[k],Lalp12[k] = scalar_obs(U, Phi, 1, 2, sp, lp, ymws)
            # @printf(flog, "  THM %d/%d (beta: %4.3f):   %s   %6.2e    %20.12e    %20.12e    %20.12e    %20.12e\n",
            #     j, nth, beta, acc[k] ? "true " : "false", dh[k],
            #     pl[k], rho1[k], Lphi1[k], Lalp1[k])
            @printf(flog, "  THM %d/%d (beta: %4.3f):   %s   %6.2e    %20.12e    %20.12e    %20.12e    %20.12e  %20.12e    %20.12e    %20.12e\n",
                j, nth, beta, acc[k] ? "true " : "false", dh[k],
                pl[k], rho1[k], rho2[k], Lphi1[k], Lphi2[k], Lalp1[k], Lalp2[k])


        end
        println(flog, " ")

        # MC chain
        kflw=0
        jflw=0
        chk=0 #checkpoints - save config
        ncfg = 1
        for j in 1:niter
            k += 1
            chk += 1
            dh[k], acc[k] = HMC!(U,Phi, int,lp, gp, sp, ymws, sws)
            pl[k]  = plaquette(U,lp, gp, ymws)
            rho1[k],Lphi1[k],Lalp1[k] = scalar_obs(U, Phi, 1, sp, lp, ymws)
            rho2[k],Lphi2[k],Lalp2[k] = scalar_obs(U, Phi, 2, sp, lp, ymws)
            rho12[k],Lphi12[k],Lalp12[k] = scalar_obs(U, Phi,1, 2, sp, lp, ymws)
            if corr == 1
                if sm
                    # save field U from GPU to cpu
                    Ucpu .= Array(U)
                    Phismear .= Phi
                    # CUDA.@allowscalar print(Phismear[1,1,1],"\t")
                    smearing(U, Phismear, smear, sp, lp, ymws, gp, sws)
                    # CUDA.@allowscalar print(Phismear[1,1,1],"\n")
                    h2[1,:,:,j], w1[1,:,:,:,j]      = scalar_corr(U, Phismear, 1, 1, sp, lp, ymws, gp, sws)
                    h2[2,:,:,j], w1[2,:,:,:,j]      = scalar_corr(U, Phismear, 2, 2, sp, lp, ymws, gp, sws)
                    h2[3,:,:,j], w1[3,:,:,:,j]      = scalar_corr(U, Phismear, 1, 2, sp, lp, ymws, gp, sws)
                    # Pass the saved field again to GPU
                    U .= CuArray(Ucpu)
                else
                    h2[1,:,:,j], w1[1,:,:,:,j]      = scalar_corr(U, Phis, 1, 1, sp, lp, ymws, gp, sws)
                    h2[2,:,:,j], w1[2,:,:,:,j]      = scalar_corr(U, Phis, 2, 2, sp, lp, ymws, gp, sws)
                    h2[3,:,:,j], w1[3,:,:,:,j]      = scalar_corr(U, Phis, 1, 2, sp, lp, ymws, gp, sws)
                end
            end

            @printf(flog, "  MSM %d/%d (beta: %4.3f):   %s   %6.2e    %20.12e    %20.12e    %20.12e    %20.12e\n",
                j, niter, beta, acc[k] ? "true " : "false", dh[k],
                pl[k], rho1[k], Lphi1[k], Lalp1[k])

            if chk == dtr_chk
                cfg_name = string(cnfg_dir,"/h2_beta",beta,sclr_s,"_n",string(ncfg),".cnfg.bdio")
                save_cnfg(cfg_name, U, lp, gp, run=run_dir_name)
                ncfg += 1
                chk = 0
            end


            # flow measurement with adapt. step size intgrator
            # flow every 'flw_dtr'
            kflw += 1
            if (kflw == flw_dtr)
                if flow_bool != 0
                    print(flog, "\n\t## START Flow:")
                    jflw+=1
                    # save field U from GPU to cpu
                    Ucpu .= Array(U)
                    if flow_bool ==1
                        measure_flow!(E, Ecl, U, flwtime, flw_steps, flw_s, flwint, jflw, gp, lp, ymws, flog)
                    elseif flow_bool==2
                        measure_flow!(E, Ecl, U, flwtime, flwint, jflw, gp, lp, ymws, flog)
                    end
                    # Pass the saved field again to GPU
                    U .= CuArray(Ucpu)
                    kflw = 0
                    println(flog, "\n\t## END flow measurements")
                end
            end
        end
    end

    ################################### SAVE TO FILE ##################################
    # Write to BDIO
    # Plaquette
    BDIO_start_record!(fb, BDIO_BIN_F64LE, 3, true)
    BDIO_write!(fb, pl[begin:end])

    # Rho1
    BDIO_start_record!(fb, BDIO_BIN_F64LE, 4, true)
    BDIO_write!(fb, rho1[begin:end])
    BDIO_write!(fb, rho2[begin:end])
    BDIO_write!(fb, rho12[begin:end])
    BDIO_write_hash!(fb)

    # Lphi1
    BDIO_start_record!(fb, BDIO_BIN_F64LE, 5, true)
    BDIO_write!(fb, Lphi1[begin:end])
    BDIO_write!(fb, Lphi2[begin:end])
    BDIO_write!(fb, Lphi12[begin:end])
    BDIO_write_hash!(fb)

    # Lalp1
    BDIO_start_record!(fb, BDIO_BIN_F64LE, 6, true)
    BDIO_write!(fb, Lalp1[begin:end])
    BDIO_write!(fb, Lalp2[begin:end])
    BDIO_write!(fb, Lalp12[begin:end])
    BDIO_write_hash!(fb)

    #Energy conservation
    BDIO_start_record!(fb, BDIO_BIN_F64LE, 8, true)
    BDIO_write!(fb, dh[begin:end])
    BDIO_write_hash!(fb)

    # FLOW OBSERVABLES
    if flow_bool == 1 || flow_bool == 2
        # Flow time
        BDIO_start_record!(fb, BDIO_BIN_F64LE, 9, true)
        BDIO_write!(fb, flwtime)
        BDIO_write_hash!(fb)
        # E Plaquette
        BDIO_start_record!(fb, BDIO_BIN_F64LE, 10, true)
        BDIO_write!(fb, E)
        BDIO_write_hash!(fb)
        # E Clover
        BDIO_start_record!(fb, BDIO_BIN_F64LE, 11, true)
        BDIO_write!(fb, Ecl)
        BDIO_write_hash!(fb)
    end

    #CORRELATION FUNCTIONS
    #Higgs interpolator
    BDIO_start_record!(fb, BDIO_BIN_F64LE, 12, true)
    for k in 1:niter
        for i in 1:4
            BDIO_write!(fb, h2[1,i,:,k]) #\phi_1^\dagger \phi_1 \sigma_a, a=1,2,3,4
        end
    end
    for k in 1:niter
        for i in 1:4
            BDIO_write!(fb, h2[2,i,:,k]) #\phi_2^\dagger \phi_2
        end
    end
    for k in 1:niter
        for i in 1:4
            BDIO_write!(fb, h2[3,i,:,k]) #\phi_1^\dagger \phi_2
        end
    end

    #W-boson interpolator
    BDIO_start_record!(fb, BDIO_BIN_F64LE, 13, true)
    for k in 1:niter
        for mu in 1:3
            for i in 1:4
                BDIO_write!(fb, (w1[1,mu,i,:,k])) #\phi_1^\dagger U_\mu \phi_1 \sigma_a
            end
        end
    end
    for k in 1:niter
        for mu in 1:3
            for i in 1:4
                BDIO_write!(fb, (w1[2,mu,i,:,k])) #\phi_2^\dagger U_\mu \phi_2 \sigma_a
            end
        end
    end
    for k in 1:niter
        for mu in 1:3
            for i in 1:4
                BDIO_write!(fb, (w1[3,mu,i,:,k])) #\phi_1^\dagger U_\mu \phi_1 \sigma_a
            end
        end
    end
    BDIO_write_hash!(fb)

    BDIO_close!(fb)

    println(flog, "\n\n")

    println(flog, "## Timming results")
    print_timer(flog, linechars = :ascii)


    println(flog, "## END")
end
