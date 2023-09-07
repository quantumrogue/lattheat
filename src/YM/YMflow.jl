###
### "THE BEER-WARE LICENSE":
### Alberto Ramos wrote this file. As long as you retain this 
### notice you can do whatever you want with this stuff. If we meet some 
### day, and you think this stuff is worth it, you can buy me a beer in 
### return. <alberto.ramos@cern.ch>
###
### file:    YMflow.jl
### created: Sat Sep 25 08:37:14 2021
###                               


struct FlowIntr{N,T}
    r::T
    e0::NTuple{N,T}
    e1::NTuple{N,T}

    add_zth::Bool
    c0::T

    eps::T
    tol::T
    eps_ini::T
    max_eps::T
    sft_fac::T
end

# pre-defined integrators
wfl_euler(::Type{T}, eps::T, tol::T) where T = FlowIntr{0,T}(one(T),(),(),false,one(T),eps,tol,one(T)/200,one(T)/10,9/10)
zfl_euler(::Type{T}, eps::T, tol::T) where T = FlowIntr{0,T}(one(T),(),(),true, (one(T)*5)/3,eps,tol,one(T)/200,one(T)/10,9/10)
wfl_rk2(::Type{T}, eps::T, tol::T)   where T = FlowIntr{1,T}(one(T)/2,(-one(T)/2,),(one(T),),false,one(T),eps,tol,one(T)/200,one(T)/10,9/10)
zfl_rk2(::Type{T}, eps::T, tol::T)   where T = FlowIntr{1,T}(one(T)/2,(-one(T)/2,),(one(T),),true, (one(T)*5)/3,eps,tol,one(T)/200,one(T)/10,9/10)
wfl_rk3(::Type{T}, eps::T, tol::T)   where T = FlowIntr{2,T}(one(T)/4,(-17/36,-one(T)),(8/9,3/4),false,one(T),eps,tol,one(T)/200,one(T)/10,9/10)
zfl_rk3(::Type{T}, eps::T, tol::T)   where T = FlowIntr{2,T}(one(T)/4,(-17/36,-one(T)),(8/9,3/4),true, (one(T)*5)/3,eps,tol,one(T)/200,one(T)/10,9/10)

function Base.show(io::IO, int::FlowIntr{N,T}) where {N,T}

    if (abs(int.c0-1) < 1.0E-10)
        println(io, "WILSON flow integrator")
    elseif (abs(int.c0-5/3) < 1.0E-10) && int.add_zth
        println(io, "ZEUTHEN flow integrator")
    elseif (abs(int.c0-5/3) < 1.0E-10) && !int.add_zth
        println(io, "SYMANZIK flow integrator")
    else
        println(io, "CUSTOM flow integrator")
        if int.add_zth
            println(io, "  - ", int.c0, " (with zeuthen term)")
        else
            println(io, "  - ", int.c0)
        end
    end

    if N == 0
        println(io, " * Euler schem3")
    elseif N == 1
        println(io, " * One stage scheme. Coefficients3")
        println(io, "    stg 1: ", int.e0[1], " ", int.e1[1])
    elseif N == 2
        println(io, " * Two stage scheme. Coefficients:")
        println(io, "    stg 1: ", int.e0[1], " ", int.e1[1])
        println(io, "    stg 2: ", int.e0[2], " ", int.e1[2])
    end

    println(io, " * Fixed step size parameters: eps = ", int.eps)
    println(io, " * Adaptive step size parameters: tol = ", int.tol)
    println(io, "    - max eps:      ", int.max_eps)
    println(io, "    - initial eps:  ", int.eps_ini)
    println(io, "    - safety scale: ", int.sft_fac)

    return nothing
end

"""
    function add_zth_term(ymws::YMworkspace, U, lp)

Assuming that the gauge improved (LW) force is in ymws.frc1, this routine
adds the "Zeuthen term" and returns the full zeuthen force in ymws.frc1
"""
function add_zth_term(ymws::YMworkspace, U, lp)

    CUDA.@sync begin
        CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_add_zth!(ymws.frc1,ymws.frc2,U,lp)
    end
    ymws.frc1 .= ymws.frc2

    return nothing
end

function krnl_add_zth!(frc, frc2::AbstractArray{TA}, U::AbstractArray{TG}, lp::SpaceParm{N,M,B,D}) where {TA,TG,N,M,B,D}

    b, r = CUDA.threadIdx().x, CUDA.blockIdx().x
    it = point_time((b, r), lp)

    SFBC = ((B == BC_SF_AFWB) || (B == BC_SF_ORBI) )

    @inbounds for id in 1:N

        bu, ru = up((b,r), id, lp)
        bd, rd = dw((b,r), id, lp)

        X = frc[bu,id,ru]

        Y  = frc[bd,id,rd]
        Ud = U[bd,id,rd]

        if SFBC
            if (it > 1) && (it < lp.iL[end])
                frc2[b,id,r] = (5/6)*F[b,id,r] + (1/6)*(projalg(Ud\Y*Ud) +
                                                     projalg(U[b,id,r]*X/U[b,id,r]))
            elseif (it == lp.iL[end]) && (id < N)
                frc2[b,id,r] = (5/6)*F[b,id,r] + (1/6)*(projalg(Ud\Y*Ud) +
                                                     projalg(U[b,id,r]*X/U[b,id,r]))
            end
        else
            frc2[b,id,r] = (5/6)*F[b,id,r] + (1/6)*(projalg(Ud\Y*Ud) +
                                                 projalg(U[b,id,r]*X/U[b,id,r]))
        end
    end

    return nothing
end


function flw(U, int::FlowIntr{NI,T}, ns::Int64, eps, gp::GaugeParm, lp::SpaceParm, ymws::YMworkspace) where {NI,T}
    @timeit "Integrating flow equations" begin
        for i in 1:ns
            force_gauge(ymws, U, int.c0, 1, gp, lp)
            if int.add_zth
                add_zth_term(ymws::YMworkspace, U, lp)
            end
            ymws.mom .= ymws.frc1
            U .= expm.(U, ymws.mom, 2*eps*int.r)

            for k in 1:NI
                force_gauge(ymws, U, int.c0, 1, gp, lp)
                if int.add_zth
                    add_zth_term(ymws::YMworkspace, U, lp)
                end
                ymws.mom .= int.e0[k].*ymws.mom .+ int.e1[k].*ymws.frc1
                U .= expm.(U, ymws.mom, 2*eps)
            end
        end
    end

    return nothing
end
flw(U, int::FlowIntr{NI,T}, ns::Int64, gp::GaugeParm, lp::SpaceParm, ymws::YMworkspace) where {NI,T} = flw(U, int, ns, int.eps, gp, lp, ymws)


##
# Adaptive step size integrators
##

function flw_adapt(U, int::FlowIntr{NI,T}, tend::T, epsini::T, gp::GaugeParm, lp::SpaceParm, ymws::YMworkspace) where {NI,T}

    eps = int.eps_ini
    dt  = tend
    nstp = 0
    while true
        ns = convert(Int64, floor(dt/eps))
        if ns > 10
            flw(U, int, 9, eps, gp, lp, ymws)
            ymws.U1 .= U
            flw(U, int, 2, eps/2, gp, lp, ymws)
            flw(ymws.U1, int, 1, eps, gp, lp, ymws)

            dt = dt - 10*eps
            nstp = nstp + 10

            # adjust step size
            ymws.U1 .= ymws.U1 ./ U
            maxd = CUDA.mapreduce(dev_one, max, ymws.U1, init=zero(tend))
            eps  = min(int.max_eps, 2*eps, int.sft_fac*eps*(int.tol/maxd)^(one(tend)/3))

        else
            flw(U, int, ns, eps, gp, lp, ymws)
            dt = dt - ns*eps

            flw(U, int, 1, dt, gp, lp, ymws)
            dt = zero(tend)

            nstp = nstp + ns + 1
        end

        if dt == zero(tend)
            break
        end
    end

    return nstp, eps
end
flw_adapt(U, int::FlowIntr{NI,T}, tend::T, gp::GaugeParm, lp::SpaceParm, ymws::YMworkspace) where {NI,T} = flw_adapt(U, int, tend, int.eps_ini, gp, lp, ymws)


######################################### Plaquette ##########################################

"""
    function Eoft_plaq([Eslc,] U, gp::GaugeParm, lp::SpaceParm, ymws::YMworkspace)

Measure the action density `E(t)` using the plaquette discretization. If the argument `Eslc`
the contribution for each Euclidean time slice and plane are returned.
"""
function Eoft_plaq(Eslc, U, gp::GaugeParm{T,G,NN}, lp::SpaceParm{N,M,B,D}, ymws::YMworkspace) where {T,G,NN,N,M,B,D}

    @timeit "E(t) plaquette measurement" begin

        ztw = ztwist(gp, lp)
        SFBC = ((B == BC_SF_AFWB) || (B == BC_SF_ORBI) )

        tp = ntuple(i->i, N-1)
        V3 = prod(lp.iL[1:end-1])

        fill!(Eslc,zero(T))
        Etmp = zeros(T,lp.iL[end])
        for ipl in 1:M
            fill!(Etmp, zero(T))
            CUDA.@sync begin
                CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_plaq_pln!(ymws.cm, U, gp.Ubnd, ztw[ipl], ipl, lp)
            end

            Etmp .=  (gp.ng .- reshape(Array(CUDA.mapreduce(real, +, ymws.cm;dims=tp)),lp.iL[end]) ./ V3 )
            if ipl < N
                for it in 2:lp.iL[end]
                    Eslc[it,ipl] = Etmp[it] + Etmp[it-1]
                end
                if !SFBC
                    Eslc[1,ipl] = Etmp[1] + Etmp[end]
                end
            else
                for it in 1:lp.iL[end]
                    Eslc[it,ipl] = 2*Etmp[it]
                end
            end
        end

    end


    return sum(Eslc)/lp.iL[end]
end

Eoft_plaq(U, gp::GaugeParm{T,G,NN}, lp::SpaceParm{N,M,B,D}, ymws::YMworkspace) where {T,G,NN,N,M,B,D} = Eoft_plaq(zeros(T,lp.iL[end],M), U, gp, lp, ymws)

function krnl_plaq_pln!(plx, U::AbstractArray{T}, Ubnd, ztw, ipl, lp::SpaceParm{N,M,B,D}) where {T,N,M,B,D}

    b, r = CUDA.threadIdx().x, CUDA.blockIdx().x

    id1, id2 = lp.plidx[ipl]
    SFBC = ((B == BC_SF_AFWB) || (B == BC_SF_ORBI)) && (id1 == lp.iL[end])

    bu1, ru1 = up((b, r), id1, lp)
    bu2, ru2 = up((b, r), id2, lp)

    if SFBC && (ru1 != r)
        gt = Ubnd[id2]
    else
        gt = U[bu1,id2,ru1]
    end

    I = point_coord((b,r), lp)
    plx[I] = ztw*tr(U[b,id1,r]*gt / (U[b,id2,r]*U[bu2,id1,ru2]))

    return nothing
end


######################################### Plaquette Derivative ##########################################

"""
    Measure time derivative of the action density 'E(t)' by its explicit functional form
    for a single plane lp.plidx[ipl]
"""

function dEoft_plaq(Eslc, U, gp::GaugeParm{T,G,NN}, lp::SpaceParm{N,M,B,D}, ymws::YMworkspace,int::FlowIntr{NI,T}) where {T,G,NN,N,M,B,D,NI}

    @timeit "dE(t) plaquette measurement" begin

        force_gauge(ymws, U, int.c0, 1, gp, lp)
        ztw = ztwist(gp, lp)
        SFBC = ((B == BC_SF_AFWB) || (B == BC_SF_ORBI) )

        tp = ntuple(i->i, N-1) # (1,2,3,...,N-1)
        V3 = prod(lp.iL[1:end-1]) # spatial volume

        fill!(Eslc,zero(T))
        Etmp = zeros(T,lp.iL[end])

        for ipl in 1:M # lattice planes
            fill!(Etmp, zero(T))
            CUDA.@sync begin
                CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_dEoft_plaq_pln!(ymws.cm, ymws.frc1, U, gp.Ubnd, ztw[ipl], ipl, lp)
            end

            # actions: - take real of each element of ymws.cm
            #          - sum its elements in the spatial directions
            #          - result in 1x1x1xLt array
            #          - reshape - 1D array with time elements
            Etmp .=  reshape(Array(CUDA.mapreduce(real, +, ymws.cm; dims=tp)),lp.iL[end]) ./ (V3)

            if ipl < N # first N-1 planes include time direction
                for it in 2:lp.iL[end]
                    Eslc[it,ipl] = Etmp[it] + Etmp[it-1]
                end
                if !SFBC
                    Eslc[1,ipl] = Etmp[1] + Etmp[end]
                end
            else
                for it in 1:lp.iL[end]
                    Eslc[it,ipl] = 2*Etmp[it]
                end
            end
        end

    end

    return -40*sum(Eslc)/lp.iL[end]
end
dEoft_plaq(U, gp::GaugeParm{T,G,NN}, lp::SpaceParm{N,M,B,D}, ymws::YMworkspace,int::FlowIntr{NI,T}) where {T,G,NN,N,M,B,D,NI} = dEoft_plaq( zeros(T,lp.iL[end],M), U, gp, lp, ymws,int)

"""
    Kernel function to compute flow-time derivative of E(t) for a specific plane
"""
function krnl_dEoft_plaq_pln!(plx, frc, U::AbstractArray{T}, Ubnd, ztw, ipl, lp::SpaceParm{N,M,B,D}) where {T,N,M,B,D}
    # plx = scalar_field_point(...) = CuArray{T,N}(undef, (16,16,16,16))
    # U   = vector_field(group) = (nr threads in the block)x(dimension of the lattice)x(nr of blocks)
    # Ubnd = gauge field boundary

    # coordinates of the point
    b, r = CUDA.threadIdx().x, CUDA.blockIdx().x

    # directions of the plane
    id1, id2 = lp.plidx[ipl]
    # boundary conditions
    SFBC = ((B == BC_SF_AFWB) || (B == BC_SF_ORBI)) && (id1 == lp.iL[end])

    # forward points next direction
    bu1, ru1 = up((b, r), id1, lp)
    bu2, ru2 = up((b, r), id2, lp)

    # boundary fields
    if SFBC && (ru1 != r)
        gt = Ubnd[id2]
    else
        gt = U[bu1,id2,ru1]
    end

    I = point_coord((b,r), lp)
    plx[I] = ztw*( tr( frc[b,id1,r]*U[b,id1,r]*gt / (U[b,id2,r]*U[bu2,id1,ru2])) +
                   tr( U[b,id1,r]*frc[bu1,id2,ru1]*gt / (U[b,id2,r]*U[bu2,id1,ru2])) +
                   -tr( frc[bu2,id1,ru2]/U[b,id2,r]*U[b,id1,r]*gt/U[bu2,id1,ru2]) +
                   -tr( U[b,id1,r]\frc[b,id2,r]*U[b,id1,r]*gt/U[bu2,id1,ru2]))
    return nothing
end


######################################### Top. Charge ##########################################

"""
    Qtop([Qslc,] U, lp, ymws)

Measure the topological charge `Q` of the configuration `U`. If the argument `Qslc` is present
the contribution for each Euclidean time slice are returned.
"""
function Qtop(Qslc, U, gp::GaugeParm, lp::SpaceParm{4,M,B,D}, ymws::YMworkspace) where {M,B,D}

    @timeit "Qtop measurement" begin

        ztw = ztwist(gp, lp)
        tp = (1,2,3)

        fill!(ymws.rm, zero(eltype(ymws.rm)))
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_field_tensor!(ymws.frc1, ymws.frc2, U, gp.Ubnd, 1,5, ztw[1], ztw[5], lp)
        end
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_add_qd!(ymws.rm, -, ymws.frc1, ymws.frc2, U, lp)
        end

        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_field_tensor!(ymws.frc1, ymws.frc2, U, gp.Ubnd, 2,4, ztw[2], ztw[4], lp)
        end
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_add_qd!(ymws.rm, +, ymws.frc1, ymws.frc2, U, lp)
        end

        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_field_tensor!(ymws.frc1, ymws.frc2, U, gp.Ubnd, 3,6, ztw[3], ztw[6], lp)
        end
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_add_qd!(ymws.rm, -, ymws.frc1, ymws.frc2, U, lp)
        end

        Qslc .= reshape(Array(CUDA.reduce(+, ymws.rm; dims=tp)),lp.iL[end])./(32*pi^2)
    end

    return sum(Qslc)
end
Qtop(U, gp::GaugeParm, lp::SpaceParm{4,M,D}, ymws::YMworkspace{T}) where {T,M,D} = Qtop(zeros(T,lp.iL[end],M), U, gp, lp, ymws)



######################################### Clover ##########################################

"""
    function Eoft_clover([Eslc,] U, gp::GaugeParm, lp::SpaceParm, ymws::YMworkspace)

Measure the action density `E(t)` using the clover discretization. If the argument `Eslc`
the contribution for each Euclidean time slice and plane are returned.
"""
function Eoft_clover(Eslc, U, gp::GaugeParm, lp::SpaceParm{4,M,B,D}, ymws::YMworkspace{T}) where {T,M,B,D}

    function acum(ipl1, ipl2, Etmp)

        tp = (1,2,3)
        V3 = prod(lp.iL[1:end-1])

        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_add_et!(ymws.rm, +, ymws.frc1, U, lp)
        end
        Etmp .=  reshape(Array(CUDA.reduce(+, ymws.rm;dims=tp)),lp.iL[end])/V3
        for it in 1:lp.iL[end]
            Eslc[it,ipl1] = Etmp[it]/8
        end

        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_add_et!(ymws.rm, +, ymws.frc2, U, lp)
        end
        Etmp .=  reshape(Array(CUDA.reduce(+, ymws.rm;dims=tp)),lp.iL[end])/V3
        for it in 1:lp.iL[end]
            Eslc[it,ipl2] = Etmp[it]/8
        end

        return nothing
    end


    @timeit "E(t) clover measurement" begin

        ztw = ztwist(gp, lp)
        fill!(Eslc,zero(T))
        Etmp = zeros(T,lp.iL[end])

        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_field_tensor!(ymws.frc1, ymws.frc2, U, gp.Ubnd, 1,2, ztw[1], ztw[2], lp)
        end
        acum(1,2,Etmp)


        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_field_tensor!(ymws.frc1, ymws.frc2, U, gp.Ubnd, 3,4, ztw[3], ztw[4], lp)
        end
        acum(3,4,Etmp)

        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_field_tensor!(ymws.frc1, ymws.frc2, U, gp.Ubnd, 5,6, ztw[5], ztw[6], lp)
        end
        acum(5,6,Etmp)

    end

    return sum(Eslc)/lp.iL[end]
end
Eoft_clover(U, gp::GaugeParm, lp::SpaceParm{N,M,B,D}, ymws::YMworkspace{T}) where {T,N,M,B,D} = Eoft_clover(zeros(T,lp.iL[end],M), U, gp, lp, ymws)

function krnl_add_et!(rm, op, frc1, U, lp::SpaceParm{4,M,B,D}) where {M,B,D}

    b, r = CUDA.threadIdx().x, CUDA.blockIdx().x

    X1 = (frc1[b,1,r]+frc1[b,2,r]+frc1[b,3,r]+frc1[b,4,r])

    I = point_coord((b,r), lp)
    rm[I] = dot(X1,X1)

    return nothing
end

function krnl_add_qd!(rm, op, frc1, frc2, U, lp::SpaceParm{4,M,B,D}) where {M,B,D}

    b, r = CUDA.threadIdx().x, CUDA.blockIdx().x

    I = point_coord((b,r), lp)
    rm[I] += op(dot( (frc1[b,1,r]+frc1[b,2,r]+frc1[b,3,r]+frc1[b,4,r]),
                     (frc2[b,1,r]+frc2[b,2,r]+frc2[b,3,r]+frc2[b,4,r]) ) )
    return nothing
end

function krnl_field_tensor!(frc1::AbstractArray{TA}, frc2, U::AbstractArray{T}, Ubnd, ipl1, ipl2, ztw1, ztw2, lp::SpaceParm{4,M,B,D}) where {TA,T,M,B,D}

    b, r = CUDA.threadIdx().x, CUDA.blockIdx().x
    it = point_time((b,r), lp)
    SFBC = ((B == BC_SF_AFWB) || (B == BC_SF_ORBI) )


    #First plane
    id1, id2 = lp.plidx[ipl1]

    SFBC = ((B == BC_SF_AFWB) || (B == BC_SF_ORBI) ) && (id1 == 4)

    bu1, ru1 = up((b, r), id1, lp)
    bu2, ru2 = up((b, r), id2, lp)
    bd, rd   = up((bu1, ru1), id2, lp)

    if SFBC && (it == lp.iL[end])
        gt1 = Ubnd[id2]
    else
        gt1 = U[bu1,id2,ru1]
    end

    gt2 = U[bu2,id1,ru2]

    l1 = gt1/gt2
    l2 = U[b,id2,r]\U[b,id1,r]

    if SFBC && (it == lp.iL[end])
        frc1[b,1,r]     = projalg(U[b,id1,r]*l1/U[b,id2,r])
        frc1[bu1,2,ru1] = zero(TA)
        frc1[bd,3,rd]   = zero(TA)
        frc1[bu2,4,ru2] = projalg(l2*l1)
    else
        frc1[b,1,r]     = projalg(ztw1, U[b,id1,r]*l1/U[b,id2,r])
        frc1[bu1,2,ru1] = projalg(ztw1, l1*l2)
        frc1[bd,3,rd]   = projalg(ztw1, gt2\(l2*gt1))
        frc1[bu2,4,ru2] = projalg(ztw1, l2*l1)
    end

    # Second plane
    id1, id2 = lp.plidx[ipl2]
    sync_threads()

    SFBC = ((B == BC_SF_AFWB) || (B == BC_SF_ORBI) ) && (id1 == 4)

    bu1, ru1 = up((b, r), id1, lp)
    bu2, ru2 = up((b, r), id2, lp)
    bd, rd   = up((bu1, ru1), id2, lp)

    if SFBC && (it == lp.iL[end])
        gt1 = Ubnd[id2]
    else
        gt1 = U[bu1,id2,ru1]
    end

    gt2 = U[bu2,id1,ru2]

    l1 = gt1/gt2
    l2 = U[b,id2,r]\U[b,id1,r]

    if SFBC && (it == lp.iL[end])
        frc2[b,1,r]     = projalg(U[b,id1,r]*l1/U[b,id2,r])
        frc2[bu1,2,ru1] = zero(TA)
        frc2[bd,3,rd]   = zero(TA)
        frc2[bu2,4,ru2] = projalg(l2*l1)
    else
        frc2[b,1,r]     = projalg(ztw2, U[b,id1,r]*l1/U[b,id2,r])
        frc2[bu1,2,ru1] = projalg(ztw2, l1*l2)
        frc2[bd,3,rd]   = projalg(ztw2, gt2\(l2*gt1))
        frc2[bu2,4,ru2] = projalg(ztw2, l2*l1)
    end

    return nothing
end

######################################### Clover Derivative ##########################################

function krnl_add_det!(rm, frc1, frc2, U, lp::SpaceParm{4,M,B,D}) where {M,B,D}

    b, r = CUDA.threadIdx().x, CUDA.blockIdx().x

    I = point_coord((b,r), lp)
    rm[I] = dot( (frc1[b,1,r]+frc1[b,2,r]+frc1[b,3,r]+frc1[b,4,r]),
                     (frc2[b,1,r]+frc2[b,2,r]+frc2[b,3,r]+frc2[b,4,r]) )
    return nothing
end

#     function Eoft_clover([Eslc,] U, gp::GaugeParm, lp::SpaceParm, ymws::YMworkspace)

# Measure the action density `E(t)` using the clover discretization. If the argument `Eslc`
# the contribution for each Euclidean time slice and plane are returned.

# The Clover definition includes computing `G_\mu\nu(x)` which is an element of the Algebra.
# This is done through the projection to the hermitian traceless
# M -> (M-M*)/2 - Tr(M-M*)/(2N)

#     -> krnl_field_tensor!(...) computes these algebra elements with the projection
#     -> krnl_add_et!() computes the contraction in the direction indices

function dEoft_clover(Eslc, U, gp::GaugeParm, lp::SpaceParm{4,M,B,D}, ymws::YMworkspace{T},int::FlowIntr{NI,T}) where {T,M,B,D,NI}

    # \sum_μν [G_μν G_μν] for a single plane
    # krnl_add_qd!(...) sums the contribution of the four clover plaquettes
    function acum(ipl, Etmp)

        tp = (1,2,3)
        V3 = prod(lp.iL[1:end-1])

        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_add_det!(ymws.rm, ymws.frc1, ymws.frc2,U,lp)
        end
        Etmp .=  reshape(Array(CUDA.reduce(+, ymws.rm;dims=tp)),lp.iL[end])/V3
        for it in 1:lp.iL[end]
            Eslc[it,ipl] = Etmp[it]/8.0
        end

        return nothing
    end

    @timeit "dE(t) clover measurement" begin

        ztw = ztwist(gp, lp)
        fill!(Eslc,zero(T))
        Etmp = zeros(T,lp.iL[end])

        for pln in lp.npls

            # fill ymws.frc1 with force field for the current configuration
            force_gauge(ymws, U, int.c0, 1, gp, lp)

            CUDA.@sync begin
                CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_field_dtensor!(ymws.frc1, ymws.frc2, U, gp.Ubnd, pln, ztw[pln], lp)
            end
            acum(pln,Etmp)

        end
    end

    return 60*sum(Eslc)/lp.iL[end]
end
dEoft_clover(U, gp::GaugeParm, lp::SpaceParm{4,M,B,D}, ymws::YMworkspace{T},int::FlowIntr{NI,T}) where {T,M,B,D,NI} = dEoft_clover(zeros(T,lp.iL[end],M), U, gp, lp, ymws, int)


#     Time derivative - Clover definition of the plaquette for one plane
#         -> Contributions P,Q,R,S saved in the 4 directions of frc1
#         -> Time derivative of P,Q,R,S saved in the 4 directions of frc2
#     4 dimensions
#     Each plaquette P_\mu\nu(x) = U_μ(x) U_ν(x+μ) U*_μ(x+ν) U*_ν(x) enters the definition of the clover in 4 different points:
#         -> As P for x
#         -> As S for x+μ
#         -> As R for x+μ+ν
#         -> As Q for x+ν
#     Thus only 1 plaquette is computed for each lattice point


function krnl_field_dtensor!(frc1::AbstractArray{TA}, frc2::AbstractArray{TA}, U::AbstractArray{T}, Ubnd, ipl, ztw, lp::SpaceParm{4,M,B,D}) where {TA,T,M,B,D}

    b, r = CUDA.threadIdx().x, CUDA.blockIdx().x
    it = point_time((b,r), lp)
    SFBC = ((B == BC_SF_AFWB) || (B == BC_SF_ORBI) )

    id1, id2 = lp.plidx[ipl]

    # TIME DERIVATIVE OF FIELD TENSOR
    # Force is stored originally in frc1

    SFBC = ((B == BC_SF_AFWB) || (B == BC_SF_ORBI) ) && (id1 == 4)

    # For plane id1, id2 = μ,ν
    bu1, ru1 = up((b, r), id1, lp) # x+μ
    bu2, ru2 = up((b, r), id2, lp) # x+ν
    bd, rd   = up((bu1, ru1), id2, lp) # x+μ+ν
    # gt1 = U_ν(x+μ)
    if SFBC && (it == lp.iL[end])
        gt1 = Ubnd[id2]
    else
        gt1 = U[bu1,id2,ru1]
    end

    gt2 = U[bu2,id1,ru2]
    # gt2 = U_μ(x+ν)

    l1 = gt1/gt2 # U_ν(x+μ) U*_μ(x+ν)
    l2 = U[b,id2,r]\U[b,id1,r] # U*_ν(x) U_μ(x)

    if SFBC && (it == lp.iL[end])
        frc2[b,1,r]     = projalg( frc1[b,id1,r]*U[b,id1,r]*l1/U[b,id2,r] +
                                   U[b,id1,r]*frc1[bu1,id2,ru1]*l1/U[b,id2,r] +
                                   -1.0*( (U[b,id1,r]*l1)*(frc1[bu2,id1,ru2]/U[b,id2,r]) ) +
                                   -1.0*( (U[b,id1,r]*l1/U[b,id2,r])*frc1[b,id2,r] ) )
        frc2[bu1,2,ru1] = zero(TA)
        frc2[bd,3,rd]   = zero(TA)
        frc2[bu2,4,ru2] = projalg( -1.0*( (U[b,id2,r]\frc1[b,id2,r])*U[b,id1,r]*l1 ) +
                                   U[b,id2,r]\frc1[b,id1,r]*U[b,id1,r]*l1 +
                                   l2*frc1[bu1,id2,ru1]*l1 +
                                   -1.0*( l2*l1*frc1[bu2,id1,ru2] ) )
    else
        #P for x
        frc2[b,1,r]     = projalg(ztw,   frc1[b,id1,r]*U[b,id1,r]*l1/U[b,id2,r] +
                                          U[b,id1,r]*frc1[bu1,id2,ru1]*l1/U[b,id2,r] +
                                         -1.0*( (U[b,id1,r]*l1)*(frc1[bu2,id1,ru2]/U[b,id2,r]) ) +
                                         -1.0*( (U[b,id1,r]*l1/U[b,id2,r])*frc1[b,id2,r] ) )
        #S for x+μ
        # U_ν(x+μ) U*_μ(x+ν) U*_ν(x) U_μ(x)
        frc2[bu1,2,ru1] = projalg(ztw,  frc1[bu1,id2,ru1]*l1*l2 +
                                        -1.0*( l1*frc1[bu2,id1,ru2]*l2 ) +
                                        -1.0*( (l1/U[b,id2,r])*(frc1[b,id2,r]*U[b,id1,r]) ) +
                                               (l1/U[b,id2,r])*(frc1[b,id1,r]*U[b,id1,r]) )
        #R for x+μ+ν
        # U*_μ(x+ν) U*_ν(x) U_μ(x) U_ν(x+μ)
        frc2[bd,3,rd]   = projalg(ztw,  -1.0*( (gt2\frc1[bu2,id1,ru2])*(l2*gt1) ) +
                                        -1.0*( (gt2*U[b,id2,r])\frc1[b,id2,r]*U[b,id1,r]*U[bu1,id2,ru1] ) +
                                               (gt2*U[b,id2,r])\frc1[b,id1,r]*U[b,id1,r]*U[bu1,id2,ru1] +
                                               (gt2*U[b,id2,r])\U[b,id1,r]*frc1[bu1,id2,ru1]*U[bu1,id2,ru1])
        #Q for x+ν
        # U*_ν(x) U_μ(x) U_ν(x+μ) U*_μ(x+ν)
        frc2[bu2,4,ru2] = projalg(ztw,  -1.0*( (U[b,id2,r]\frc1[b,id2,r])*U[b,id1,r]*l1 ) +
                                          U[b,id2,r]\frc1[b,id1,r]*U[b,id1,r]*l1 +
                                          l2*frc1[bu1,id2,ru1]*l1 +
                                        -1.0*( l2*l1*frc1[bu2,id1,ru2] ) )
    end
    sync_threads()

    # FIELD TENSOR
    # stores Plaquettes in frc1

    if SFBC && (it == lp.iL[end])
        frc1[b,1,r]     = projalg(U[b,id1,r]*l1/U[b,id2,r])
        frc1[bu1,2,ru1] = zero(TA)
        frc1[bd,3,rd]   = zero(TA)
        frc1[bu2,4,ru2] = projalg(l2*l1)
    else
        frc1[b,1,r]     = projalg(ztw, U[b,id1,r]*l1/U[b,id2,r]) # U_μ(x) U_ν(x+μ) U*_μ(x+ν) U*_ν(x)
        frc1[bu1,2,ru1] = projalg(ztw, l1*l2) # U_ν(x+μ) U*_μ(x+ν) U*_ν(x) U_μ(x)
        frc1[bd,3,rd]   = projalg(ztw, gt2\(l2*gt1)) # U*_μ(x+ν) U*_ν(x) U_μ(x) U_ν(x+μ)
        frc1[bu2,4,ru2] = projalg(ztw, l2*l1) # U*_ν(x) U_μ(x) U_ν(x+μ) U*_μ(x+ν)
    end

    return nothing
end
