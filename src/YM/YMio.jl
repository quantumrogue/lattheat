###
### "THE BEER-WARE LICENSE":
### Alberto Ramos wrote this file. As long as you retain this
### notice you can do whatever you want with this stuff. If we meet some
### day, and you think this stuff is worth it, you can buy me a beer in
### return. <alberto.ramos@cern.ch>
###
### file:    YMio.jl
### created: Wed Nov 10 12:58:27 2021
###

"""
    read_cnfg(U, lp::SpaceParm{4,M,B,D},)

Reads configuration from file `fname` using the native (BDIO) format.
"""
function read_cnfg(fname::String)

    UID_HDR = 14
    fb = BDIO_open(fname, "r")
    while BDIO_get_uinfo(fb) != UID_HDR
        BDIO_seek!(fb)
    end
    ihdr = Vector{Int32}(undef, 2)
    BDIO_read(fb, ihdr)
    if (ihdr[1] != convert(Int32, 1653996111)) && (ihdr[2] != convert(Int32, 2))
        error("Wrong file format [header]")
    end

    run = BDIO.BDIO_read_str(fb)

    while BDIO_get_uinfo(fb) != 1
        BDIO_seek!(fb)
    end

    ifoo = Vector{Int32}(undef, 4)
    BDIO_read(fb, ifoo)
    ndim = convert(Int64, ifoo[1])
    npls = convert(Int64, round(ndim*(ndim-1)/2))
    ibc  = convert(Int64, ifoo[2])
    nf   = ifoo[4]

    ifoo = Vector{Int32}(undef, ndim+convert(Int32, npls))
    BDIO_read(fb, ifoo)
    iL  = ntuple(i -> convert(Int64, ifoo[i]),ndim)
    ntw = ntuple(i -> convert(Int64, ifoo[i+ndim]), npls)

    dfoo = Vector{Float64}(undef, 5)
    BDIO_read(fb, dfoo)
    ng = dfoo[5]
    G = SU2
    if ng == 3
        G = SU3
    end

    lp = SpaceParm{ndim}(iL, (4,4,4,4), ibc, ntw)
    gp = GaugeParm{Float64}(SU3{Float64}, dfoo[1], dfoo[2])

    dtr = (2,3,4,1)
    assign(id, V, i3,::Type{SU3{T}}) where T <: AbstractFloat = SU3{T}(V[1,dtr[id],i3],V[2,dtr[id],i3],V[3,dtr[id],i3],
                                     V[4,dtr[id],i3],V[5,dtr[id],i3],V[6,dtr[id],i3])
    assign(id, V, i3,::Type{SU2{T}}) where T <: AbstractFloat = SU2{T}(V[1,dtr[id],i3],V[2,dtr[id],i3])

    while BDIO_get_uinfo(fb) != 8
        BDIO_seek!(fb)
    end
    Ucpu = Array{G{Float64}, 3}(undef, lp.bsz, lp.ndim, lp.rsz)
    #Matrix dimensions
    mdim(::Type{SU2{T}}) where T <: AbstractFloat = 4
    mdim(::Type{SU3{T}}) where T <: AbstractFloat = 9
    m = mdim(G{Float64})
    V = Array{ComplexF64, 3}(undef, m, lp.ndim, lp.iL[3])
    for i4 in 1:lp.iL[4]
        for i1 in 1:lp.iL[1]
            for i2 in 1:lp.iL[2]
                BDIO_read(fb, vec(V))
                for i3 in 1:lp.iL[3]
                    b, r = point_index(CartesianIndex(i1,i2,i3,i4), lp)
                    for id in 1:lp.ndim
                        Ucpu[b,id,r] = assign(id, V, i3, G{Float64})
                    end
                end
            end
        end
    end

    if ibc == BC_SF_AFWB || ibc == BC_SF_ORBI
        BDIO_read(fb, V)
        Ubnd = ntuple(i->assign(i, V, 1, G{Float64}), 3)
        BDIO_close!(fb)

        return CuArray(Ucpu), Ubnd
    else
        BDIO_close!(fb)
        return CuArray(Ucpu)
    end
end


"""
    save_cnfg(fname, U, lp::SpaceParm, gp::GaugeParm; run::Union{Nothing,String}=nothing)

Saves configuration `U` in the file `fname` using the native (BDIO) format.
"""
function save_cnfg(fname::String, U, lp::SpaceParm{4,M,B,D}, gp::GaugeParm{T,G,N}; run::Union{Nothing,String}=nothing) where {M,B,D,T,G,N}

    ihdr = [convert(Int32, 1653996111),convert(Int32,2)]
    UID_HDR = 14

    degree(::Type{SU2{T}}) where T <: AbstractFloat = 2
    degree(::Type{SU3{T}}) where T <: AbstractFloat = 3
    ng = degree(G)

    if isfile(fname)
        fb = BDIO_open(fname, "a")
    else
        fb = BDIO_open(fname, "w", "Configuration of LatticeGPU.jl")
        BDIO_start_record!(fb, BDIO_BIN_GENERIC, UID_HDR)
        BDIO_write!(fb, ihdr)
        if run != nothing
            BDIO_write!(fb, run*"\0")
        end
        BDIO_write_hash!(fb)

        dfoo = Vector{Float64}(undef, 16)
        BDIO_start_record!(fb, BDIO_BIN_GENERIC, 1)
        BDIO_write!(fb, [convert(Int32, 4)])
        BDIO_write!(fb, [convert(Int32, B)]) #boundary
        BDIO_write!(fb, [convert(Int32, gp.ng)])
        BDIO_write!(fb, [convert(Int32, 0)])
        BDIO_write!(fb, [convert(Int32, lp.iL[i]) for i in 1:4])
        BDIO_write!(fb, [convert(Int32, lp.ntw[i]) for i in 1:M])
        BDIO_write!(fb, [gp.beta, gp.c0, gp.cG[1], gp.cG[2], ng])
    end
    BDIO_write_hash!(fb)

    dtr = (2,3,4,1)

    function assign!(id, V, i3, M::M2x2{T}) where T

        V[1,dtr[id],i3] = M.u11
        V[2,dtr[id],i3] = M.u12
        V[3,dtr[id],i3] = M.u21
        V[4,dtr[id],i3] = M.u22
        return nothing
    end
    assign!(id, V, i3, g::SU2{T}) where T = assign!(id,V,i3,convert(M2x2{T}, g))

    function assign!(id, V, i3, M::M3x3{T}) where T

        V[1,dtr[id],i3] = M.u11
        V[2,dtr[id],i3] = M.u12
        V[3,dtr[id],i3] = M.u13
        V[4,dtr[id],i3] = M.u21
        V[5,dtr[id],i3] = M.u22
        V[6,dtr[id],i3] = M.u23
        V[7,dtr[id],i3] = M.u31
        V[8,dtr[id],i3] = M.u32
        V[9,dtr[id],i3] = M.u33
        return nothing
    end
    assign!(id, V, i3, g::SU3{T}) where T = assign!(id,V,i3,convert(M3x3{T}, g))


    Ucpu = Array(U)
    #Matrix dimensions
    mdim(::Type{SU2{T}}) where T <: AbstractFloat = 4
    mdim(::Type{SU3{T}}) where T <: AbstractFloat = 9
    m = mdim(G)

    BDIO_start_record!(fb, BDIO_BIN_F64LE, 8, true)
    #way to write all fields coming from the same point in the same array
    #index 1 - 4 matrix elements for 2x2; 9 matrix elements of 3x3
    #index 2 - number of directions \mu
    #index 3 - lattice points in direction z
    V = Array{ComplexF64, 3}(undef, m, lp.ndim, lp.iL[3])
    for i4 in 1:lp.iL[4]
        for i1 in 1:lp.iL[1]
            for i2 in 1:lp.iL[2]
                for i3 in 1:lp.iL[3]
                    b, r = point_index(CartesianIndex(i1,i2,i3,i4), lp)
                    for id in 1:lp.ndim
                        assign!(id, V, i3, Ucpu[b,id,r])
                    end
                end
                # write all all L fields along space direction 3
                BDIO_write!(fb, V)
            end
        end
    end

    if B == BC_SF_AFWB || B == BC_SF_ORBI
        for i3 in 1:lp.iL[3]
            for id in 1:lp.ndim-1
                assign!(id, V, i3, Ubnd[id])
            end
            if ng == 2
                assign!(4, V, i3, zero(M2x2{Float64}))
            elseif ng == 3
                assign!(4, V, i3, zero(M3x3{Float64}))
            end
        end
        for i1 in 1:lp.iL[1]
            for i2 in 1:lp.iL[2]
                BDIO_write!(fb, V)
            end
        end
    end
    BDIO_write_hash!(fb)
    BDIO_close!(fb)

    return nothing
end



"""
    function import_lex64(fname::String, lp::SpaceParm)

import a double precision configuration in lexicographic format. SF boundary conditions are assummed.
"""
function import_lex64(fname, lp::SpaceParm)

    fp = open(fname, "r")

    dtr = [2,3,4,1]

    assign(id, V, i3) = SU3{Float64}(V[1,dtr[id],i3],V[2,dtr[id],i3],V[3,dtr[id],i3],
                                     V[4,dtr[id],i3],V[5,dtr[id],i3],V[6,dtr[id],i3])

    Ucpu = Array{SU3{Float64}, 3}(undef, lp.bsz, lp.ndim, lp.rsz)
    V = Array{ComplexF64, 3}(undef, 9, lp.ndim, lp.iL[3])
    for i4 in 1:lp.iL[4]
        for i1 in 1:lp.iL[1]
            for i2 in 1:lp.iL[2]
                read!(fp, V)
                for i3 in 1:lp.iL[3]
                    b, r = point_index(CartesianIndex(i1,i2,i3,i4), lp)
                    for id in 1:lp.ndim
                        Ucpu[b,id,r] = assign(id, V, i3)
                    end
                end
            end
        end
    end

    read!(fp, V)
    Ubnd = ntuple(i->assign(i, V, 1), 3)
    close(fp)

    return CuArray(Ucpu), Ubnd
end

"""
    function import_cern64(fname::String, ibc, lp::SpaceParm)

import a double precision configuration in CERN format.
"""
function import_cern64(fname, ibc, lp::SpaceParm; log=true)

    fp = open(fname, "r")
    iL = Vector{Int32}(undef, 4)
    read!(fp, iL)
    avgpl = Vector{Float64}(undef, 1)
    read!(fp, avgpl)
    if log
        println("# [import_cern64] Read from conf file: ", iL, " (plaq: ", avgpl, ")")
    end

    dtr = [4,1,2,3]
    assign(V, ic) = SU3{Float64}(V[1,ic],V[2,ic],V[3,ic],V[4,ic],V[5,ic],V[6,ic])

    Ucpu = Array{SU3{Float64}, 3}(undef, lp.bsz, lp.ndim, lp.rsz)
    V = Array{ComplexF64, 2}(undef, 9, 2)
    for i4 in 1:lp.iL[4]
        for i1 in 1:lp.iL[1]
            for i2 in 1:lp.iL[2]
                for i3 in 1:lp.iL[3]
                    if (mod(i1+i2+i3+i4-4, 2) == 1)
                        b, r = point_index(CartesianIndex(i1,i2,i3,i4), lp)
                        for id in 1:lp.ndim
                            read!(fp, V)
                            Ucpu[b,dtr[id],r] = assign(V, 1)

                            bd, rd = dw((b,r), dtr[id], lp)
                            Ucpu[bd,dtr[id],rd] = assign(V, 2)
                        end
                    end
                end
            end
        end
    end
    close(fp)

    return CuArray(Ucpu)
end
