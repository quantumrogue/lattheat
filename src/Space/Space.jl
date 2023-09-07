###
### "THE BEER-WARE LICENSE":
### Alberto Ramos wrote this file. As long as you retain this 
### notice you can do whatever you want with this stuff. If we meet some 
### day, and you think this stuff is worth it, you can buy me a beer in 
### return. <alberto.ramos@cern.ch>
###
### file:    Space.jl
### created: Mon Jul 12 16:44:35 2021
###                               


module Space

import Base.show

const BC_PERIODIC = 0
const BC_SF_ORBI  = 1
const BC_SF_AFWB  = 2
const BC_OPEN     = 3

"""
    struct SpaceParm{N,M,B,D}

This structure contains information about the lattice being simulated. The parameters that define the structure are
- `N`: The number of dimensions
- `M`: The number of planes (i.e. \`\` N(N-1)/2 \`\`)
- `B`: The boundary conditions in Euclidean time. Acceptable values are
    - `BC_PERIODIC`: Periodic boundary conditions
    - `BC_SF_AFWB`: Schrödinger Funtional Aoki-Frezzoptti-Weisz Choice B.
    - `BC_SF_ORBI`: Schrödinger Funtional orbifold constructions.
    - `BC_OPEN`: Open boundary conditions.

The structure conatins the following components:
- `iL`: Tuple containing the lattice length in each dimension.
- `plidx`: The directions of each plane
- `blk`: The block size in each each dimension
- `rbk`: The number of blocks in each dimension
- `bsz`: The number of points in each block
- `rsz`: The number of blocks in the lattice
- `ntw`: The twist tensor in each plane
"""

struct SpaceParm{N,M,B,D}
    # N : dimension of the lattice
    # M : nr of planes N(N-1)/2
    # B : boundary conditions
    # D : Total nr of sites per block
    ndim::Int64 # = N
    iL::NTuple{N,Int64} # sizes of lattice (16, 16, 16, 16)
    npls::Int64 # nr planes
    plidx::NTuple{M,Tuple{Int64, Int64}} # tuple of M tuples - define all planes

    blk::NTuple{N,Int64} # sizes of sub-blocks (2, 2, 2, 2)
    blkS::NTuple{N,Int64}
    rbk::NTuple{N,Int64} # nr of blocks in each lat direction (8, 8, 8, 8)
    rbkS::NTuple{N,Int64}

    bsz::Int64 # number of points per block = nr of threads
    rsz::Int64 # total number of blocks

    ntw::NTuple{M,Int64}

    function SpaceParm{N}(x, y, nt::Union{Nothing,NTuple{I,Int64}}=nothing) where {N,I}
        # x: sizes of lattice - iL
        # y: sizes of sub-blocks - blk

        M = convert(Int64, round(N*(N-1)/2))
        N == length(x) || throw(ArgumentError("Lattice size incorrect length for dimension $N"))
        N == length(y) || throw(ArgumentError("Block   size incorrect length for dimension $N"))

        if any(i->i!=0, x.%y)
            error("Lattice size not divisible by block size.")
        end

        # get direction of all planes
        pls = Vector{Tuple{Int64, Int64}}()
        for i in N:-1:1
            for j in 1:i-1
                push!(pls, (i,j))
            end
        end

        r  = div.(x, y) # r contains the number of blocks in each direction
        rS = ones(N) # partition of the blocks per node in each direction
        yS = ones(N)
        for i in 2:N
            for j in 1:i-1
                rS[i] = rS[i]*r[j]
                yS[i] = yS[i]*y[j]
            end
        end

        D = prod(y)
        if nt == nothing
            ntw = ntuple(i->0, M)
        else
            ntw = nt
        end
        return new{N,M,BC_PERIODIC,D}(N, x, M, tuple(pls...), y,
                                      tuple(yS...), tuple(r...), tuple(rS...), prod(y), prod(r), ntw)
    end

    function SpaceParm{N}(x, y, ibc::Int, nt::Union{Nothing,NTuple{I,Int64}}=nothing) where {N,I}
        M = convert(Int64, round(N*(N-1)/2))
        N == length(x) || throw(ArgumentError("Lattice size incorrect length for dimension $N"))
        N == length(y) || throw(ArgumentError("Block   size incorrect length for dimension $N"))

        if any(i->i!=0, x.%y)
            error("Lattice size not divisible by block size.")
        end

        if nt!=nothing
            if (ibc==BC_SF_AFWB) || (ibc==BC_SF_ORBI)
                if any(i->i!=0, nt[1:N-1])
                    error("Planes in T direction cannot be twisted with SF boundary conditions")
                end
            end
        end

        pls = Vector{Tuple{Int64, Int64}}()
        for i in N:-1:1
            for j in 1:i-1
                push!(pls, (i,j))
            end
        end

        r  = div.(x, y)
        rS = ones(N)
        yS = ones(N)
        for i in 2:N
            for j in 1:i-1
                rS[i] = rS[i]*r[j]
                yS[i] = yS[i]*y[j]
            end
        end

        D = prod(y)
        if nt == nothing
            ntw = ntuple(i->0,M)
        else
            ntw = nt
        end
        return new{N,M,ibc,D}(N, x, M, tuple(pls...), y,
                              tuple(yS...), tuple(r...), tuple(rS...), prod(y), prod(r), ntw)
    end
end
export SpaceParm

function Base.show(io::IO, lp::SpaceParm{N,M,B,D}) where {N,M,B,D}
    println(io, "Lattice dimensions:       ", lp.ndim)

    print(io, "Lattice size:             ")
    for i in 1:lp.ndim-1
        print(io, lp.iL[i], " x ")
    end
    println(io,lp.iL[end])
    print(io, "Time boundary conditions: ")
    if B == BC_PERIODIC
        println(io, "PERIODIC")
    elseif B == BC_OPEN
        println(io, "OPEN")
    elseif B == BC_SF_AFWB
        println(io, "SF (AFW option B)")
    elseif B == BC_SF_ORBI
        println(io, "SF (orbifold improvement)")
    end

    print(io, "Thread block size:        ")
    for i in 1:lp.ndim-1
        print(io, lp.blk[i], " x ")
    end
    println(io,lp.blk[end], "     [", lp.bsz,
            "] (Number of blocks: [", lp.rsz,"])")
    println(io, "Twist tensor: ", lp.ntw)

    return
end

"""
    function up(p::NTuple{2,Int64}, id::Int64, lp::SpaceParm)

Given a point `x` with index `p`, this routine returns the index of the point
`x + a id`.
"""
@inline function up(p::NTuple{2,Int64}, id::Int64, lp::SpaceParm)

    ic = mod(div(p[1]-1,lp.blkS[id]),lp.blk[id])
    if (ic == lp.blk[id]-1)
        b = p[1] - (lp.blk[id]-1)*lp.blkS[id]

        ic = mod(div(p[2]-1,lp.rbkS[id]),lp.rbk[id])
        if (ic == lp.rbk[id]-1)
            r = p[2] - (lp.rbk[id]-1)*lp.rbkS[id]
        else
            r = p[2] + lp.rbkS[id]
        end
    else
        b = p[1] + lp.blkS[id]
        r = p[2]
    end


    return b, r
end

"""
    function dw(p::NTuple{2,Int64}, id::Int64, lp::SpaceParm)

Given a point `x` with index `p`, this routine returns the index of the point
`x - a id`.
"""
@inline function dw(p::NTuple{2,Int64}, id::Int64, lp::SpaceParm)

    ic = mod(div(p[1]-1,lp.blkS[id]),lp.blk[id])
    if (ic == 0)
        b = p[1] + (lp.blk[id]-1)*lp.blkS[id]
        ic = mod(div(p[2]-1,lp.rbkS[id]),lp.rbk[id])
        if (ic == 0)
            r = p[2] + (lp.rbk[id]-1)*lp.rbkS[id]
        else
            r = p[2] - lp.rbkS[id]
        end
    else
        b = p[1] - lp.blkS[id]
        r = p[2]
    end


    return b, r
end

"""
    function updw(p::NTuple{2,Int64}, id::Int64, lp::SpaceParm)

Given a point `x` with index `p`, this routine returns the index of the points
`x + a id` and `x - a id`.
"""
@inline function updw(p::NTuple{2,Int64}, id::Int64, lp::SpaceParm)

    ic = mod(div(p[1]-1,lp.blkS[id]),lp.blk[id])
    if (ic == lp.blk[id]-1)
        bu = p[1] - (lp.blk[id]-1)*lp.blkS[id]
        bd = p[1] - lp.blkS[id]

        ic = mod(div(p[2]-1,lp.rbkS[id]),lp.rbk[id])
        if (ic == lp.rbk[id]-1)
            ru = p[2] - (lp.rbk[id]-1)*lp.rbkS[id]
        else
            ru = p[2] + lp.rbkS[id]
        end
        rd = p[2]
    elseif (ic == 0)
        bu = p[1] + lp.blkS[id]
        bd = p[1] + (lp.blk[id]-1)*lp.blkS[id]

        ic = mod(div(p[2]-1,lp.rbkS[id]),lp.rbk[id])
        if (ic == 0)
            rd = p[2] + (lp.rbk[id]-1)*lp.rbkS[id]
        else
            rd = p[2] - lp.rbkS[id]
        end
        ru = p[2]
    else
        bu = p[1] + lp.blkS[id]
        bd = p[1] - lp.blkS[id]
        rd = p[2]
        ru = p[2]
    end


    return bu, ru, bd, rd
end

@inline cntb(nb, id::Int64, lp::SpaceParm) = mod(div(nb-1,lp.blkS[id]),lp.blk[id])
@inline cntr(nr, id::Int64, lp::SpaceParm) = mod(div(nr-1,lp.rbkS[id]),lp.rbk[id])
@inline cnt(nb, nr, id::Int64, lp::SpaceParm)  = 1 + cntb(nb,id,lp) + cntr(nr,id,lp)*lp.blk[id]

@inline function point_coord(p::NTuple{2,Int64}, lp::SpaceParm{2,M,D}) where {M,D}

    i1 = cnt(p[1], p[2], 1, lp)
    i2 = cnt(p[1], p[2], 2, lp)

#    pt = ntuple(i -> cnt(p[1], p[2], i, lp), lp.ndim)
    return CartesianIndex{2}(i1,i2)
end

@inline function point_coord(p::NTuple{2,Int64}, lp::SpaceParm{3,M,D}) where {M,D}

    i1 = cnt(p[1], p[2], 1, lp)
    i2 = cnt(p[1], p[2], 2, lp)
    i3 = cnt(p[1], p[2], 3, lp)

#    pt = ntuple(i -> cnt(p[1], p[2], i, lp), lp.ndim)
    return CartesianIndex{3}(i1,i2,i3)
end

@inline function point_coord(p::NTuple{2,Int64}, lp::SpaceParm{4,M,D}) where {M,D}

    i1 = cnt(p[1], p[2], 1, lp)
    i2 = cnt(p[1], p[2], 2, lp)
    i3 = cnt(p[1], p[2], 3, lp)
    i4 = cnt(p[1], p[2], 4, lp)

#    pt = ntuple(i -> cnt(p[1], p[2], i, lp), lp.ndim)
    return CartesianIndex{4}(i1,i2,i3,i4)
end

@inline function point_coord(p::NTuple{2,Int64}, lp::SpaceParm{5,M,D}) where {M,D}

    i1 = cnt(p[1], p[2], 1, lp)
    i2 = cnt(p[1], p[2], 2, lp)
    i3 = cnt(p[1], p[2], 3, lp)
    i4 = cnt(p[1], p[2], 4, lp)
    i5 = cnt(p[1], p[2], 5, lp)

#    pt = ntuple(i -> cnt(p[1], p[2], i, lp), lp.ndim)
    return CartesianIndex{5}(i1,i2,i3,i4,i5)
end

@inline function point_coord(p::NTuple{2,Int64}, lp::SpaceParm{6,M,D}) where {M,D}

    i1 = cnt(p[1], p[2], 1, lp)
    i2 = cnt(p[1], p[2], 2, lp)
    i3 = cnt(p[1], p[2], 3, lp)
    i4 = cnt(p[1], p[2], 4, lp)
    i5 = cnt(p[1], p[2], 5, lp)
    i6 = cnt(p[1], p[2], 6, lp)

#    pt = ntuple(i -> cnt(p[1], p[2], i, lp), lp.ndim)
    return CartesianIndex{6}(i1,i2,i3,i4,i5,i6)
end

@inline function point_time(p::NTuple{2,Int64}, lp::SpaceParm{N,M,D}) where {N,M,D}
    return cnt(p[1], p[2], N, lp)
end


@inline function point_index(pt::CartesianIndex, lp::SpaceParm)

    b = 1
    r = 1
    for i in 1:length(pt)
        b = b + ((pt[i]-1)%lp.blk[i])*lp.blkS[i]
        r = r + div((pt[i]-1),lp.blk[i])*lp.rbkS[i]
    end

    return (b,r)
end

"""
    function point_color(p::NTuple{2,Int64}, lp::SpaceParm)

Returns the sum of the cartesian coordinates of the point p=(b,r).
"""
@inline function point_color(p::NTuple{2,Int64}, lp::SpaceParm)

    s = cnt(p[1], p[2], 1, lp)
    for i in 2:lp.ndim
        s = s + cnt(p[1], p[2], i, lp)
    end

    return s
end

export up, dw, updw, point_index, point_coord, point_time
export BC_PERIODIC, BC_OPEN, BC_SF_AFWB, BC_SF_ORBI

end
