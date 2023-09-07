###
### "THE BEER-WARE LICENSE":
### Alberto Ramos wrote this file. As long as you retain this
### notice you can do whatever you want with this stuff. If we meet some
### day, and you think this stuff is worth it, you can buy me a beer in
### return. <alberto.ramos@cern.ch>
###
### file:    YMhmc.jl
### created: Thu Jul 15 11:27:28 2021
###

"""

    function gauge_action(U, lp::SpaceParm, gp::GaugeParm, ymws::YMworkspace)

Returns the value of the gauge plaquette action for the configuration U. The parameters `\beta` and `c0` are taken from the `gp` structure.
"""
function gauge_action(U, lp::SpaceParm, gp::GaugeParm, ymws::YMworkspace{T}) where T <: AbstractFloat

    ztw = ztwist(gp, lp)
    if abs(gp.c0-1) < 1.0E-10
        @timeit "Wilson gauge action" begin
            CUDA.@sync begin
                CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_plaq!(ymws.cm, U, gp.Ubnd, gp.cG[1], ztw, lp)
            end
        end
    else
        @timeit "Improved gauge action" begin
            CUDA.@sync begin
                CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_impr!(ymws.cm, U, gp.c0, (1-gp.c0)/8, gp.Ubnd, gp.cG[1], ztw, lp)
            end
        end
    end
    S = gp.beta*( prod(lp.iL)*lp.npls*(gp.c0 + (1-gp.c0)/8) -
                  CUDA.mapreduce(real, +, ymws.cm)/gp.ng )

    return S
end

function plaquette(U, lp::SpaceParm{N,M,B,D}, gp::GaugeParm, ymws::YMworkspace) where {N,M,B,D}

    ztw = ztwist(gp, lp)
    @timeit "Plaquette measurement" begin
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_plaq!(ymws.cm, U, gp.Ubnd, one(gp.cG[1]), ztw, lp)
        end
    end

    return CUDA.mapreduce(real, +, ymws.cm)/(prod(lp.iL)*lp.npls)
end

function hamiltonian(mom, U, lp, gp, ymws)
    @timeit "Computing Hamiltonian" begin
        K = CUDA.mapreduce(norm2, +, mom)/2
        V = gauge_action(U, lp, gp, ymws)
    end

    return K+V
end

function HMC!(U, int::IntrScheme, lp::SpaceParm, gp::GaugeParm, ymws::YMworkspace{T}; noacc=false) where T

    @timeit "HMC trayectory" begin

        ymws.U1 .= U

        randomize!(ymws.mom, lp, ymws)
        hini = hamiltonian(ymws.mom, U, lp, gp, ymws)

        MD!(ymws.mom, U, int, lp, gp, ymws)

        dh   = hamiltonian(ymws.mom, U, lp, gp, ymws) - hini
        pacc = exp(-dh)

        acc = true
        if (noacc)
            return dh, acc
        end

        if (pacc < 1.0)
            r = rand()
            if (pacc < r)
                U .= ymws.U1
                acc = false
            end
        end

        U .= unitarize.(U)

    end
    return dh, acc
end
HMC!(U, eps, ns, lp::SpaceParm, gp::GaugeParm, ymws::YMworkspace{T}; noacc=false) where T = HMC!(U, omf4(T, eps, ns), lp, gp, ymws; noacc=noacc)

function MD!(mom, U, int::IntrScheme{NI, T}, lp::SpaceParm, gp::GaugeParm, ymws::YMworkspace{T}) where {NI, T <: AbstractFloat}

    @timeit "MD evolution" begin

        ee = int.eps*gp.beta/gp.ng
        force_gauge(ymws, U, gp.c0, gp, lp)
        mom .= mom .+ (int.r[1]*ee) .* ymws.frc1
        for i in 1:int.ns
            k   = 2
            off = 1
            for j in 1:NI-1
                U .= expm.(U, mom, int.eps*int.r[k])
                if k == NI
                    off = -1
                end
                k += off

                force_gauge(ymws, U, gp.c0, gp, lp)
                if (i < int.ns) && (k == 1)
                    mom .= mom .+ (2*int.r[k]*ee) .* ymws.frc1
                else
                    mom .= mom .+ (int.r[k]*ee) .* ymws.frc1
                end
                k += off
            end
        end
    end

    return nothing
end
