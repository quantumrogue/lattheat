###                               
### author:  pietro.butti
### file:    Phi4.jl
### created: Wed 27 Sep 2023 15:21:29 CEST
### 


module Phi4

    using CUDA, Random, TimerOutputs, BDIO
    using ..Space
    using ..Groups
    using ..Fields
    using ..MD

    import Base.show

    ## ============================================= BASE ======================================================
        struct Phi4Parm{T}
            κ::T
            λ::T
        end
        function Base.show(io::IO, gp::Phi4Patm{T}) where {T}
            println(io, "ϕ⁴ theory with")
            println(io, " - κ:  ", gp.κ)
            println(io, " - λ:  ", gp.λ)
            return nothing
        end

        function Phi4workspace{T}
            PRC
            frc
            mom
            ϕ
            cbin # complex of volume
            rbin # float of volume

            ϕbin

            function Phi4workspace(::Type{T}, lp::SpaceParm) where T<:AbstractFloat
                @timeit "Allocating workspace" begin
                    F   = scalar_field(T,lp)
                    PI  = scalar_field(T,lp)
                    phi = scalar_field(T,lp)
                    bin = scalar_field(T,lp)

                    cs = scalar_field_point(Complex{T}, lp)
                    rs = scalar_field_point(T, lp)
                end
                return new{T}(T,F,PI,phi,cs,rs,bin)
            end
        end
    ## ========================================================================================================

    ## ============================================== FIELD ====================================================
        function randomize!(f,ϕp::Phi4Parm,lp::SpaceParm)
            @timeit "Randomize scalar field" begin
                m = CUDA.randn(ϕws.PRC, lp.bsz, 4, 1, lp.rsz)
                CUDA.@sync begin
                    CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_assign_scalar!(f,m,sp,lp)
                end
            end
        
            return nothing
        end
        function krnl_assign_scalar!(f::AbstractArray{T},m,ϕp::Phi4Parm,lp::SpaceParm) where T
            SR2::typeof(ϕp.κ) = 1.4142135623730951
            
            b, r = CUDA.threadIdx().x, CUDA.blockIdx().x
            f[b,r] = m[b,1,1,r] * SR2

            return nothing
        end
    ## ========================================================================================================

    ## ================================================ ACTION =================================================
        function phi4_action(ϕ, lp::SpaceParm, ϕp::Phi4Parm{T}, ϕws::Phi4workspace{T}) where {T <: AbstractFloat}

            @timeit "ϕ⁴ action" begin
                CUDA.@sync begin
                    CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_act!(ϕws.rbin,ϕ,ϕp,lp)
                end
            end
                
            S = CUDA.reduce(+,ϕws.rbin)
            return S
        end

        function krnl_act!(act, ϕ::AbstractArray{TS}, ϕp::Phi4Parm{T}, lp::SpaceParm{N,M,B,D}) where {TS,T,N,M,B,D}
            b = Int64(CUDA.threadIdx().x)
            r = Int64(CUDA.blockIdx().x)
            
            S = zero(eltype(act))
            for μ in 1:N
                # up_b, up_r, dw_b, dw_r = updw((b,r),dir,lp)
                bu, ru = up((b, r), μ, lp)
                S += -2. * ϕp.κ * (ϕ[b,r]*(ϕ[bu,ru] + ϕ[b,r]) - ϕp.λ*(ϕ[b,r]-one(T)) - ϕp.λ)
            end

            I = point_coord((b,r), lp)
            act[I] = S

            return nothing
        end
    ## ========================================================================================================

    ## ================================================ FORCE =================================================
        function phi4_force(ϕws::Phi4workspace{T}, ϕ, ϕp::Phi4Parm{T}, lp::SpaceParm) where T<:AbstractFloat
            @timeit "ϕ⁴ force" begin
                CUDA.@sync begin
                    CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_phi4_force!(ϕws.frc,ϕ,ϕp,lp)
                end
            end
            return nothing
        end
        function krnl_phi4_force!(Fϕ, ϕ::AbstractArray{TS}, ϕp::Phi4Parm{T}, lp::SpaceParm{N,M,B,D}) where {TS,T,N,M,B,D}
            b = Int64(CUDA.threadIdx().x)
            r = Int64(CUDA.blockIdx().x)
            
            Fϕ[b,r] = 2. * ϕ[b,r] + 4. * ϕp.λ*(ϕ[b,r]-1.)*ϕ[b,r]
            for μ in 1:N
                up_b, up_r, dw_b, dw_r = updw((b,r),μ,lp)
                Fϕ[b,r] -=  2. * ϕp.κ * (ϕ[up_b,up_r] - ϕ[dw_b,dw_r])
            end       
            return nothing 
        end
    ## ========================================================================================================

    ## ============================================= HMC ======================================================
        function hamiltonian(mom,ϕ,lp,ϕp,ϕws)
            @timeit "Computing Hamiltonian" begin
                Sϕ = phi4_force(ϕws,ϕ,ϕp,lp)
                Sπ = CUDA.mapreduce(norm2, +, mom)/2
            end
            return Sϕ + Sπ
        end

        function MD!(mom,ϕ, int::IntrScheme{NI, T}, lp::SpaceParm, ϕp::Phi4Parm, ϕws::Phi4workspace{T}) where {NI, T <: AbstractFloat}
            @timeit "MD evolution" begin

                phi4_force(ϕws,ϕ,ϕp,lp)
                mom .= mom .+ (int.r[1]*int.eps) .* ϕws.frc

                for i in 1:int.ns
                    k   = 2
                    off = 1
                    for j in 1:NI-1
                        ϕ .= ϕ .+ (int.eps*int.r[k]).*mom
                        if k == NI
                            off = -1
                        end
                        k += off

                        phi4_force(ϕws,ϕ,ϕp,lp)
                        mom .= mom .+ ((((i < int.ns) && (k == 1)) ? 2 : 1) * int.r[k]*int.eps) .* ϕ.frc

                        k += off
                    end
                end
            end
            return nothing
        end

        function HMC!(ϕ,  int::IntrScheme, lp::SpaceParm, gp::Phi4Parm, ϕws::Phi4workspace{T}; noacc=false) where T
            @timeit "HMC trajectory" begin
                ϕws.bin .= ϕ

                randomize!(ϕws.mom,lp,ϕws)
                Hin = hamiltonian(ϕws.mom,ϕ,lp,ϕp,ϕws)

                MD!(ϕws.mom,ϕ,int,lp,ϕp,ϕws)

                ΔH = hamiltonian(ϕws.mom,ϕ,lp,ϕp,ϕws) - Hin
                pacc = exp(-ΔH)

                acc = true
                if (noacc)
                    return dh, acc
                end
        
                if (pacc < 1.0)
                    r = rand()
                    if (pacc < r) 
                        ϕ .= ϕws.ϕ
                        acc = false
                    end
                end
            end
            return ΔH, acc
        end
    ## ========================================================================================================



    export Phi4Parm, Phi4workspace
    export randomize!
    export phi4_action
    export phi4_force
    export HMC!
end

