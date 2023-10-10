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
        # struct Phi4Parm{T}
        #     κ::T
        #     λ::T
        # end
        # Phi4Parm(kappa::T,lambda::T) where T<:AbstractFloat = Phi4Parm{T}(kappa,lambda)
        # function Base.show(io::IO, gp::Phi4Parm{T}) where {T}
        #     println(io, "ϕ⁴ theory with")
        #     println(io, " - κ:  ", gp.κ)
        #     println(io, " - λ:  ", gp.λ)
        #     return nothing
        # end

        # Phi4ParmM2L(m2,lambda) = Phi4Parm{typeof(m2)}(1/(m2+8+4*lambda),2*lambda/(m2+8+4*lambda))

        struct Phi4Parm{T}
            m2::T
            λ::T
        end
        Phi4Parm(m2::T,lambda::T) where T<:AbstractFloat = Phi4Parm{T}(m2,lambda)
        function Base.show(io::IO, gp::Phi4Parm{T}) where {T}
            println(io, "ϕ⁴ theory with")
            println(io, " - m²: ", gp.m2)
            println(io, " - λ:  ", gp.λ)
            return nothing
        end

        struct Phi4workspace{T}
            PRC
            frc
            mom
            ϕ
            cbin # complex of volume
            rbin # float of volume

            bin

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
        function randomize!(f,ϕws::Phi4workspace,ϕp::Phi4Parm,lp::SpaceParm)

            @timeit "Randomize scalar field" begin
                m = CUDA.randn(ϕws.PRC, lp.bsz, lp.rsz)
                CUDA.@sync begin
                    CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_assign_scalar!(f,m,ϕp)
                end
            end
        
            return nothing
        end
        function krnl_assign_scalar!(f::AbstractArray{T},m,ϕp::Phi4Parm) where T
            b, r = CUDA.threadIdx().x, CUDA.blockIdx().x
            f[b,r] = m[b,r]
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
            
            ## This is the action like [Sternbeck]
            # phi = ϕ[b,r]
            # S = phi^2 + ϕp.λ * (phi*phi-one(TS))^2 - ϕp.λ
            # for μ in 1:N
            #     bu, ru = up((b, r), μ, lp)
            #     S += -2. * ϕp.κ * phi * ϕ[bu,ru]
            # end
            
            ## This is the action like 2307.15406
            S = ϕ[b,r]*ϕ[b,r] * ϕp.m2/2 + (ϕ[b,r]*ϕ[b,r]*ϕ[b,r]*ϕ[b,r]) * ϕp.λ
            for μ in 1:N
                bu, ru = up((b, r), μ, lp)
                S += (ϕ[bu,ru] - ϕ[b,r])*(ϕ[bu,ru] - ϕ[b,r])/2
            end

            I = point_coord((b,r), lp)
            act[I] = S

            return nothing
        end


        function hopping(ϕ, lp::SpaceParm, ϕws::Phi4workspace{T}) where {T <: AbstractFloat}
            @timeit "Hopping term" begin
                CUDA.@sync begin
                    CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_hopping!(ϕws.rbin,ϕ,lp)
                end
            end
            J = CUDA.reduce(+,ϕws.rbin)
            return J
        end
        function krnl_hopping!(hop, ϕ::AbstractArray{TS}, lp::SpaceParm{N,M,B,D}) where {TS,N,M,B,D}
            b = Int64(CUDA.threadIdx().x)
            r = Int64(CUDA.blockIdx().x)
            
            J = zero(TS)
            ## This is the hopping like [Sternbeck]
            # for μ in 1:N
            #     bu, ru = up((b, r), μ, lp)
            #     J += ϕ[bu,ru]*ϕ[b,r]
            # end

            ## This is the hopping for 2307.15406
            for μ in 1:N
                bu, ru = up((b, r), μ, lp)
                J += (ϕ[bu,ru] - ϕ[b,r])*(ϕ[bu,ru] - ϕ[b,r])
            end

            I = point_coord((b,r), lp)
            hop[I] = J

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
            
            ## This is the force like [Sternbeck]
            # phi = ϕ[b,r]
            # Fϕ[b,r] = 2. * phi + 4. * ϕp.λ*(phi*phi-one(TS))*phi
            # for μ in 1:N
            #     up_b, up_r, dw_b, dw_r = updw((b,r),μ,lp)
            #     Fϕ[b,r] -=  2. * ϕp.κ * (ϕ[up_b,up_r] + ϕ[dw_b,dw_r])
            # end       

            ## This is the force like 2307.15406
            Fϕ[b,r] = (ϕp.m2 + 8)*ϕ[b,r] + 4*ϕp.λ*ϕ[b,r]*ϕ[b,r]*ϕ[b,r]
            for μ in 1:N
                up_b, up_r, dw_b, dw_r = updw((b,r),μ,lp)
                Fϕ[b,r] -= (ϕ[up_b,up_r] + ϕ[dw_b,dw_r])
            end  

            return nothing
        end
    ## ========================================================================================================

    ## ============================================= HMC ======================================================
        function hamiltonian(mom,ϕ,lp,ϕp,ϕws)
            @timeit "Computing Hamiltonian" begin
                Sϕ = phi4_action(ϕ,lp,ϕp,ϕws)
                Sπ = CUDA.mapreduce(abs2, +, mom)/2
            end
            return Sϕ + Sπ
        end

        function MD!(mom,ϕ,Δτ, lp::SpaceParm, ϕp::Phi4Parm, ϕws::Phi4workspace{T}) where {T <: AbstractFloat}
            phi4_force(ϕws,ϕ,ϕp,lp)
            mom .= mom .- Δτ/2 .* ϕws.frc

            ϕ .= ϕ .+ Δτ .* mom
        
            phi4_force(ϕws,ϕ,ϕp,lp)
            mom .= mom .- Δτ/2 .* ϕws.frc
        
            return nothing
        end

        function HMC!(ϕ,  int::IntrScheme, lp::SpaceParm, ϕp::Phi4Parm, ϕws::Phi4workspace{T}; noacc=false) where T
            @timeit "HMC trajectory" begin
                # Copy gauge fields
                ϕws.ϕ .= ϕ

                # Initialize momenta
                randomize!(ϕws.mom,ϕws,ϕp,lp)

                # Calculate initial hamiltonian
                Hin = hamiltonian(ϕws.mom,ϕ,lp,ϕp,ϕws)

                # Perform molecular dynamics stes
                for _ in 1:int.ns MD!(ϕws.mom,ϕ,int.eps,lp,ϕp,ϕws) end

                # Perform metropolis
                ΔH = hamiltonian(ϕws.mom,ϕ,lp,ϕp,ϕws) - Hin
                pacc = exp(-ΔH)

                acc = true
                if (noacc)
                    return ΔH, acc
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
        # HMC!(ϕ,eps,ns,lp::SpaceParm,ϕp::Phi4Parm,ϕws::Phi4workspace{T}; noacc=false) where T = HMC!(ϕ,omf4(T,eps,ns),lp,ϕp,ϕws;noacc=noacc)
    ## ========================================================================================================



    export Phi4Parm, Phi4ParmM2L, Phi4workspace
    export randomize!
    export phi4_action, hopping
    export phi4_force
    export hamiltonian, HMC!
end

