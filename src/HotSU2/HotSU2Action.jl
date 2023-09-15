

function krnl_action!(act, U::AbstractArray{TG}, Sigma::AbstractArray{TS}, Pi::AbstractArray{TP}, sp::HotSU2Param{T}, lp::SpaceParm{N,M,B,D}) where {TG,TS,TP,T,N,M,B,D}

    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)

    S = zero(eltype(act))

    # kinetic term Σ
    Sig2 = Sigma[b,r] * Sigma[b,r] 
    S += Sig2
    for id in 1:N
        bu, ru = up((b,r), id, lp)
        S -= Sigma[bu,ru] * Sigma[b,r]
    end

    # kinetic term Π

    for id in 1:N
        bu, ru = up((b,r), id, lp)
        

    end



end