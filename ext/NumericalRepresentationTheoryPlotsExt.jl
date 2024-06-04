module NumericalRepresentationTheoryPlotsExt
using NumericalRepresentationTheory, Plots
import Plots: plot, @recipe

@recipe function f(σ::Partition)
    legend --> false
    ratio --> 1.0
    axis --> false
    grid --> false
    color --> :orange
    ticks --> false
    linewidth --> 2

    ret = Shape[]
    m = length(σ)
    for j = 1:m, k = 1:σ[j]
        push!(ret, Shape([k-1,k-1,k,k],[1-j,-j,-j,1-j]))
    end
    ret
end


function plot(mults::Dict{Partition,<:Integer}; kwds...)
    ret = Any[]
    M = mapreduce(maximum, max, keys(mults))
    N = mapreduce(length, max, keys(mults))
    for (σ,m) in sort(mults)
        push!(ret, plot(σ; title="$m", xlims=(0,M), ylims=(-N,0)))
    end
    plot(ret...; kwds...)
end


end