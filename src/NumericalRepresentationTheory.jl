module NumericalRepresentationTheory
using Base, LinearAlgebra, Permutations, SparseArrays, BlockBandedMatrices, BlockArrays, FillArrays
import Base: getindex, size, setindex!, maximum, Int, length,
                ==, isless, copy, kron, hash, first, show, lastindex, |, Integer, BigInt

import LinearAlgebra: adjoint, transpose, eigen
import SparseArrays: blockdiag, AbstractSparseMatrixCSC
## Kronecker product of Sn



export Partition, YoungMatrix, partitions, youngtableaux, YoungTableau, ⊗, ⊕,
        Representation, multiplicities, generators, standardrepresentation, randpartition,
        blockdiagonalize, hooklength


# utility function
function kronpow(p,m)
    ret = p
    for k=1:m-1
        ret = kron(ret,p)
    end
    ret
end


struct Partition
    σ::Vector{Int}
    function Partition(σ)
        if !issorted(σ; lt=Base.:>)
            error("input vector $σ should be sorted")
        end
        if !all(x -> x > 0, σ)
            error("input vector $σ should be all positive")
        end
        new(σ)
    end
end

Partition(σ::Int...) = Partition([σ...])

function isless(a::Partition, b::Partition)
    n,m = Int(a), Int(b)
    if n ≠ m
        return n < m
    end
    
    M,N = length(a.σ), length(b.σ)
    for k = 1:min(M,N)
        if a.σ[k] ≠ b.σ[k] 
            return a.σ[k] < b.σ[k]
        end
    end

    return false
end
(==)(a::Partition, b::Partition) = a.σ == b.σ
hash(a::Partition) = hash(a.σ)

copy(a::Partition) = Partition(copy(a.σ))

lastindex(a::Partition) = lastindex(a.σ)

function show(io::IO, σ::Partition)
    print(io, "$(Int(σ)) = ")
    for k = 1:length(σ)-1
        print(io, "$(σ[k]) + ")
    end
    print(io, "$(σ[end])")
end

function transpose(σ::Partition)
    cols = Vector{Int}(undef, maximum(σ))
    j_end = 1
    for k = length(σ):-1:1
        for j = j_end:σ[k]
            cols[j] = k
        end
        j_end = σ[k]+1
    end
    Partition(cols)
end
adjoint(σ::Partition) = transpose(σ)
getindex(σ::Partition, k::Int) = σ.σ[k]
setindex!(σ::Partition, v, k::Int) = setindex!(σ.σ, v, k)
for op in (:maximum, :length, :first)
    @eval $op(σ::Partition) = $op(σ.σ)
end

Int(σ::Partition) = sum(σ.σ)
BigInt(σ::Partition) = BigInt(Int(σ))
Integer(σ::Partition) = Int(σ)


function partitions(N)
    N == 1 && return [Partition([1])]
    ret = Partition[]
    part = partitions(N-1)
    for p in part
        if (length(p.σ) < 2 || p.σ[end-1] > p.σ[end])
            p_new = copy(p)
            p_new.σ[end] += 1
            push!(ret, p_new)
        end

        push!(ret, Partition([p.σ ; 1]))
    end
    ret
end



struct YoungMatrix <: AbstractMatrix{Int}
    data::Matrix{Int}
    rows::Partition
    columns::Partition

    function YoungMatrix(data, rows, columns)
        @assert size(data) == (length(rows), length(columns))
        @assert rows == columns'
        new(data, rows, columns)
    end
end

YoungMatrix(data::Matrix{Int}, σ::Partition) = YoungMatrix(data, σ, σ')
YoungMatrix(::UndefInitializer, σ::Partition) = YoungMatrix(zeros(Int, length(σ), maximum(σ)), σ)

YoungMatrix(dat, σ::Vector{Int}) = YoungMatrix(dat, Partition(σ))

copy(Y::YoungMatrix) = YoungMatrix(copy(Y.data), Y.rows, Y.columns)

size(Y::YoungMatrix) = size(Y.data)
getindex(Y::YoungMatrix, k::Int, j::Int) = ifelse(k ≤ Y.columns[j] && j ≤ Y.rows[k], Y.data[k,j], 0)
function setindex!(Y::YoungMatrix, v, k::Int, j::Int)
    @assert k ≤ Y.columns[j] && j ≤ Y.rows[k]
    Y.data[k,j] = v
end

hash(Y::YoungMatrix) = hash(Y.data)


struct YoungTableau
	partitions::Vector{Partition}
end

function isless(a::YoungTableau, b::YoungTableau)
    if length(a.partitions) ≠ length(b.partitions)
        return length(a.partitions) < length(b.partitions)
    end
    for (p,q) in zip(a.partitions, b.partitions)
        p ≠ q && return isless(p,q)
    end
    return false
end

function YoungMatrix(Yt::YoungTableau)
	ps = Yt.partitions

	Y = YoungMatrix(undef, ps[end])
	Y[1,1] = 1

	for k=2:length(ps)
		if length(ps[k]) > length(ps[k-1])
			Y[length(ps[k]),1] = k
		else
			for j = 1:length(ps[k])
				if ps[k][j] > ps[k-1][j]
					Y[j,ps[k][j]] = k
					break
				end
			end
		end
	end
	Y
end

function lowerpartitions(σ)
	p = Partition[]
	n = length(σ)

	if σ[n] > 1
		p_new = copy(σ)
		p_new[n] -= 1
		push!(p, p_new)
	else
		push!(p, Partition(σ.σ[1:n-1]))
	end

	for k = n-1:-1:1
		if σ[k] > σ[k+1]
			p_new = copy(σ)
			p_new[k] -= 1
		 	push!(p, p_new)
		end
	end

	p
end


function youngtableaux(σ::Partition)
	σ == Partition([1]) && return [YoungTableau([σ])]
	Yts = mapreduce(youngtableaux, vcat, lowerpartitions(σ))
	sort!(map(Yt -> YoungTableau([Yt.partitions; σ]), Yts))
end


# function isyoungtableau(Y::YoungMatrix)
#     Y[1,1] == 1 || return false
#     for j=1:length(Y.cols), k=1:Y.cols[j]
#
# end

function hooklength(σ::Partition)
    ret = BigInt(1)
    m = length(σ)
    for k = 1:m, j=1:σ[k]
        ret_2 = 0
        ret_2 += σ[k]-j
        for p = k:m
            σ[p] < j && break
            ret_2 += 1
        end
        ret *= ret_2
    end
    factorial(BigInt(σ)) ÷ ret
end

# sample by Plancherel
function randpartition(N, m)
    Σ = partitions(N)
    l = hooklength.(Σ)

    cs = cumsum(l)

    rs = rand(1:cs[end], m)
    p = (r -> findfirst(k -> k ≥ r, cs)).(rs)
    Σ[p]
end

randpartition(N) = randpartition(N, 1)[1]

function irrepgenerator(σ::Partition, i::Int)
    Is = Int[]; Js = Int[]; Vs = Float64[]
    t = YoungMatrix.(youngtableaux(σ))
    t_lookup = Dict(tuple.(t, 1:length(t)))
    d = length(t)

    for j = 1:d
        Y = t[j]
        ii = Base._to_subscript_indices(Y, Tuple(findfirst(isequal(i), Y))...)
        ip = Base._to_subscript_indices(Y, Tuple(findfirst(isequal(i+1), Y))...)

        if ii[1] == ip[1]
            push!(Is, j)
            push!(Js, j)
            push!(Vs, 1)
        elseif ii[2] == ip[2]
            push!(Is, j)
            push!(Js, j)
            push!(Vs, -1)
        else
            Yt = copy(Y) # Tableau with swapped indices
            Yt[ii...], Yt[ip...] = (Yt[ip...], Yt[ii...])
            k = t_lookup[Yt]
            # set entries to matrix [1/r sqrt(1-1/r^2); sqrt(1-1/r^2) -1/r]
            push!(Is, j, j)
            push!(Js, j, k)
            ai = ii[2]-ii[1]
            ap = ip[2]-ip[1]
            r = ap - ai
            push!(Vs, 1/r, sqrt(1-1/r^2))
        end
    end
    sparse(Is, Js, Vs)
end

irrepgenerators(σ::Partition) = [irrepgenerator(σ, i) for i=1:Int(σ)-1]


"""
    Representation([τ₁,…,τₙ])

is a representation of the symmetric group generated by τ₁,…,τₙ.
"""
struct Representation{MT}
       generators::Vector{MT}
end


"""
    Representation(λ₁::Int, …, λₙ::Int)
    Representation(Partition(λ₁, …, λₙ))

both generate the irreducible representation associated with the given partition.
"""
Representation(σ::Int...) = Representation(Partition(σ...))
Representation(σ::Partition) = Representation(irrepgenerators(σ))
Representation{MT}(ρ::Representation) where MT = Representation(convert.(MT, ρ.generators))


"""
    Representation(::Dict{Partition,Int})

makes a representation corresponding to a direct sum of the irreducible representations encoded in
by the partitions. The partitions are sorted for a consistent ordering.
"""
function Representation(d::Dict{Partition,Int})
    kys = sort(collect(keys(d)))

    reps = Representation{SparseMatrixCSC{Float64,Int}}[]
    for k in kys
        irrep = Representation(k)
        for j in 1:d[k]
            push!(reps, irrep)
        end
    end
    blockdiag(reps...)
end

kron(A::Representation, B::Representation) = Representation(kron.(A.generators, B.generators))

_blockdiag(A::AbstractSparseMatrixCSC...) = blockdiag(A...)
_blockdiag(A::AbstractMatrix...) = blockdiag(map(sparse, A)...)
blockdiag(A::Representation...) = Representation(_blockdiag.(getfield.(A, :generators)...))


conjugate(ρ::Representation, Q) = Representation(map(g -> Q'*g*Q, ρ.generators))


⊗(A::Representation, B::Representation) = kron(A, B)

|(A::Representation, n::AbstractVector) = Representation(A.generators[n])

generators(R::Representation) = R.generators
size(R::Representation, k) = size(R.generators[1], k)
size(R::Representation) = size(R.generators[1])

diagm(A::Vector{<:Representation}) = Representation(blockdiag.(generators.(A)...))
⊕(A::Representation...) = Representation(blockdiag.(generators.(A)...))


function (R::Representation)(P)
    if isempty(CoxeterDecomposition(P).terms)
        # Identity
        one(first(R.generators))
    else
        *(map(i -> R.generators[i], CoxeterDecomposition(P).terms)...)
    end
end

# determine multiplicities of eigs on diagonal, assuming sorted
function eigmults(λ::Vector{Int})
       mults = Vector{Int}()
       n = length(λ)
       tol = 0.01 # integer coefficents
       c = 1
       for k = 2:n
           if λ[k] > λ[k-1]
              push!(mults, c)
              c = 0
           end
           c += 1
       end
       push!(mults, c)
       mults
end


function gelfandbasis(gen::Vector{MT}) where MT<:Matrix
    n = length(gen)+1
    w = Vector{MT}(undef, n-1)
    tmp = similar(gen[1])
    a = similar(tmp)
    for k = 1:n-1
        a .= gen[k]
        w[k] = copy(a)
        for j = k-1:-1:1
            mul!(tmp, a, gen[j])
            mul!(a, gen[j], tmp)
            w[k] .+= a
        end
    end
    w
end

function gelfandbasis(gen::Vector{MT}) where MT
    n = length(gen)+1
    w = Vector{MT}(undef, n-1)
    for k = 1:n-1
        a = gen[k]
        w[k] = a
        for j = k-1:-1:1
            a = gen[j]*a*gen[j]
            w[k] += a
        end
    end
    w
end

gelfand_reduce(R::Representation) = gelfand_reduce(convert.(Matrix, gelfandbasis(R.generators)))

function sortcontentsperm(Λ)
    ps = contents2partition(Λ)
    perm = Int[]


    for pₖ in union(ps)
        dₖ = hooklength(pₖ)
        inds = findall(==(pₖ), ps)
        aₖ = length(inds) ÷ dₖ # number of times repeated
        for j = 1:aₖ
            append!(perm, inds[j:aₖ:end])
        end
    end

    perm
end

function gelfand_reduce(X)
       λ̃, Q₁ = eigen(Symmetric(X[1]))
       λ = round.(Int, λ̃)
       if !(Q₁'Q₁ ≈ I)
            error("The eigenvalue decomposition has failed")
       end
       if !(isapprox(λ, λ̃; atol=1E-7))
           error("$λ̃ are not all approximately an integer")
       end
       length(X) == 1 && return reshape(λ,length(λ),1),Q₁

       m = eigmults(λ)
       c_m = [0;cumsum(m)]

       Q = zero(Q₁)
       Λ = Matrix{Int}(undef, length(λ), length(X))
       Λ[:,1] = λ

       for j=2:length(c_m)
              j_inds = c_m[j-1]+1:c_m[j]
              Qʲ = Q₁[:,j_inds] # deflated rows
              Xⱼ = map(X -> Qʲ'*X*Qʲ, X[2:end])
              Λⱼ, Qⱼ = gelfand_reduce(Xⱼ)
              Q[j_inds,j_inds] = Qⱼ
              Λ[j_inds,2:end] = Λⱼ
       end

       Λ, Q₁*Q
end

"""
    singlemultreduce(ρ)

reduces a representation containing only a single irrep, multiple times.
"""

function singlemultreduce(ρ)
    m = multiplicities(ρ)
    @assert length(m) == 1
    singlemultreduce(ρ, Representation(first(keys(m))))
end

"""
    singlemultreduce(ρ, σ)

reduces a representation containing only a single irrep `σ`, multiple times.
"""
function singlemultreduce(ρ, σ)
    m = size(σ,1)
    n = size(ρ,1)
    A = vcat((kron.(Ref(I(m)), ρ.generators) .- kron.(σ.generators, Ref(I(n))))...)
    Q̃ = nullspace(convert(Matrix,A); atol=1E-10)*sqrt(m)
    reshape(vec(Q̃), n, n)
end


# Creates vcat((kron.(Ref(I(m)), ρ.generators) .- kron.(σ.generators, Ref(I(ℓ))))...);
# but with the rows and columns corresponding to zero entries removed 
function singlemultreducedkron(ρ, σ)
    tol = 1E-10
    m = size(σ,1)
    ℓ = size(ρ,1)
    μ = ℓ ÷ m
    n = length(σ.generators)+1
    B = zeros(m*(n-1)*ℓ, m*μ)
    for κ=1:m, j= 1:μ, k = 1:n-1
        B[range((k-1)*m*ℓ + (κ-1)*ℓ + 1; length=ℓ),(κ-1)*μ+j] = ρ.generators[k][:,(j-1)*m+κ]
        B[range((k-1)*m*ℓ + (j-1)*m + κ; step=ℓ, length=m),(κ-1)*μ+j] -= σ.generators[k][:,κ]
    end

    # determine non-zero rows of A[:,jr]
    kr = Int[]
    for k = 1:size(B,1)
        if maximum(abs,view(B,k,:)) > tol
            push!(kr, k)
        end
    end
    B[kr,:]
end


"""
    singlemultreduce_blockdiag(ρ, σ)

reduces a representation containing only a single irrep `σ`, multiple times.
Unlike `singlemultreduce`, it is assumed that `ρ` comes from `gelfand_reduce` and hence
the returned `Q` is guaranteed to be block-diagonal.
"""
function singlemultreduce_blockdiag(ρ, σ)
    tol = 1E-10
    m = size(σ,1)
    ℓ = size(ρ,1)
    μ = ℓ ÷ m
    
    # determine columns of `A` corresponding to non-zero entries of `Q`
    jr = Int[]
    for k = 1:m
        append!(jr, range((k-1)*μ*m+k; step=m, length=μ))
    end


    Aₙ = singlemultreducedkron(ρ, σ)
    V = svd(Aₙ).V[:,end-μ+1:end]*sqrt(m) # nullspace  corresponds to last μ singular vectors


    # now populate non-zero entries of `Q`
    Q = BandedBlockBandedMatrix{Float64}(undef, Fill(m,μ), Fill(m,μ), (ℓ-1,ℓ-1), (0,0))
    for j = 1:size(V,2)
        sh = (j-1) * size(V,1)
        for k = 1:size(V,1)
            view(Q,:,Block(j))[jr[k]] = V[k,j]
        end
    end

    Q
end
    


function blockdiagonalize(ρ::Representation)
    Λ,Q = gelfand_reduce(ρ)
    p = sortcontentsperm(Λ)
    Q = Q[:,p]
    Λ = Λ[p,:]
    n = length(ρ.generators)+1
    
    Q̃ = similar(Q)
    # diagonalised generators    
    c = contents2partition(Λ)

    k = 0
    for pⱼ in union(c)
        j = findall(isequal(pⱼ), c)
        Qⱼ = Q[:,j]
        ρⱼ = conjugate(ρ, Qⱼ)
        Q̃ⱼ = singlemultreduce_blockdiag(ρⱼ, Representation(pⱼ))
        m = length(j)
        Q̃[:,k+1:k+m] = Qⱼ * Q̃ⱼ
        k += m
    end

    multiplicities(c), Q̃
end

function contents2partition(part::Vector)
    part = sort(part)
    p = zeros(Int, 1-first(part))
    k = 1
    while k ≤ length(part)
        pₖ = part[k]
        rₖ = pₖ < 0 ? 1-pₖ : 1
        p[rₖ] += 1
        k += 1
        while k ≤ length(part) && part[k] == pₖ
            k += 1
            rₖ += 1
            p[rₖ] += 1
        end
    end
    Partition(p)
end

function contents2partition(m::Matrix{Int})
    ret = Vector{Partition}(undef, size(m,1))
    for k=1:size(m,1)
        ret[k] = contents2partition([0;vec(m[k,:])])
    end
    ret
end

# these are the multiplicities without dividing by dimension
function _multiplicities(parts::Vector{Partition})
    dict = Dict{Partition,Int64}()

    for part in parts
        if !haskey(dict,part)
            dict[part] = 0
        end
        dict[part] += 1
    end
    dict
end

function multiplicities(Λ::Vector{Partition})
    mults = _multiplicities(Λ)

    for σ in keys(mults)
           mults[σ] = mults[σ] ÷ hooklength(σ)
    end

    mults
end

multiplicities(R::Representation) = multiplicities(contents2partition(gelfand_reduce(R)[1]))



## Representations


function perm(a,b,n)
    ret = Matrix(I,n,n)
    ret[a,a] = ret[b,b] = 0
    ret[a,b] = ret[b,a] = 1
    ret
end

standardrepresentation(n) = Representation(Matrix{Float64}[perm(k,k+1,n) for k=1:n-1])

include("canonicalprojection.jl")


end #module
