module RepresentationTheory
using Base, Compose, Compat, Permutations
import Base: ctranspose, transpose, getindex, size, setindex!, maximum, Int, length,
                ==, isless, copy, kron, eig, hash, first
## Kronecker product of Sn



export perm, permkronpow, permmults, topart, plotmults,
    standardgenerators, Partition, YoungMatrix, partitions,
    youngtableaux, YoungTableau, ⊗, ⊕, Representation, multiplicities, generators



function kronpow(p,m)
    ret = p
    for k=1:m-1
        ret = kron(ret,p)
    end
    ret
end

permkronpow(m) = (a,b,n)->kronpow(perm(a,b,n),m)

## Algorithm
function gelfandbasis(gen::Vector{MT}) where MT
    n = length(gen)+1
    w = Vector{MT}(n-1)
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

struct Partition
    σ::Vector{Int}
    function Partition(σ)
        @assert issorted(σ; lt=Base.:>)
        @assert all(x -> x > 0, σ)
        new(σ)
    end
end


function isless(a::Partition, b::Partition)
    n,m = Int(a), Int(b)
    n < m && return true
    if n == m
        for k = 1:min(length(a.σ), length(b.σ))
            a.σ[k] > b.σ[k] && return true
            a.σ[k] < b.σ[k] && return false
        end
    end

    return false
end
(==)(a::Partition, b::Partition) = a.σ == b.σ
hash(a::Partition) = hash(a.σ)

copy(a::Partition) = Partition(copy(a.σ))

function transpose(σ::Partition)
    cols = Vector{Int}(uninitialized, maximum(σ))
    j_end = 1
    for k = length(σ):-1:1
        for j = j_end:σ[k]
            cols[j] = k
        end
        j_end = σ[k]+1
    end
    Partition(cols)
end
ctranspose(σ::Partition) = transpose(σ)
getindex(σ::Partition, k::Int) = σ.σ[k]
setindex!(σ::Partition, v, k::Int) = setindex!(σ.σ, v, k)
for op in (:maximum, :length, :first)
    @eval $op(σ::Partition) = $op(σ.σ)
end

Int(σ::Partition) = sum(σ.σ)

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
YoungMatrix(::Uninitialized, σ::Partition) = YoungMatrix(Matrix{Int}(uninitialized, length(σ), maximum(σ)), σ)

YoungMatrix(dat, σ::Vector{Int}) = YoungMatrix(dat, Partition(σ))

copy(Y::YoungMatrix) = YoungMatrix(copy(Y.data), Y.rows, Y.columns)

size(Y::YoungMatrix) = size(Y.data)
getindex(Y::YoungMatrix, k::Int, j::Int) = ifelse(k ≤ Y.columns[j] && j ≤ Y.rows[k], Y.data[k,j], 0)
function setindex!(Y::YoungMatrix, v, k::Int, j::Int)
    @assert k ≤ Y.columns[j] && j ≤ Y.rows[k]
    Y.data[k,j] = v
end



struct YoungTableau
	partitions::Vector{Partition}
end

function YoungMatrix(Yt::YoungTableau)
	ps = Yt.partitions

	Y = YoungMatrix(uninitialized, ps[end])
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
	map(Yt -> YoungTableau([Yt.partitions; σ]), Yts)
end


# function isyoungtableau(Y::YoungMatrix)
#     Y[1,1] == 1 || return false
#     for j=1:length(Y.cols), k=1:Y.cols[j]
#
# end

function hooklength(σ::Partition)
    ret = 1
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
    factorial(Int(σ)) ÷ ret
end



function irrepgenerator(σ::Partition, i::Int)
    Is = Int[]; Js = Int[]; Vs = Float64[]
    t = YoungMatrix.(youngtableaux(σ))
    d = length(t)

    for j = 1:d
        Y = t[j]
        ii = Base._to_subscript_indices(Y, findfirst(Y, i))
        ip = Base._to_subscript_indices(Y, findfirst(Y, i+1))

        if ii[1] == ip[1]
            push!(Is, j)
            push!(Js, j)
            push!(Vs, 1)
        elseif ii[2] == ip[2]
            push!(Is, j)
            push!(Js, j)
            push!(Vs, -1)
        else
            ai = ii[2]-ii[1]
            ap = ip[2]-ip[1]
            r = ap - ai
            Yt = copy(Y) # Tableau with swapped indices
            Yt[ii...], Yt[ip...] = (Yt[ip...], Yt[ii...])
            k = findfirst(t, Yt)
            # set entries to matrix [1/r sqrt(1-1/r^2); sqrt(1-1/r^2) -1/r]
            push!(Is, j, j)
            push!(Js, j, k)
            push!(Vs, 1/r, sqrt(1-1/r^2))
        end
    end
    sparse(Is, Js, Vs)
end

irrepgenerators(σ::Partition) = [irrepgenerator(σ, i) for i=1:Int(σ)-1]



struct Representation{MT}
       generators::Vector{MT}
end

Representation(σ::Partition) = Representation(irrepgenerators(σ))
kron(A::Representation, B::Representation) = Representation(kron.(A.generators, B.generators))
⊗(A::Representation, B::Representation) = kron(A, B)

generators(R::Representation) = R.generators
size(R::Representation, k) = size(R.generators[1], k)
size(R::Representation) = size(R.generators[1])

diagm(A::Vector{<:Representation}) = Representation(blkdiag.(generators.(A)...))
⊕(A::Representation...) = Representation(blkdiag.(generators.(A)...))


(R::Representation)(P::Permutation) = *(map(sᵢ -> R.generators[sᵢ.i], CoxeterDecomposition(P))...)

# determine multiplicities of eigs on diagonal, assuming sorted
function eigmults(λ::Vector{Int})
       mults = Vector{Int}(uninitialized)
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

gelfand_reduce(R::Representation) = gelfand_reduce(Matrix.(gelfandbasis(R.generators)))

function gelfand_reduce(X)
       λ̃, Q₁ = eig(Symmetric(X[1]))
       λ = round.(Int, λ̃)
       length(X) == 1 && return reshape(λ,length(λ),1),Q₁

       m = eigmults(λ)
       c_m = [0;cumsum(m)]

       Q = zeros(Q₁)
       Λ = Matrix{Int}(uninitialized, length(λ), length(X))
       Λ[:,1] = λ

       for j=2:length(c_m)
              j_inds = c_m[j-1]+1:c_m[j]
              Qʲ = Q₁[:,j_inds] # deflated rows
              Xⱼ = map(X -> ctranspose(Qʲ)*X*Qʲ, X[2:end])
              Λⱼ, Qⱼ = gelfand_reduce(Xⱼ)
              Q[j_inds,j_inds] = Qⱼ
              Λ[j_inds,2:end] = Λⱼ
       end

       Λ, Q₁*Q
end

function topart(part::Vector)
    part=sort(part)
    p=zeros(Int64,1-first(part))
    k=1
    while k≤length(part)
        pk=part[k]
        rk=pk<0?1-pk:1
        p[rk]+=1
        k+=1
        while k≤length(part)&&part[k]==pk
            k+=1
            rk+=1
            p[rk]+=1
        end
    end
    Partition(p)
end

function topart(m::Matrix{Int})
    ret = Vector{Partition}(size(m,1))
    for k=1:size(m,1)
        ret[k] = topart([0;vec(m[k,:])])
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

multiplicities(R::Representation) = multiplicities(topart(gelfand_reduce(R)[1]))



## Representations


function perm(a,b,n)
    ret = eye(n)
    ret[a,a] = ret[b,b] = 0
    ret[a,b] = ret[b,a] = 1
    ret
end

standardrepresentation(n) = Representation(Matrix{Float64}[perm(k,k+1,n) for k=1:n-1])


## Plotting

function plotpart(part::Partition)
    ε=0.01
    x,y=part[1],length(part)
    box(ε,k,j,x,y)=rectangle((k-1)/x+ε,(j-1)/y+ε,1/x-ε,1/y-ε)
    cnt=compose(context())

    for k=1:length(part),j=1:part[k]
        cnt=compose(cnt,box(ε,j,k,x,y))
    end

    compose(cnt,fill("tomato"),stroke("black"))
end



function plotmults(dict::Dict)
    cnt = compose(context());k=0
    kys = keys(dict)
    ml = mapreduce(length,max,kys)
    nl = mapreduce(first,max,kys)
    ε=0.05
    m=ceil(Integer,sqrt(length(kys)))
    for ky in kys
        j,i=div(k,m),mod(k,m)
        compose!(cnt,(context(i/m+ε/m,j/m,first(ky)/(nl*m)-ε/m,0.5*length(ky)/(nl*ml)),RepresentationTheory.plotpart(ky)),
                 (context(),text(i/m*(1+ε)+0.2/m,j/m+(0.6+1/ml)/nl,string(dict[ky])),fontsize(4)))
        k+=1
    end
    cnt
end


plotmults(R::Representation) = plotmults(multiplicities(R))

end #module
