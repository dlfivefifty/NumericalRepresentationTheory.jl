module RepresentationTheory
using Base, Compose, Compat
import Base: ctranspose, transpose, getindex, size, setindex!, maximum, Int, length,
                ==, isless, copy
## Kronecker product of Sn

export perm, permkronpow, permmults, topart, plotmults,
    irrepgenerators, standardgenerators, Partition, YoungMatrix, partitions,
    youngtableaux, YoungTableau

function perm(a,b,n)
    ret = eye(n)
    ret[a,a] = ret[b,b] = 0
    ret[a,b] = ret[b,a] = 1
    ret
end

standardgenerators(n) = Matrix{Float64}[perm(k,k+1,n) for k=1:n-1]


function kronpow(p,m)
    ret=p
    for k=1:m-1
        ret=kron(ret,p)
    end
    ret
end

permkronpow(m) = (a,b,n)->kronpow(perm(a,b,n),m)

## Algorithm

function wilkshift(A)
    if size(A,1)==1
        A[1,1]
    else
        a1,a,b=A[end-1,end-1],A[end,end],A[end-1,end]
        if a1 == a == b == 0
            0.0
        else
            δ=(a1-a)/2
            a-sign(δ)*b^2/(abs(δ)+sqrt(δ^2+b^2))
        end
    end
end


function slnorm(m,kr,jr)
    ret=0.0
    for j=jr
        for k=kr
            @inbounds ret=ret+abs2(m[k,j])
        end
    end
    ret
end

function deflateblock(v)
    tol=10E-8
    for m=1:size(v[1],1)-1
        allzero=true
        for vk in v
            if slnorm(vk,m+1:size(vk,2),1:m) >tol
                allzero=false
                break
            end
        end

        if allzero
            w1 = Vector{Matrix{Complex128}}(length(v))
            w2 = Vector{Matrix{Complex128}}(length(v))
            for k = 1:length(v)
                w1[k] = v[k][1:m,1:m]
                w2[k] = v[k][m+1:end,m+1:end]
            end
            return w1,w2
        end
    end

    # no subblocks
    v,Vector{Matrix{Complex128}}(0)
end

function deflate(v)
    if size(v[1],1)==1
        return v
    end

    ret=Vector{Vector{Matrix{Complex128}}}(size(v[1],1))

    for n=1:size(v[1],1)
        w1,v=deflateblock(v)
        ret[n]=w1
        if isempty(v)
            return ret[1:n]
        end
    end

    error("There's a bug: Should have less blocks then dimension of matrix")
end


function nullQ(w,μ)
#    @assert imag(μ)<1000000eps()
    svd(w-μ*I)[end][:,end:-1:1]
end
nullQ(M)=nullQ(M,eigvals(M)|>first)


# returns Q
function simdiagonalize{T}(v::Vector{Matrix{T}})
    if size(v[1],1)==1
        return eye(1)
    end

    for k=1:length(v)
        if slnorm(v[k],2:size(v[k],1),1)>100000eps()
            Q=nullQ(v[k])
            df=deflate(map(v->Q'*v*Q,v))
            if length(df)>1
                Q2=simdiagonalize(df)
                return Q*Q2
            end
        end
    end

    ## Already diagonalized
    warn("Matrix not deflated.  Returning current decomposition.")
    eye(size(v[1],1))
end

# a list of each blocks
# returns Q
function simdiagonalize{T}(v::Vector{Vector{Matrix{T}}})
    d=mapreduce(vk->size(vk[1],1),+,v)
    Q=zeros(Complex{Float64},d,d)
    m=1
    for j= 1:length(v)
        Qk=simdiagonalize(v[j])
        dk=size(Qk,1)
        Q[m:m+dk-1,m:m+dk-1]=Qk
        m+=dk
    end
    Q
end

function gelfandbasis(perm,n)
    w=Array(Array{Float64,2},n-1)

    for k=1:n-1
        w[k]=perm(1,k+1,n)
        for j=2:k
            w[k]+=perm(j,k+1,n)
        end
    end
    w
end

function gelfandbasis(gen)
    n=length(gen)+1
    w=Vector{Matrix{Float64}}(n-1)
    for k=1:n-1
        a=gen[k]
        w[k]=a
        for j=k-1:-1:1
            a=gen[j]*a*gen[j]
            w[k]+=a
        end
    end
    w
end


function reducesn(perm...)
    w=gelfandbasis(perm...)
    simdiagonalize(deflate(w)),w
end





function permmults(perm...)
    Q,v=reducesn(perm...)
    map(x->round(Int,real(x)),hcat(map(v->diag(Q'*v*Q),v)...))
end




function partmults(parts)
    dict=Dict{Vector{Int64},Int64}()

    for part in parts
        if !haskey(dict,part)
            dict[part]=0
        end
        dict[part]+=1
    end
    dict
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
    p
end

function topart(m::Matrix)
    ret=Vector{Vector{Int64}}(size(m,1))
    for k=1:size(m,1)
        ret[k]=topart([0;vec(m[k,:])])
    end
    ret
end



## Plotting

function plotpart(part)
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
    cnt=compose(context());k=0
    kys=keys(dict)
    ml=mapreduce(length,max,kys)
    nl=mapreduce(first,max,kys)
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


function plotmults(perm...)
    mults=permmults(perm...)
    plotmults(partmults(topart(mults)))
end



function irrepgenerators(part)
    run(`/Applications/SageMath-8.1.app/sage $(Pkg.dir())/RepresentationTheory/exportgenerators.py $part`)
    n=sum(part)
    ret=Array{Matrix{Float64}}(n-1)
    for k=1:n-1
       ret[k]=readcsv("/tmp/gen"*string(k)*".csv")
    end
    ret
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
Base.:(==)(a::Partition, b::Partition) = a.σ == b.σ

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
for op in (:maximum, :length)
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




end #module
