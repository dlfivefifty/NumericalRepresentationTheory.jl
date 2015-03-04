module RepresentationTheory
    using Base

## Kronecker product of Sn

export perm,permkronpow,permmults,topart

function perm(a,b,n)
    ret=eye(n)
    ret[a,a]=ret[b,b]=0
    ret[a,b]=ret[b,a]=1
    ret
end

function kronpow(p,m)
    ret=p
    for k=1:m-1
        ret=kron(ret,p)
    end
    ret
end

permkronpow(m)=(a,b,n)->kronpow(perm(a,b,n),m)

## Algorithm

function wilkshift(A)
    if size(A,1)==1
        A[1,1]
    else
       a1,a,b=A[end-1,end-1],A[end,end],A[end-1,end]
        if a1==a==b==0
            0.0
        else
            δ=(a1-a)/2
            a-sign(δ)*b^2/(abs(δ)+sqrt(δ^2+b^2))
        end
    end
end


function deflateblock(v)
    tol=10E-8
    for m=1:size(v[1],1)-1
        allzero=true
        for vk in v
            if norm(vk[1:m,m+1:end],Inf)>tol
                allzero=false
                break
            end
        end

        if allzero
            w1=Array(Matrix{Float64},length(v))
            w2=Array(Matrix{Float64},length(v))
            for k=1:length(v)
                w1[k]=v[k][1:m,1:m]
                w2[k]=v[k][m+1:end,m+1:end]
            end
            return w1,w2
        end
    end

    # no subblocks
    v,Array(Matrix{Float64},0)
end

function deflate(v)
    if size(v[1],1)==1
        return v
    end

    ret=Array(Vector{Matrix{Float64}},size(v[1],1))

    for n=1:size(v[1],1)
        w1,v=deflateblock(v)
        ret[n]=w1
        if isempty(v)
            return ret[1:n]
        end
    end

    error("There's a bug: Should have less blocks then dimension of matrix")
end



function simdiagonalize(v::Vector{Matrix{Float64}})
    if size(v[1],1)==1
        return eye(1)
    end

    for k=length(v):-1:1
        if norm(v[k][1:end-1,end],Inf)>100eps()
            μ=wilkshift(v[k])
            Q=eigfact(Symmetric(v[k]-μ*I))[:vectors]
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
function simdiagonalize(v::Vector{Vector{Matrix{Float64}}})
    d=mapreduce(vk->size(vk[1],1),+,v)
    Q=zeros(d,d)
    m=1
    for vk in v
        Qk=simdiagonalize(vk)
        dk=size(Qk,1)
        Q[m:m+dk-1,m:m+dk-1]=Qk
        m+=dk
    end
    Q
end


function reducesn(perm,n)
    w=Array(Array{Float64,2},n-1)

    for k=1:n-1
        w[k]=perm(1,k+1,n)
        for j=2:k
            w[k]+=perm(j,k+1,n)
        end
    end

    simdiagonalize(deflate(w)),w
end


function permmults(perm,n)
    Q,v=reducesn(perm,n)
    int(hcat(map(v->diag(Q'*v*Q),v)...))
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
  ret=Array(Vector{Int64},size(m,1))
  for k=1:size(m,1)
    ret[k]=topart([0;vec(m[k,:])])
  end
  ret
end
end #module


