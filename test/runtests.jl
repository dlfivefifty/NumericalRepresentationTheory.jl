using RepresentationTheory, Test
import RepresentationTheory: gelfandbasis

σ = Partition([3,3,2,1])
@test length(youngtableaux(σ)) == hooklength(σ)
@test all(isdiag,gelfandbasis(Representation(σ).generators)[3])

@test multiplicities(standardrepresentation(4))[Partition([3,1])] == 1


s = standardrepresentation(3)
ρ = s ⊗ s
@test multiplicities(ρ)[Partition([2,1])] == 3


ρ = Representation(Partition([3,2,1])) ⊗ Representation(Partition([2,2,2]))
@test multiplicities(ρ)[Partition([3,2,1])] == 2


R = ρ = Representation(Partition([3,2,1])) ⊗ Representation(Partition([2,2,2])) ⊗ Representation(Partition([3,1,1,1]))
multiplicities(ρ)

Λ,Q = gelfand_reduce(R)

Q̃ = similar(Q)

c = contents2partition(Λ)

k = 0
for pⱼ in partitions(6)
    j = findall(isequal(pⱼ), c)
    if !isempty(j)
        @show pⱼ
        Qⱼ = Q[:,j]
        ρⱼ = Representation(map(g -> Qⱼ'*g*Qⱼ, R.generators))
        Q̃ⱼ = singlemultreduce(ρⱼ)
        m = length(j)
        Q̃[:,k+1:k+m] = Qⱼ * Q̃ⱼ
        k += m
    end
end

(Qⱼ * Q̃ⱼ)' * R.generators[4] * Qⱼ * Q̃ⱼ

m = multiplicities(ρⱼ)

function singlemultreduce(ρ)
    m = multiplicities(ρ)
    @assert length(m) == 1
    singlemultreduce(ρ, Representation(first(keys(m))))
end

function singlemultreduce(ρ, σ)
    m = size(σ,1)
    n = size(ρ,1)
    A = vcat((kron.(Ref(I(m)), ρ.generators) .- kron.(σ.generators, Ref(I(n))))...)
    Q̃ = nullspace(convert(Matrix,A);atol=1E-10)*sqrt(m)
    reshape(vec(Q̃), n, n)
end
    



Q̄'*ρ.generators[4]*Q̄


reshape(Q̃[:,2],n,m)'*(reshape(Q̃[:,2],n,m))



.generators



σ  = sortperm(c)
Q = Q[:,σ]



Q'*R.generators[3]*Q

σ = Partition([3,3,1])
ρ = Representation(σ)
Q = qr(randn(size(ρ))).Q

ρ̃ = Representation(map(g -> Q'g*Q, ρ.generators))

blkdiagonalize(ρ̃)[2]

Q = qr(randn(2 .* size(ρ))).Q
ρ̃ = Representation(map(g -> Q'*Float64[g zero(g); zero(g) g]*Q, ρ.generators))

@test multiplicities(ρ̃)[σ] == 2

_,Q = blkdiagonalize(ρ̃)


d, Q = blkdiagonalize(ρ)
s_1 = d.generators[1][2:end-2,2:end-2]
s_2 = d.generators[2][2:end-2,2:end-2]

s_1

σ = Representation(Partition([2,1]))
σ_1,σ_2 = σ.generators
m = size(σ_1,1)
n = size(s_1,1)

Qs = nullspace([kron(I(m), s_1) - kron(σ_1, I(n)); kron(I(m), s_2) - kron(σ_2,I(n))])

Q̃ = sqrt(2) * [reshape(Qs[:,1],n,m) reshape(Qs[:,2],n,m) reshape(Qs[:,3],n,m)]

Q̃' * s_1 * Q̃
Q̃' * s_2 * Q̃

s_1*reshape(Qs[:,1],n,m) - reshape(Qs[:,1],n,m) * σ_1

(kron(s_1,I(m)) - kron(I(n),σ_1)) * Qs[:,1] |> norm

reshape(kron(I(m), s_1) * Qs[:,1],n,m) - s_1 * reshape(Qs[:,1],n,m) 

RepresentationTheory.gelfand_reduce(standardrepresentation(4).generators)[1]