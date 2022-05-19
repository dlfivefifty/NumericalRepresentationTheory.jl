using RepresentationTheory, Test
import RepresentationTheory: gelfandbasis

σ = Partition([3,3,2,1])
@test length(youngtableaux(σ)) == hooklength(σ)
@test all(isdiag,gelfandbasis(Representation(σ).generators)[3])

@test multiplicities(standardrepresentation(4))[Partition([3,1])] == 1


s = standardrepresentation(3)
ρ = s ⊗ s
@test multiplicities(ρ)[Partition([2,1])] == 3




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