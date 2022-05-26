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


ρ = Representation(Partition([3,2])) ⊗ Representation(Partition([2,2,1])) ⊗ Representation(Partition([3,1,1]))
λ,Q = blockdiagonalize(ρ)

for k = 1:length(λ.generators)
    @test Q' * ρ.generators[k] * Q ≈ λ.generators[k]
end

