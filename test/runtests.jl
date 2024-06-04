using NumericalRepresentationTheory, Permutations, LinearAlgebra, Test
import NumericalRepresentationTheory: gelfandbasis, canonicalrepresentation

@testset "Representations" begin
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

    @testset "Rotate irrep" begin
        ρ  = Representation(3,2,1,1)
        λ,Q = blockdiagonalize(ρ)
        @test Q ≈ I
        @test multiplicities(ρ)[Partition(3,2,1,1)] == 1

        Q = qr(randn(size(ρ,1), size(ρ,1))).Q
        ρ̃ = Representation(map(τ -> Q*τ*Q', ρ.generators))
        @test multiplicities(ρ̃) == multiplicities(ρ)
        @test abs.(blockdiagonalize(ρ̃)[2]) ≈ abs.(Matrix(Q))
    end

    @testset "Rico Bug" begin
        ρ = Representation(2,1)
        g = Permutation([1,2,3])
        @test ρ(g) == I(2)
    end
end


# basis = gelfandbasis(Representation(Partition([3,2,1])).generators)
# Λ = Matrix{Int}(undef, size(basis[1],1), length(basis))
# for k in axes(Λ,1), j in axes(Λ,2)
#     Λ[k,j] = round(Int,basis[j][k,k])
# end

