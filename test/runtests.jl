using NumericalRepresentationTheory, Permutations, LinearAlgebra, SparseArrays, Test
import NumericalRepresentationTheory: gelfandbasis, canonicalprojection

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

@testset "Canonical Projection" begin
    @testset "standard" begin
        ρ = standardrepresentation(4)
        @test rank(canonicalprojection(Partition(4), ρ)) == 1
        @test rank(canonicalprojection(Partition(3,1), ρ)) == 3

        P_1 = canonicalprojection(Partition(4), ρ)
        P_2 = canonicalprojection(Partition(3,1), ρ)
        @test P_1^2 ≈ P_1
        @test P_2^2 ≈ P_2

        Q_1 = qr(P_1).Q[:,1:1]
        @test Q_1' ≈ Q_1'*P_1
        Q_2 = qr(P_2).Q[:,1:3]
        @test Q_2' ≈ Q_2'*P_2
        Q = [Q_1 Q_2]'
        @test Q'Q ≈ I
        ρ_2 = Representation(3,1)
        @test Q*ρ.generators[1]*Q' ≈ blockdiag(sparse(I(1)),ρ_2.generators[3])
        @test Q*ρ.generators[2]*Q' ≈ blockdiag(sparse(I(1)),ρ_2.generators[2])
        @test Q*ρ.generators[3]*Q' ≈ blockdiag(sparse(I(1)),ρ_2.generators[1])
    end
    @testset "tensor" begin
        s = standardrepresentation(4)
        ρ = s ⊗ s
        λ,Q = blockdiagonalize(ρ)
        @test rank(canonicalprojection(Partition(4), ρ)) == 2
        @test rank(canonicalprojection(Partition(3,1), ρ)) == 3*3
        @test rank(canonicalprojection(Partition(2,2), ρ)) == 2
        @test rank(canonicalprojection(Partition(1,1,1,1), ρ)) == 0

        P_4 = canonicalprojection(Partition(4), ρ)
        @test P_4^2 ≈ P_4
        P_31 = canonicalprojection(Partition(3,1), ρ)
        @test P_31^2 ≈ P_31
        P_22 = canonicalprojection(Partition(2,2), ρ)
        @test P_22^2 ≈ P_22
        P_211 = canonicalprojection(Partition(2,1,1), ρ)
        @test P_211^2 ≈ P_211

        @test P_4*Q[:,1:2] ≈ Q[:,1:2]
        @test norm(P_4*Q[:,3:end]) ≤ 1E-15
        @test norm(P_31*Q[:,1:2]) ≤ 1E-15
        @test P_31*Q[:,3:3 +8] ≈ Q[:,3:3 +8]
        @test norm(P_31*Q[:,3+9:end]) ≤ 1E-15
        @test norm(P_22*Q[:,1:11]) ≤ 1E-15
        @test P_22*Q[:,12:13] ≈ Q[:,12:13]
        @test norm(P_22*Q[:,14:end]) ≤ 1E-14
        @test norm(P_211*Q[:,1:13]) ≤ 1E-14
        @test P_211*Q[:,14:end] ≈ Q[:,14:end]


        Q_31 = svd(P_31).U[:,1:9]
        @test Q_31'*P_31 ≈ Q_31'
        @test Q_31'*Q_31 ≈ I

        σ = Partition(3,1)
        n = Int(σ)
        ρ_31 = Representation(σ)
        n_σ = size(ρ_31,1)
        p_11 = n_σ/factorial(n) * sum(ρ_31(inv(g))[1,1]*ρ(g) for g in PermGen(n))
        @test p_11^2 ≈ p_11

        # sects a single representation
        @test rank(p_11 * P_31) == rank(p_11) == 3
        V_31_11 = svd(p_11).U[:,1:3]

        # V_31_11 is a basis
        Q̃_31 = hcat([Matrix(svd(hcat([(
        p_α1 = n_σ/factorial(n) * sum(ρ_31(inv(g))[1,α]*ρ(g) for g in PermGen(n));
        p_α1 * V_31_11[:,j]) for α = 1:3]...)).U) for j = 1:3]...)
        @test Q̃_31'Q̃_31 ≈ I

        # Q̃_31 spans same space as columns of Q
        @test norm(Q̃_31'Q[:,1:2]) ≤ 1E-14
        @test rank(Q̃_31'Q[:,3:3 +8]) == 9
        @test norm(Q̃_31'Q[:,12:end]) ≤ 1E-14

        # this shows the action of ρ acts on each column space separately
        @test Q̃_31'ρ.generators[1]*Q̃_31 ≈ Diagonal(Q̃_31'ρ.generators[1]*Q̃_31)
        @test Q̃_31'ρ.generators[2]*Q̃_31 ≈ blockdiag(sparse((Q̃_31'ρ.generators[2]*Q̃_31)[1:3,1:3]), sparse((Q̃_31'ρ.generators[2]*Q̃_31)[4:6,4:6]), sparse((Q̃_31'ρ.generators[2]*Q̃_31)[7:end,7:end]))
        @test Q̃_31'ρ.generators[3]*Q̃_31 ≈ blockdiag(sparse((Q̃_31'ρ.generators[3]*Q̃_31)[1:3,1:3]), sparse((Q̃_31'ρ.generators[3]*Q̃_31)[4:6,4:6]), sparse((Q̃_31'ρ.generators[3]*Q̃_31)[7:end,7:end]))

        # replace basis
        Q̃ = copy(Q)
        Q̃[:,3:3 +8] = Q̃_31
        ρ_22 = Representation(2,2)
        ρ_211 = Representation(2,1,1)
        @test Q'*ρ.generators[1]*Q ≈ blockdiag(sparse(I(2)),fill(ρ_31.generators[1],3)..., ρ_22.generators[1], ρ_211.generators[1])
        @test Q'*ρ.generators[2]*Q ≈ blockdiag(sparse(I(2)),fill(ρ_31.generators[2],3)..., ρ_22.generators[2], ρ_211.generators[2])
        @test Q'*ρ.generators[3]*Q ≈ blockdiag(sparse(I(2)),fill(ρ_31.generators[3],3)..., ρ_22.generators[3], ρ_211.generators[3])

        @test_skip Q̃'*ρ.generators[1]*Q̃ ≈ blockdiag(sparse(I(2)),fill(ρ_31.generators[1],3)..., ρ_22.generators[1], ρ_211.generators[1])
        @test_skip Q̃'*ρ.generators[2]*Q̃ ≈ blockdiag(sparse(I(2)),fill(ρ_31.generators[2],3)..., ρ_22.generators[1], ρ_211.generators[1])
    end
end


# basis = gelfandbasis(Representation(Partition([3,2,1])).generators)
# Λ = Matrix{Int}(undef, size(basis[1],1), length(basis))
# for k in axes(Λ,1), j in axes(Λ,2)
#     Λ[k,j] = round(Int,basis[j][k,k])
# end

