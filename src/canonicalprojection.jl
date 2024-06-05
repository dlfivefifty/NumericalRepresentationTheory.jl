###############
# This implements the "canonical projection" a la Hymabaccus 2020
##############

function canonicalprojection(σ::Partition, ρ)
    n = Int(σ)
    ρ_σ = Representation(σ)
    n_σ = size(ρ_σ,1)
    n_σ/factorial(n) * sum(tr(ρ_σ(g))ρ(g) for g in PermGen(n))
end 