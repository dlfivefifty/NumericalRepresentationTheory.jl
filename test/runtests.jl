using RepresentationTheory, Test

σ = Partition([3,3,2,1])
@test length(youngtableaux(σ)) == RepresentationTheory.hooklength(σ)
