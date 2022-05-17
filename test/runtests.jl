using RepresentationTheory, Test

σ = Partition([3,3,2,1])
@test length(youngtableaux(σ)) == RepresentationTheory.hooklength(σ)

@test multiplicities(standardrepresentation(4))[Partition([3,1])] == 1


RepresentationTheory.gelfandbasis(standardrepresentation(4).generators)