# NumericalRepresentationTheory.jl
A Julia package for representation theory of the symmetric group




This package supports basic representation theory of the symmetric group. One can form irreducible representations (irreps) by specifying the corresponding permutation, combine representations via direct sum and Kronecker product, and also calculate the resulting irrep multipliciplities. For example, the following code calculates the Kronecker coefficients of two irreps of S₇, specified by the partitions `5+1+1` and `2+2+2+1`:
```julia
julia> using NumericalRepresentationTheory, Permutations, Plots

julia> R₁ = Representation(5,1,1);

julia> g = Permutation([7,2,1,3,4,6,5]); Matrix(R₁(g)) # Matrix representation of a specific permutation
15×15 Array{Float64,2}:
 -0.2       -0.0408248  -0.0527046  …   0.0         0.0         0.0     
  0.163299   0.241667    0.31199        0.0         0.0         0.0     
  0.0       -0.161374    0.0138889      0.0         0.0         0.0     
  0.0        0.0        -0.157135       0.467707    0.810093    0.0     
  0.0        0.0         0.0            0.270031   -0.155902   -0.881917
 -0.966092   0.0493007   0.0636469  …   0.0         0.0         0.0     
  0.0       -0.190941    0.0164336      0.0         0.0         0.0     
  0.0        0.0        -0.185924       0.0790569   0.136931    0.0     
  0.0        0.0         0.0            0.0456435  -0.0263523  -0.149071
  0.0        0.935414   -0.0805076      0.0         0.0         0.0     
  0.0        0.0         0.91084    …   0.0968246   0.167705    0.0     
  0.0        0.0         0.0            0.0559017  -0.0322749  -0.182574
  0.0        0.0         0.0            0.125       0.216506    0.0     
  0.0        0.0         0.0            0.0721688  -0.0416667  -0.235702
  0.0        0.0         0.0           -0.816497    0.471405   -0.333333

julia> R₂ = Representation(2,2,2,1);

julia> R = R₁ ⊗ R₂; # Tensor product representation

julia> multiplicities(R) # Returns a dictionary whose keys are partitions and values are the multiplicities
Dict{Partition, Int64} with 8 entries:
  7 = 2 + 2 + 2 + 1     => 1
  7 = 4 + 2 + 1         => 1
  7 = 3 + 1 + 1 + 1 + 1 => 1
  7 = 3 + 2 + 2         => 1
  7 = 3 + 3 + 1         => 1
  7 = 2 + 2 + 1 + 1 + 1 => 1
  7 = 4 + 1 + 1 + 1     => 1
  7 = 3 + 2 + 1 + 1     => 2

julia> plot(multiplicities(R)) # We can also plot
```
<img src=https://github.com/dlfivefifty/NumericalRepresentationTheory.jl/raw/master/images/mults.png width=500 height=400>

In addition, one can find an orthogonal transformation that reduces a representation to irreducibles:
```julia
julia> ρ,Q = blockdiagonalize(R); # Q'R(g)*Q ≈ ρ(g) where ρ is a direct sum (block diagonal) of irreducibles.

julia> Q'R(g)*Q ≈ ρ(g)
true

julia> ρ(g) ≈ (Representation(4,2,1) ⊕ Representation(4,1,1,1) ⊕ Representation(3,3,1) ⊕ Representation(3,2,2) ⊕ Representation(3,2,1,1) ⊕ Representation(3,2,1,1) ⊕ Representation(3,1,1,1,1) ⊕ Representation(2,2,2,1) ⊕ Representation(2,2,1,1,1))(g)
true
```