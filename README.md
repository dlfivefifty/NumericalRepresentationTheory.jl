# RepresentationTheory.jl
A Julia package for representation theory of the symmetric group




This package supports basic representation theory of the symmetric group. One can form irreducible representations (irreps) by specifying the corresponding permutation, combine representations via direct sum and Kronecker product, and also calculate the resulting irrep multipliciplities. For example, the following code calculates the Kronecker coefficients of two irreps of S₇, given by the partitions `3+3+1` and `2+2+2+1`:
```julia
julia> using RepresentationTheory, Permutations, Plots

julia> R₁ = Representation(Partition([5,1,1]));

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

julia> R₂ = Representation(Partition([2,2,2,1]));

julia> R = R₁ ⊗ R₂; # Tensor product representation

julia> plot(multiplicities(R));
```
<img src=https://github.com/dlfivefifty/RepresentationTheory.jl/raw/master/images/mults.png width=500 height=400>
