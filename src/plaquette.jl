# 4/27/2026
# Ground state preparation for the interferometer
# This file contains the code to generate the indices for each hexagonal plaquette

using ITensors
using ITensorMPS

"""
  hexagonal_plaquettes(N::Integer, width::Integer) -> Vector{Vector{Int}}

  Return the six-site index lists for each hexagonal plaquette of a honeycomb
  cylinder with circumference `width` and `N` total sites, in the brick-lattice
  site labeling.

  Each plaquette is ordered `(a, b, c, d, e, f)` so that the standard Kitaev
  plaquette operator reads `(iY)_a Z_b X_c X_d Z_e (iY)_f`.

  # Examples
  ```julia
    plaquettes_w3 = hexagonal_plaquettes(24, 3)
    plaquettes_w4 = hexagonal_plaquettes(66, 4)
  ```
"""


function hexagonal_plaquettes(N::Integer, width::Integer)
    width ≥ 2 || throw(ArgumentError("width must be ≥ 2, got $width"))

    r = 2 * width                       # sites per ring around the cylinder
    plaquettes = Vector{Vector{Int}}()

    for idx in 1:2:N-r
        if mod(idx, r) == 1             # plaquette wrapping the cylinder seam
            push!(plaquettes, [idx, idx+1, idx+r, idx+r-1, idx+2r-2, idx+2r-1])
        else                            # interior plaquette
            push!(plaquettes, [idx, idx+1, idx+r, idx-1,   idx+r-2,  idx+r-1])
        end
    end
    return plaquettes
end