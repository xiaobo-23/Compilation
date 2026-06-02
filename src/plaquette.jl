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


"""
    interferometry_plaquette_reference_obc(input_sites, input_length,
                                           input_width, input_gauge) -> Vector{Int}

Find the **reference (anchor) site** of every hexagonal plaquette on the
open-boundary interferometer lattice. Each hexagon is identified by a single
lower-left vertex; `interferometry_plaquette_obc` later expands each anchor
into its six vertices.

A site is selected as an anchor based on its lattice coordinates `(x, y)`:
  - `x` = column index, recovered from `input_gauge`
    (`input_gauge[x] < site ≤ input_gauge[x+1]`);
  - `y` = row within the column (1-based), only meaningful for width-4
    columns (left as 0 otherwise).
Only odd columns with `x < 2·input_length - 1` can anchor a hexagon (so the
plaquette does not run off the right edge). Within those, width-3 columns
always anchor; width-4 columns anchor except on specific boundary rows,
which depend on the widths of the neighboring columns `x-1` and `x+1`.

# Arguments
- `input_sites::Int`: total number of lattice sites `N`.
- `input_length::Int`: number of unit cells along x (`Nx_unit`); bounds the
  rightmost anchor column.
- `input_width::Vector{Int}`: per-column site count (the `width_profile`,
  values 3 or 4). Required despite the `Int[]` default.
- `input_gauge::Vector{Int}`: cumulative-sum gauge mapping a site to its
  column (the `x_gauge`). Required despite the `Int[]` default.

# Returns
- `Vector{Int}`: site indices of all hexagon anchors, in increasing order.
"""
function interferometry_plaquette_reference_obc(input_sites::Int, input_length::Int, 
	input_width::Vector{Int}=Int[], input_gauge::Vector{Int}=Int[])
	
	# Loop through all the sites and select the sites that can be used as the reference points for plaquettes
	plaquette_refs = Int[]
	for site_idx in 1 : input_sites
		# Determine the x coordinate of the site
		x = 0
		for idx in 1 : length(input_gauge) - 1
			if site_idx > input_gauge[idx] && site_idx <= input_gauge[idx + 1]
				x = idx
				break
			end
		end

		# Determine the y coordinate of the site
		y = 0
		if input_width[x] == 4
			for idx in 1 : length(input_gauge) - 1
				if site_idx > input_gauge[idx] && site_idx <= input_gauge[idx + 1]
					tmp = site_idx - input_gauge[idx]
					y = mod(tmp - 1, 4) + 1
					break
				end
			end
		end
		# @show site_idx, x, y


		# Determine if the site can be used as a reference point based on its (x, y) coordinates
		if mod(x, 2) == 1 && x < 2*input_length - 1
			if input_width[x] == 3
				push!(plaquette_refs, site_idx)
			elseif input_width[x] == 4
				if input_width[x - 1] == 4 && input_width[x + 1] == 3
					if y != 1 && y != 4
						push!(plaquette_refs, site_idx)
					end
				elseif input_width[x - 1] == 3 && input_width[x + 1] == 4
					if y != 1 && y != 2
						push!(plaquette_refs, site_idx)	
					end
				elseif input_width[x - 1] == 4 && input_width[x + 1] == 4
					if y != 1
						push!(plaquette_refs, site_idx)
					end
				end
			end
		end
	end
	# @show plaquette_refs

	return plaquette_refs
end


"""
    interferometry_plaquette_obc(input_width, input_gauge, input_refs) -> Matrix{Int}

Expand each plaquette anchor into the **six site indices** of its hexagon.
Row `p` of the returned matrix lists the six vertices of plaquette `p` in a
fixed traversal order around the hexagon; that order is paired position-by-
position with the operator string in `measure_plaquettes`, so the column
order here must stay in sync with `plaquette_ops`.

Vertices are built from the anchor by fixed offsets: columns 1, 2, 6 are
`reference`, `reference+4`, `reference+3`; columns 3 and 5 step by +4 when
the next two columns are both width-4, otherwise by +3; column 4 is
`column 5 + 4`. The +3 vs +4 choice accounts for the narrowing at width-3
(constriction) columns.

# Arguments
- `input_width::Vector{Int}`: per-column site count (`width_profile`).
  Required despite the `Int[]` default.
- `input_gauge::Vector{Int}`: cumulative-sum column gauge (`x_gauge`).
  Required despite the `Int[]` default.
- `input_refs::Vector{Int}`: anchor sites from
  `interferometry_plaquette_reference_obc`.

# Returns
- `Matrix{Int}` of size `(length(input_refs), 6)`: row `p` holds the six
  ordered site indices of plaquette `p`.
"""
function interferometry_plaquette_obc(input_width::Vector{Int}=Int[], input_gauge::Vector{Int}=Int[], input_refs::Vector{Int}=Int[])
	# Initialize the plaquette matrix 
	plaquette = Matrix{Int}(undef, length(input_refs), 6)

	
	for (idx, reference) in enumerate(input_refs)
		# Determine the x coordinate of the site
		x = 0
		for tmp_idx in 1 : length(input_gauge) - 1
			if reference > input_gauge[tmp_idx] && reference <= input_gauge[tmp_idx + 1]
				x = tmp_idx
				break
			end
		end

		# Construct indices for the plaquette based on the reference site
		plaquette[idx, 1] = reference
		plaquette[idx, 2] = reference + 4
		plaquette[idx, 6] = reference + 3

		if input_width[x + 1] == 4 && input_width[x + 2] == 4
			plaquette[idx, 3] = plaquette[idx, 2] + 4
			plaquette[idx, 5] = plaquette[idx, 6] + 4
		else
			plaquette[idx, 3] = plaquette[idx, 2] + 3
			plaquette[idx, 5] = plaquette[idx, 6] + 3
		end

		plaquette[idx, 4] = plaquette[idx, 5] + 4
	end
	
	# show(IOContext(stdout, :limit => false), "text/plain", plaquette)
	# println("")
	
	return plaquette
end