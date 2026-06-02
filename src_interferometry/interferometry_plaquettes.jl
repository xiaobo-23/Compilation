# 6/1/2026
# Helper functions that set up the indices for the plaquettes in the interferometry lattice
# Helper functions that compute the expectation values of the plaquette operators


"""
	Generate a list of site indices as the reference points for the plaquettes 
	in the interferometry lattice with open boundary conditions
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
	Generate a list of site indices for each plaquette in the interferometry lattice 
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



"""
    measure_plaquettes(ψ, sites, plaquette_indices, plaquette_ops; imag_tol=1e-8)
        -> Vector{Float64}

Expectation value ⟨ψ|Wₚ|ψ⟩ of the six-site plaquette operator on every
hexagon. `plaquette_indices[p, :]` holds the six site indices of plaquette
`p`; `plaquette_ops` is the fixed operator string applied in that order.

The string uses "iY" = i·σʸ (real matrix) to keep ITensor in real
arithmetic. Each pair of "iY" factors contributes i² = -1, so the physical
plaquette value is `real(⟨Wₚ⟩) / iⁿ` where n = number of "iY". For the
standard 6-site string with two "iY", this sign is simply -1.
"""
function measure_plaquettes(ψ::MPS, sites,
                            plaquette_indices::AbstractMatrix{Int},
                            plaquette_ops::AbstractVector{<:AbstractString};
                            imag_tol::Real = 1e-8)
    n_iY = count(==("iY"), plaquette_ops)
    iseven(n_iY) || error("Odd number of 'iY' factors → operator is non-Hermitian")
    sign = real(im^n_iY)            # = -1 for the standard two-iY string

    nplaq = size(plaquette_indices, 1)
    vals  = zeros(Float64, nplaq)

    for p in 1:nplaq
        idx  = @view plaquette_indices[p, :]
        os_w = OpSum()
        add!(os_w, 1.0, Iterators.flatten(zip(plaquette_ops, idx))...)
        W = MPO(os_w, sites)

        z = inner(ψ', W, ψ)
        abs(imag(z)) < imag_tol ||
            @warn "Plaquette $p has non-negligible imaginary part" imag(z)
        vals[p] = sign * real(z)
    end
    return vals
end