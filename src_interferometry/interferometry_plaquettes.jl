# Generate the reference indices for the plaquettes in the interferometry lattice
function interferometry_plaquette_reference_obc(input_sites::Int, input_length::Int)
	"""
		Generate a list of site indices as the reference points for the plaquettes 
		in the interferometry lattice with open boundary conditions
	"""


	# Set up the width profile and gauge to set up the x and y coordinates for each lattice site
	input_width = Int[]
	for i in 1:5
		append!(input_width, [3, 4, 4])
	end
	push!(input_width, 3)
	println("\nThe width profile for the interferometry lattice is:")
	@show input_width
	println("")

	
	# Set up the gauge to determine the x coordinates of each lattice site
	input_gauge = Int[]
	for idx in 0:length(input_width)
		append!(input_gauge, sum(input_width[1:idx]))
	end
	println("\nThe gauge for x coordinates of each lattice site is:")
	@show input_gauge
	println("")


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
		@show site_idx, x, y


		# Determine if the site can be used as a reference point and push it into the list
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
				end
			end
		end
	end
	# @show plaquette_refs

	return plaquette_refs
end



function interferometry_plaquette_obc()
	"""
		Generate a list of site indices for each plaquette in the interferometry lattice 
	"""

	# Set up the width profile and gauge to set up the x and y coordinates for each lattice site
	input_width = Int[]
	for i in 1:5
		append!(input_width, [3, 4, 4])
	end
	push!(input_width, 3)
	println("\nThe width profile for the interferometry lattice is:")
	@show input_width
	println("")

	
	# Set up the gauge to determine the x coordinates of each lattice site
	input_gauge = Int[]
	for idx in 0:length(input_width)
		append!(input_gauge, sum(input_width[1:idx]))
	end
	println("\nThe gauge for x coordinates of each lattice site is:")
	@show input_gauge
	println("")


	input_refs = Int[1, 2, 3, 9, 10, 17, 18, 23, 24, 25, 31, 32, 39, 40, 45, 46, 47]
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
	
	show(IOContext(stdout, :limit => false), "text/plain", plaquette)
	println()
	# return plaquette
end

# interferometry_plaquette_reference_obc(58, 8)



interferometry_plaquette_obc()