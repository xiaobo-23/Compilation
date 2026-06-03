# 12/10/2025
# Create all two-body interaction bonds and all three-spin interaction object for the interferometer
# The interferometer geometry is a specific honeycomb lattice with open boundary condition along both x and y directions, with site near the constrictions and at the edges removed

function interferometry_lattice_obc(Nx::Int, Ny::Int, Nsites::Int, geometry_profile::Vector{Int})
	"""
		Setting up all bonds on the lattice in the interferometry
		Nx is the number of columns and is an even number 
	"""
	
	# The default is open boundary condition along both the x and y directions
	# Set up the number of sites and the number of bonds
	if Nsites != Nx * Ny - 6
		error("The number of sites does not match the interferometry geometry!")
	end
	Nbond = (div(Nx, 2) - 1) * 11 - 3
	# @info "Number of bonds: $Nbond"

	
	# Obtain an array to gaue the x coordinates of each lattice point
	xcoordinate_gauge = Int[]
	for idx in 0:length(geometry_profile)
		append!(xcoordinate_gauge, sum(geometry_profile[1:idx]))
	end
	# @show xcoordinate_gauge	


	# Set up the lattice as an tuple of bonds
	latt = Lattice(undef, Nbond)
  	b = 0
	for n in 1 : Nsites
		x = 0
		for idx in 1 : length(xcoordinate_gauge) - 1
			if n > xcoordinate_gauge[idx] && n <= xcoordinate_gauge[idx + 1]
				x = idx
				break
			end
		end

		y = 0
		if geometry_profile[x] == 4
			for idx in 1 : length(xcoordinate_gauge) - 1
				if n > xcoordinate_gauge[idx] && n <= xcoordinate_gauge[idx + 1]
					tmp = n - xcoordinate_gauge[idx]
					y = mod(tmp - 1, 4) + 1
					break
				end
			end
		end
		# @show n, x, y

		
		# horizontal bonds 
		if iseven(x) && x < Nx
			if geometry_profile[x] == 3 && geometry_profile[x + 1] == 4
				latt[b += 1] = LatticeBond(n, n + Ny - 1)
				# @show n, n + Ny - 1
			elseif geometry_profile[x] == 4 && geometry_profile[x + 1] == 3
				if y != 1 
					latt[b += 1] = LatticeBond(n, n + Ny - 1)
					# @show n, n + Ny - 1
				end
			elseif geometry_profile[x] == geometry_profile[x + 1] == 4
				latt[b += 1] = LatticeBond(n, n + Ny)
				# @show n, n + Ny
			end
		end
		
		if isodd(x)
			if geometry_profile[x] == 3 && geometry_profile[x + 1] == 4
				latt[b += 1] = LatticeBond(n, n + Ny)
				latt[b += 1] = LatticeBond(n, n + Ny - 1)
				# @show n, n + Ny
				# @show n, n + Ny - 1
			elseif geometry_profile[x] == 4 && geometry_profile[x + 1] == 3
				if y == 1
					latt[b += 1] = LatticeBond(n, n + Ny)
					# @show n, n + Ny
				elseif y == Ny 
					latt[b += 1] = LatticeBond(n, n + Ny - 1)
					# @show n, n + Ny - 1
				else
					latt[b += 1] = LatticeBond(n, n + Ny)
					latt[b += 1] = LatticeBond(n, n + Ny - 1)
					# @show n, n + Ny 
					# @show n, n + Ny - 1
				end
			elseif geometry_profile[x] == geometry_profile[x + 1] == 4
				if y == 1
					latt[b += 1] = LatticeBond(n, n + Ny)
					# @show n, n + Ny
				else
					latt[b += 1] = LatticeBond(n, n + Ny - 1)
					latt[b += 1] = LatticeBond(n, n + Ny)
					# @show n, n + Ny - 1
					# @show n, n + Ny
				end
			end 
		end

	end
	
	# Check the number of bonds that have been set up
	if b != Nbond
		error("The number of bonds that have been set up does not match the expected value!")
	end

	return latt
end




# Implement the wedge object to introduce the three-body interaction on the XC geometry
function interferometry_wedge(Nx::Int, Ny::Int, Nsites::Int, geometry_profile::Vector{Int})
	"""
		Setting up all wedges on the lattice in the interferometry
		Nx is the number of columns and is an even number
	"""
	if Nsites != Nx * Ny - 6
		error("The number of sites does not match the interferometry geometry!")
	end
	Nwedge = 3 * Nsites - 4 * (Ny - 1) - 2 * (Nx - 6) - 20
	@info "Number of wedge bonds: $Nwedge"


	# Obtain an array to gaue the x coordinates of each lattice point
	xcoordinate_gauge = Int[]
	for idx in 0:length(geometry_profile)
		append!(xcoordinate_gauge, sum(geometry_profile[1:idx]))
	end
	# @show xcoordinate_gauge	
	

	wedge = Vector{WedgeBond}(undef, Nwedge)
	b = 0
	for n in 1 : Nsites
		x = 0
		for idx in 1 : length(xcoordinate_gauge) - 1
			if n > xcoordinate_gauge[idx] && n <= xcoordinate_gauge[idx + 1]
				x = idx
				break
			end
		end

		y = 0
		if geometry_profile[x] == 4
			for idx in 1 : length(xcoordinate_gauge) - 1
				if n > xcoordinate_gauge[idx] && n <= xcoordinate_gauge[idx + 1]
					tmp = n - xcoordinate_gauge[idx]
					y = mod(tmp - 1, 4) + 1
					break
				end
			end
		end
		# @show n, x, y

		if x == 1
			wedge[b += 1] = WedgeBond(n + Ny - 1, n, n + Ny)
		elseif x == Nx 
			wedge[b += 1] = WedgeBond(n - Ny, n, n - Ny + 1)
		else
			if iseven(x)
				if geometry_profile[x] == Ny 
					if geometry_profile[x + 1] == Ny - 1
						if y == 1
							wedge[b += 1] = WedgeBond(n - Ny, n, n - Ny + 1)
						elseif y == Ny
							wedge[b += 1] = WedgeBond(n - Ny, n, n + Ny - 1)
						else
							wedge[b += 1] = WedgeBond(n - Ny, n, n - Ny + 1)
							wedge[b += 1] = WedgeBond(n - Ny, n, n + Ny - 1)
							wedge[b += 1] = WedgeBond(n - Ny + 1, n, n + Ny - 1)
						end
					elseif geometry_profile[x - 1] == Ny - 1
						if y == Ny 
							wedge[b += 1] = WedgeBond(n - Ny, n, n + Ny)
						elseif y == 1
							wedge[b += 1] = WedgeBond(n - Ny + 1, n, n + Ny)
						else
							wedge[b += 1] = WedgeBond(n - Ny, n, n - Ny + 1)
							wedge[b += 1] = WedgeBond(n - Ny, n, n + Ny)
							wedge[b += 1] = WedgeBond(n - Ny + 1, n, n + Ny)
						end	
					else
						if y == Ny 
							wedge[b += 1] = WedgeBond(n - Ny, n, n + Ny)
						elseif y == 1
							wedge[b += 1] = WedgeBond(n - Ny, n, n - Ny + 1)
							wedge[b += 1] = WedgeBond(n - Ny, n, n + Ny)
							wedge[b += 1] = WedgeBond(n - Ny + 1, n, n + Ny)
						else
							wedge[b += 1] = WedgeBond(n - Ny, n, n - Ny + 1)
							wedge[b += 1] = WedgeBond(n - Ny, n, n + Ny)
							wedge[b += 1] = WedgeBond(n - Ny + 1, n, n + Ny)
						end
					end 
				elseif geometry_profile[x] == Ny - 1
					wedge[b += 1] = WedgeBond(n - Ny, n, n - Ny + 1)
					wedge[b += 1] = WedgeBond(n - Ny, n, n + Ny - 1)
					wedge[b += 1] = WedgeBond(n - Ny + 1, n, n + Ny - 1)
				end
			else
				if geometry_profile[x] == Ny - 1
					wedge[b += 1] = WedgeBond(n - Ny + 1, n, n + Ny)
					wedge[b += 1] = WedgeBond(n - Ny + 1, n, n + Ny - 1)
					wedge[b += 1] = WedgeBond(n + Ny - 1, n, n + Ny)
				else
					if geometry_profile[x - 1] == Ny - 1
						if y == 1
							wedge[b += 1] = WedgeBond(n - Ny + 1, n, n + Ny)
						elseif y == Ny 
							wedge[b += 1] = WedgeBond(n + Ny - 1, n, n + Ny)
						else
							wedge[b += 1] = WedgeBond(n - Ny + 1, n, n + Ny)
							wedge[b += 1] = WedgeBond(n - Ny + 1, n, n + Ny - 1)
							wedge[b += 1] = WedgeBond(n + Ny - 1, n, n + Ny)
						end
					elseif geometry_profile[x + 1] == Ny - 1
						if y == 1
							wedge[b += 1] = WedgeBond(n - Ny, n, n + Ny)
						elseif y == Ny
							wedge[b += 1] = WedgeBond(n - Ny, n, n + Ny - 1)
						else
							wedge[b += 1] = WedgeBond(n - Ny, n, n + Ny)
							wedge[b += 1] = WedgeBond(n - Ny, n, n + Ny - 1)
							wedge[b += 1] = WedgeBond(n + Ny - 1, n, n + Ny)
						end
					else
						if y == 1
							wedge[b += 1] = WedgeBond(n - Ny, n, n + Ny)
						else
							wedge[b += 1] = WedgeBond(n - Ny, n, n + Ny)
							wedge[b += 1] = WedgeBond(n - Ny, n, n + Ny - 1)
							wedge[b += 1] = WedgeBond(n + Ny - 1, n, n + Ny)
						end
					end
				end
			end
		end
	end
	
	# Check the number of wedges that have been set up
	if length(wedge) != Nwedge
		error("The number of wedges that have been set up does not match the expected value!")
	end
	
	# @show wedge
	return wedge
end