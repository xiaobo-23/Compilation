# 12/10/2025
# Implement the interferometry based on the honeycomb lattice geometry

# """
# A LatticeBond is a struct which represents
# a single bond in a geometrical lattice or
# else on interaction graph defining a physical
# model such as a quantum Hamiltonian.

# LatticeBond has the following data fields:

#   - s1::Int -- number of site 1
#   - s2::Int -- number of site 2
#   - x1::Float64 -- x coordinate of site 1
#   - y1::Float64 -- y coordinate of site 1
#   - x2::Float64 -- x coordinate of site 2
#   - y2::Float64 -- y coordinate of site 2
#   - type::String -- optional description of bond type
# """

struct LatticeBond
  s1::Int
  s2::Int
  x1::Float64
  y1::Float64
  x2::Float64
  y2::Float64
  type::String
end

"""
    LatticeBond(s1::Int,s2::Int)

    LatticeBond(s1::Int,s2::Int,
                x1::Real,y1::Real,
                x2::Real,y2::Real,
                type::String="")

Construct a LatticeBond struct by
specifying just the numbers of sites
1 and 2, or additional details including
the (x,y) coordinates of the two sites and
an optional type string.
"""

function LatticeBond(s1::Int, s2::Int)
  return LatticeBond(s1, s2, 0.0, 0.0, 0.0, 0.0, "")
end


function LatticeBond(
  s1::Int, s2::Int, x1::Real, y1::Real, x2::Real, y2::Real, bondtype::String=""
)
  cf(x) = convert(Float64, x)
  return LatticeBond(s1, s2, cf(x1), cf(y1), cf(x2), cf(y2), bondtype)
end


"""
Lattice is an alias for Vector{LatticeBond}
"""
const Lattice = Vector{LatticeBond}



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


function interferometry_lattice_pbc(Nx::Int, Ny::Int)::Lattice
	"""
		Set up all bonds on the interferometer lattice 
		Use periodic boundary condition along the x direction
		Use open boundary condition along the y direction
		Nx: Total number of unit cells along the x direction 
		Ny: Total number of unit cells along the y direction
	"""

	# Set up the number of sites and the number of bonds
	Nsite = 2 * Nx * Ny - 4
	input_geometry = Int[Ny - 2, Ny, Ny, Ny, Ny, Ny, Ny, Ny - 2]
	if sum(input_geometry) != Nsite
		error("The number of sites does not match the interferometry geometry!")
	end
	Nbond = 3 * div(Nsite, 2) - 8


	# Obtain an array to gaue the x coordinates of each lattice point
	xcoordinate_gauge = Int[]
	for idx in 0:length(input_geometry)
		append!(xcoordinate_gauge, sum(input_geometry[1:idx]))
	end
	# @show xcoordinate_gauge

	
	# Set up the lattice as an tuple of bonds
	latt = Lattice(undef, Nbond)
  	b = 0
	for n in 1:Nsite
		# Determine the x coordinate of the lattice point n based on the input geometry
		x = 0
		for idx in 1 : length(xcoordinate_gauge) - 1
			if n > xcoordinate_gauge[idx] && n <= xcoordinate_gauge[idx + 1]
				x = idx
				break
			end
		end
		
		# Determine the y coordinate of the lattice point n based on the input geometry
		y = 0
		for idx in 1 : length(xcoordinate_gauge) - 1
			if n > xcoordinate_gauge[idx] && n <= xcoordinate_gauge[idx + 1]
				tmp = n - xcoordinate_gauge[idx]
				y = mod(tmp - 1, input_geometry[idx]) + 1
				break
			end
		end
		# @show n, x, y
		
		
		# Set up the bonds for two-body interactions based on the x and y coordinates of the lattice point n
		if isodd(x)
			if x == 1
				latt[b += 1] = LatticeBond(n, n + Ny - 1)
				# @show n, n + Ny - 1
				if y <= 2
					latt[b += 1] = LatticeBond(n, n + Ny - 2)
					# @show n, n + Ny - 2
				else
					latt[b += 1] = LatticeBond(n, n + Ny)
					# @show n, n + Ny
				end
			elseif x == 2 * Nx - 1
				if y == 1
					latt[b += 1] = LatticeBond(n, n + Ny)
					latt[b += 1] = LatticeBond(n, n + 2 * Ny - 3)
					# @show n, n + Ny
					# @show n, n + 2 * Ny - 3
				elseif n in [36, 37]
					latt[b += 1] = LatticeBond(n, n + Ny - 1)
					# @show n, n + Ny - 1
				elseif n in [39, 40]
					latt[b += 1] = LatticeBond(n, n + Ny - 2)
					# @show n, n + Ny - 2
				else
					latt[b += 1] = LatticeBond(n, n + Ny - 1)
					latt[b += 1] = LatticeBond(n, n + Ny - 2)
					# @show n, n + Ny - 1
					# @show n, n + Ny - 2
				end
			else
				latt[b += 1] = LatticeBond(n, n + Ny)
				# @show n, n + Ny 
				if y == 1
					latt[b += 1] = LatticeBond(n, n + 2 * Ny - 1)
					# @show n, n + 2 * Ny - 1
				else
					latt[b += 1] = LatticeBond(n, n + Ny - 1)
					# @show n, n + Ny - 1
				end
			end
		else
			if x != 2 * Nx
				latt[b += 1] = LatticeBond(n, n + Ny)
				# @show n, n + Ny
			end
		end
	end
	

	# Check if the number of bonds that have been set up matches the expected value	
	if length(latt) != Nbond
		error("The number of bonds that have been set up does not match the expected value!")
	end

	return latt
end



# 01/06/2025
# Define a wedge bond to introduce the three-body interaction
struct WedgeBond
  s1::Int
  s2::Int
  s3::Int
  x1::Float64
  y1::Float64
  x2::Float64
  y2::Float64
  x3::Float64
  y3::Float64
  type::String
end


function WedgeBond(s1::Int, s2::Int, s3::Int)
	return WedgeBond(s1, s2, s3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "")
end


function WedgeBond(
  s1::Int, s2::Int, s3::Int, x1::Real, y1::Real, x2::Real, y2::Real, x3::Real, y3::Real, bondtype::String=""
)
  cf(x) = convert(Float64, x)
  return WedgeBond(s1, s2, s3, cf(x1), cf(y1), cf(x2), cf(y2), cf(x3), cf(y3), bondtype)
end


# """
# Wedge is an alias for Vector{WedgeBond}
# """
# const Wedge = Vector{WedgeBond}


# 05/21/2025
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



# 02/19/2026
# Implement the three-spin interactions on the interferometer with periodic boundary condition along the y direction
function interferometry_wedge_pbc(Nx::Int, Ny::Int, Nsites::Int)
	"""
		Set all three-spin interactions on the interfermeter lattice
		Nx is the number of unit cells along the x direction
		Ny is the number of unit cells along the y direction
	"""

	# Validate the interferometer by checking the number of sites 
	if Nsites != 2 * Nx * Ny - 4
		error("The number of sites does not match the interferometer!")
	end
	
	
	# Set up the input geometry of the interferometer
	input_geometry = Int[Ny - 2, Ny, Ny, Ny, Ny, Ny, Ny, Ny - 2]
	if sum(input_geometry) != Nsites
		error("The number of sites does not match the interferometry geometry!")
	end
	

	# Obtain an array to gaue the x coordinates of each lattice point
	xcoordinate_gauge = Int[]
	for idx in 0:length(input_geometry)
		append!(xcoordinate_gauge, sum(input_geometry[1:idx]))
	end
	# @show xcoordinate_gauge
	

    # Set up the number of three-spin interactions 	
	Nterms = 3 * Nsites - (Ny - 2) * 8
	# @info "Number of three-spin interactions: $Nterms"


	wedge = Vector{WedgeBond}(undef, Nterms)
	b = 0
	for n in 1 : Nsites
		# Determine the x coordinate of the lattice point n based on the input geometry
		x = 0
		for idx in 1 : length(xcoordinate_gauge) - 1
			if n > xcoordinate_gauge[idx] && n <= xcoordinate_gauge[idx + 1]
				x = idx
				break
			end
		end
		
		# Determine the y coordinate of the lattice point n based on the input geometry
		y = 0
		for idx in 1 : length(xcoordinate_gauge) - 1
			if n > xcoordinate_gauge[idx] && n <= xcoordinate_gauge[idx + 1]
				tmp = n - xcoordinate_gauge[idx]
				y = mod(tmp - 1, input_geometry[idx]) + 1
				break
			end
		end
		@show n, x, y
		
		
		# Set up three-spin interactions in the bulk of the interferometer	
		if isodd(x) && x != 1 && x != 2 * Nx - 1
			n₁ = n - Ny
			n₂ = n + Ny 
			if y == 1
				n₃ = n + 2 * Ny - 1
				wedge[b += 1] = WedgeBond(n₂, n, n₃)
				@show n₂, n, n₃
			else
				n₃ = n + Ny - 1
				wedge[b += 1] = WedgeBond(n₃, n, n₂)
				@show n₃, n, n₂
			end
			wedge[b += 1] = WedgeBond(n₁, n, n₂)
			wedge[b += 1] = WedgeBond(n₁, n, n₃)
			@show n₁, n, n₂
			@show n₁, n, n₃
		elseif iseven(x) && x != 2 && x != 2 * Nx
			n₁ = n - Ny
			if y == Ny 
				n₂ = n - 2 * Ny + 1
				wedge[b += 1] = WedgeBond(n₂, n, n₁)
				@show n₂, n, n₁
			else
				n₂ = n - Ny + 1
				wedge[b += 1] = WedgeBond(n₁, n, n₂)
				@show n₁, n, n₂
			end
			n₃ = n + Ny 
			wedge[b += 1] = WedgeBond(n₁, n, n₃)
			wedge[b += 1] = WedgeBond(n₂, n, n₃)
			@show n₁, n, n₃
			@show n₂, n, n₃
		end

	
		# Three-spin interactions at the boundary x = 1
		if x == 1
			n₁ = y <= 2 ? n + Ny - 2 : n + Ny - 1
			n₂ = y <= 2 ? n + Ny - 1 : n + Ny
			wedge[b += 1] = WedgeBond(n₁, n, n₂)
			@show n₁, n, n₂
		end



		# Three-spin interaction at the boundary x = 2
		if x == 2
			if y <= 2
				wedge[b += 1] = WedgeBond(n - Ny + 2, n, n + Ny)
				@show n - Ny + 2, n, n + Ny
			end

			if 2 <= y <= 5
				wedge[b += 1] = WedgeBond(n - Ny + 1, n, n + Ny)
				@show n - Ny + 1, n, n + Ny
			end

			if y >= 5
				wedge[b += 1] = WedgeBond(n - Ny, n, n + Ny)
				@show n - Ny, n, n + Ny
			end
		end
		
		

		# Three-spin interactions at the boundary x = 2 * Nx - 1
		if x == 2 * Nx - 1
			n₁ = n - Ny
			
			if y == 1
				wedge[b += 1] = WedgeBond(n₁, n, n + Ny)
				@show n₁, n, n + Ny
			end
			
			if 2 <= y <= 4
				wedge[b += 1] = WedgeBond(n₁, n, n + Ny - 1)
				@show n₁, n, n + Ny - 1
			end
			
			if 4 <= y <= 6
				wedge[b += 1] = WedgeBond(n₁, n, n + Ny - 2)
				@show n₁, n, n + Ny - 2
			end
		end


	
		# Three-spin interactions at the boundary x = 2 * Nx
		if x == 2 * Nx
			if y == 1
				n₁ = n - Ny 
				n₂ = n - Ny + 1
			elseif y == 2 || y == 3
				n₁ = n - Ny + 1
				n₂ = n - Ny + 2
			else
				n₁ = n - 2 * Ny + 3
				n₂ = n - Ny + 2
			end
			wedge[b += 1] = WedgeBond(n₁, n, n₂)
			@show n₁, n, n₂
		end
	end
	
	
	# # Check the number of three-spin interactions that have been set up
	# b == Nterms || error("Expected $Nterms three-spin interactions, but got $b")
	
	
	@show wedge
	return wedge
end




interferometry_wedge_pbc(4, 6, 44)