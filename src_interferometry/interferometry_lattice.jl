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

	
	# Construct the geometry profile dynamically based on Nx
	geometry_profile = Int[3, 4, 4, 4, 4, 4, 4, 3, 4, 4, 3, 4, 4, 3, 4, 4, 3, 4, 4, 4, 4, 4, 4, 3]
	# geometry_profile = Int[]
	# for i in 1:5
	# 	append!(geometry_profile, [3, 4, 4])
	# end
	# push!(geometry_profile, 3)
	

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
		# @show n, x

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
	Nwedge = 3 * Nsites - 4 * (Ny - 1) - 2 * (Nx - 6)
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