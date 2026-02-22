# 02/15/2026
# Implement the interferometer setup for 2D Kitaev model with periodic boundary condition along x direction
# Include two-body interactions and three-body interactions, and the constrictions of the interferometer by removing sites and bonds

using ITensors
using ITensorMPS
using HDF5
using MKL
using LinearAlgebra
using TimerOutputs



include("HoneycombLattice.jl")
include("Entanglement.jl")
include("TopologicalLoops.jl")
include("CustomObserver.jl")
include("interferometry_lattice.jl")


# Configure multithreading for BLAS/LAPACK operations
MKL_NUM_THREADS = 8
OPENBLAS_NUM_THREADS = 8
OMP_NUM_THREADS = 8

@info "BLAS configuration:" config = BLAS.get_config()
@info "BLAS threads:" num_threads = BLAS.get_num_threads()



# Set up physical parameters for the 2D Kitaev model and the lattice geometry parameters
const Jx::Float64 = 1.0
const Jy::Float64 = 1.0
const Jz::Float64 = 1.0
const κ::Float64 = -0.2
const Nx_unit = 4           # Number of unit cells in x-direction
const Ny_unit = 6           # Number of unit cells in y-direction
const Nx = 2 * Nx_unit      # Total width (honeycomb lattice)
const Ny = Ny_unit          # Height with periodic boundary condition
const N = Nx * Ny - 4       # Total number of sites

# Timing and profiling
const time_machine = TimerOutput()






let
  println(repeat("*", 200))
  println(repeat("*", 200))
  println("Obtain the ground state of the interferometry setup of the 2D Kitaev model using DMRG")
  println(repeat("*", 200))
  println(repeat("*", 200))
  println("")

  #******************************************************************************************************************************************
  #******************************************************************************************************************************************
  """ Set up the bonds on a honeycomb lattice for the interferometry setup and the corresponding two-body interactions in the Hamiltonian"""
  println(repeat("*", 200))
  println("Setting up the bonds on a honeycomb lattice")
  lattice = interferometry_lattice_pbc(Nx_unit, Ny_unit)
  number_of_bonds = length(lattice)
  
  
  println("\nSetting up two-body interactions in the Hamiltonian")
  os = OpSum()
  xbond::Int = 0
  ybond::Int = 0
  zbond::Int = 0
  
  
  # Set up the interferometer geometry
  geometry_profile = Int[]  
  for idx in 1 : Nx
    if idx == 1 || idx == Nx
      push!(geometry_profile, Ny_unit - 2)
    else
      push!(geometry_profile, Ny_unit)
    end
  end
  
	gauge_profile = Int[]
	for idx in 0:length(geometry_profile)
		append!(gauge_profile, sum(geometry_profile[1:idx]))
	end
	# @show geometry_profile, gauge_profile


  # Set up the two-body interaction terms in the Hamiltonian based on the bonds in the lattice
  for b in lattice
    # Use the first site in a tuple as the anchor point to determine the coordinates of the bond
    lattice_site = b.s1 
    
    # Determine the x coordinate of the lattice point n based on the input geometry
		xcoordinate = 0
		for idx in 1 : length(gauge_profile) - 1
			if lattice_site > gauge_profile[idx] && lattice_site <= gauge_profile[idx + 1]
				xcoordinate = idx
				break
			end
		end
		
		# Determine the y coordinate of the lattice point n based on the input geometry
		ycoordinate = 0
		for idx in 1 : length(gauge_profile) - 1
			if lattice_site > gauge_profile[idx] && lattice_site <= gauge_profile[idx + 1]
				tmp = lattice_site - gauge_profile[idx]
				ycoordinate = mod(tmp - 1, geometry_profile[idx]) + 1
				break
			end
		end
		
    # @show lattice_site, xcoordinate, ycoordinate  


    # Set up the two-body interaction terms based on the coordinates of the bond
    if iseven(xcoordinate) && xcoordinate != Nx
      os .+= -Jz, "Sz", b.s1, "Sz", b.s2
      zbond += 1
      @info "Added Sz-Sz bond" term = ("Jz", Jz, "Sz", b.s1, "Sz", b.s2)
    end

    if isodd(xcoordinate)
      if xcoordinate != 1 && xcoordinate != Nx - 1
        if abs(b.s1 - b.s2) == Ny 
          os .+= -Jy, "Sy", b.s1, "Sy", b.s2
          ybond += 1
          @info "Added Sy-Sy bond" term = ("Jy", Jy, "Sy", b.s1, "Sy", b.s2)
        else
          os .+= -Jx, "Sx", b.s1, "Sx", b.s2
          xbond += 1
          @info "Added Sx-Sx bond" term = ("Jx", Jx, "Sx", b.s1, "Sx", b.s2)
        end
      elseif xcoordinate == 1
        if ycoordinate == 1 || ycoordinate == 2
          if abs(b.s1 - b.s2) == Ny - 1
            os .+= -Jy, "Sy", b.s1, "Sy", b.s2
            ybond += 1
            @info "Added Sy-Sy bond" term = ("Jy", Jy, "Sy", b.s1, "Sy", b.s2)
          else
            os .+= -Jx, "Sx", b.s1, "Sx", b.s2
            xbond += 1
            @info "Added Sx-Sx bond" term = ("Jx", Jx, "Sx", b.s1, "Sx", b.s2)
          end
        else
          if abs(b.s1 - b.s2) == Ny
            os .+= -Jy, "Sy", b.s1, "Sy", b.s2
            ybond += 1
            @info "Added Sy-Sy bond" term = ("Jy", Jy, "Sy", b.s1, "Sy", b.s2) 
          else
            os .+= -Jx, "Sx", b.s1, "Sx", b.s2
            xbond += 1
            @info "Added Sx-Sx bond" term = ("Jx", Jx, "Sx", b.s1, "Sx", b.s2)
          end
        end
      elseif xcoordinate == Nx - 1
        if lattice_site in [36, 39]
          os .+= -Jx, "Sx", b.s1, "Sx", b.s2
          xbond += 1
          @info "Added Sx-Sx bond" term = ("Jx", Jx, "Sx", b.s1, "Sx", b.s2)
        elseif lattice_site in [37, 40]
          os .+= -Jy, "Sy", b.s1, "Sy", b.s2
          ybond += 1
          @info "Added Sy-Sy bond" term = ("Jy", Jy, "Sy", b.s1, "Sy", b.s2) 
        elseif (ycoordinate == 1 && abs(b.s1 - b.s2) == Ny) || (ycoordinate == 4 && abs(b.s1 - b.s2) == Ny - 1)
          os .+= -Jy, "Sy", b.s1, "Sy", b.s2
          ybond += 1
          @info "Added Sy-Sy bond" term = ("Jy", Jy, "Sy", b.s1, "Sy", b.s2)
        else
          os .+= -Jx, "Sx", b.s1, "Sx", b.s2
          xbond += 1
          @info "Added Sx-Sx bond" term = ("Jx", Jx, "Sx", b.s1, "Sx", b.s2)
        end
      end
    end
  end
  
  # Check whether number of bonds is correct 
  @show xbond, ybond, zbond, number_of_bonds
  if xbond + ybond + zbond != number_of_bonds
    error("The number of bonds in the Hamiltonian is not correct!")
  end
  
  
  println(repeat("*", 200), "\n")
  #******************************************************************************************************************************************
  #******************************************************************************************************************************************
  
  
  #******************************************************************************************************************************************
  #******************************************************************************************************************************************
  """Set up the wedges on a honeycomb lattice for the interferometry setup and the corresponding three-spin interactions in the Hamiltonian""" 
  # println("Setting up the wedgeds on a honeycomb lattice")
  # wedge = honeycomb_wedge_interferometry(Nx, Ny; yperiodic=true)
  # number_of_wedges = length(wedge)
  # @show number_of_wedges
  # for (idx, tmp) in enumerate(wedge)
  #   @show tmp.s1, tmp.s2, tmp.s3
  # end 
  # println(repeat("*", 200))
  # println("")

 
  # # Construct the three-spin interaction terms
  # println(repeat("*", 200))
  # println("Setting up three-body interactions in the Hamiltonian")
  
  # wedge_count::Int = 0
  # for w in wedge
  #   # Use the second term of each tuple as the anchor point to determine the coordinates of the wedge
  #   x_coordinate = div(w.s2 - 1, Ny) + 1
  #   y_coordinate = mod(w.s2 - 1, Ny) + 1

  #   if w.s1 in empty_sites || w.s2 in empty_sites || w.s3 in empty_sites
  #     effective_κ = α * κ
  #   else
  #     effective_κ = κ
  #   end


  #   # Set up the three-spin interaction terms for the odd columns
  #   if isodd(x_coordinate)
  #     if x_coordinate == 1
  #       if y_coordinate != Ny 
  #         os .+= effective_κ, "Sx", w.s1, "Sz", w.s2, "Sy", w.s3
  #         @info "Added three-spin term" term = ("Sx", w.s1, "Sz", w.s2, "Sy", w.s3, "kappa", effective_κ)
  #       else
  #         os .+= effective_κ, "Sy", w.s1, "Sz", w.s2, "Sx", w.s3
  #         @info "Added three-spin term" term = ("Sy", w.s1, "Sz", w.s2, "Sx", w.s3, "kappa", effective_κ)
  #       end
  #       wedge_count += 1
  #     else
  #       if abs(w.s1 - w.s2) == abs(w.s2 - w.s3) == Ny 
  #         os .+= effective_κ, "Sz", w.s1, "Sx", w.s2, "Sy", w.s3
  #         @info "Added three-spin term" term = ("Sz", w.s1, "Sx", w.s2, "Sy", w.s3, "kappa", effective_κ)
  #         wedge_count += 1
  #       elseif abs(w.s3 - w.s1) == 1
  #         os .+= effective_κ, "Sx", w.s1, "Sz", w.s2, "Sy", w.s3
  #         @info "Added three-spin term" term = ("Sx", w.s1, "Sz", w.s2, "Sy", w.s3, "kappa", effective_κ)
  #         wedge_count += 1
  #       elseif abs(w.s3 - w.s1) == Ny - 1
  #         os .+= effective_κ, "Sy", w.s1, "Sz", w.s2, "Sx", w.s3
  #         @info "Added three-spin term" term = ("Sy", w.s1, "Sz", w.s2, "Sx", w.s3, "kappa", effective_κ)
  #         wedge_count += 1
  #       else
  #         os .+= effective_κ, "Sz", w.s1, "Sy", w.s2, "Sx", w.s3
  #         @info "Added three-spin term" term = ("Sz", w.s1, "Sy", w.s2, "Sx", w.s3, "kappa", effective_κ)
  #         wedge_count += 1
  #       end
  #     end
  #   end


  #   # Set up the three-spin interaction terms for the even columns
  #   if iseven(x_coordinate)
  #     if x_coordinate == Nx 
  #       if abs(w.s3 - w.s1) == 1
  #         os .+= effective_κ, "Sy", w.s1, "Sz", w.s2, "Sx", w.s3
  #         @info "Added three-spin term" term = ("Sy", w.s1, "Sz", w.s2, "Sx", w.s3, "kappa", effective_κ)
  #       else
  #         os .+= effective_κ, "Sx", w.s1, "Sz", w.s2, "Sy", w.s3
  #         @info "Added three-spin term" term = ("Sx", w.s1, "Sz", w.s2, "Sy", w.s3, "kappa", effective_κ)
  #       end
  #       wedge_count += 1
  #     else
  #       if abs(w.s3 - w.s2) == abs(w.s2 - w.s1) == Ny
  #         os .+= effective_κ, "Sx", w.s1, "Sy", w.s2, "Sz", w.s3 
  #         @info "Added three-spin term" term = ("Sx", w.s1, "Sy", w.s2, "Sz", w.s3, "kappa", effective_κ)
  #         wedge_count += 1
  #       elseif abs(w.s3 - w.s1) == 1
  #         os .+= effective_κ, "Sy", w.s1, "Sz", w.s2, "Sx", w.s3
  #         @info "Added three-spin term" term = ("Sy", w.s1, "Sz", w.s2, "Sx", w.s3, "kappa", effective_κ)
  #         wedge_count += 1
  #       elseif abs(w.s3 - w.s1) == Ny - 1
  #         os .+= effective_κ, "Sx", w.s1, "Sz", w.s2, "Sy", w.s3
  #         @info "Added three-spin term" term = ("Sx", w.s1, "Sz", w.s2, "Sy", w.s3, "kappa", effective_κ)
  #         wedge_count += 1
  #       else
  #         os .+= effective_κ, "Sy", w.s1, "Sx", w.s2, "Sz", w.s3
  #         @info "Added three-spin term" term = ("Sy", w.s1, "Sx", w.s2, "Sz", w.s3, "kappa", effective_κ)
  #         wedge_count += 1
  #       end
  #     end
  #   end
  # end
  

  # # Check whether the number of three-spin interaction terms is correct 
  # if wedge_count != 3 * N - 2 * 2 * Ny
  #   error("The number of three-spin interaction terms is incorrect!")
  # end
   


  # # # Add loop operators long the y direction of the cylinder to access a specific topological sector
  # # loop_operator = ["Sx", "Sx", "Sz", "Sz", "Sz", "Sz"]            # Hard-coded for width-3 cylinders
  # # loop_indices = LoopList_RightTwist(Nx_unit, Ny_unit, "rings", "y")  
  # # @show loop_indices
  # println(repeat("*", 200))
  # println("")
  #******************************************************************************************************************************************
  #******************************************************************************************************************************************
  
  
  #******************************************************************************************************************************************
  #******************************************************************************************************************************************
  """Run DMRG simulation to find the ground-state wavefunction of the interferometry setup of the 2D Kitaev model""" 
  println(repeat("*", 200))
  println("Running DMRG simulations to find the ground-state wavefunction")

  
  
  # Initialize the wavefunction as a random MPS and set up the Hamiltonian as an MPO
  sites = siteinds("S=1/2", N)
  state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
  ψ₀ = randomMPS(sites, state, 10)
  H = MPO(os, sites)
  
  
  # Set up hyperparameters used in the DMRG simulations, including bond dimensions, cutoff etc.
  nsweeps = 8
  maxdim  = [20, 80, 350]
  cutoff  = [1E-10]
  eigsolve_krylovdim = 50
  
  # Add noise terms to prevent DMRG from getting stuck in a local minimum
  # noise = [1E-6, 1E-7, 1E-8, 0.0]
  
  
  # Measure one-point functions of the initial state
  Sx₀ = expect(ψ₀, "Sx", sites = 1 : N)
  Sy₀ = im * expect(ψ₀, "iSy", sites = 1 : N)
  Sz₀ = expect(ψ₀, "Sz", sites = 1 : N)


  # Construct a custom observer and stop the DMRG calculation early if criteria are met
  # custom_observer = DMRGObserver(; energy_tol=1E-9, minsweeps=2, energy_type=Float64)
  custom_observer = CustomObserver()
  @show custom_observer.etolerance
  @show custom_observer.minsweeps
  @timeit time_machine "dmrg simulation" begin
    energy, ψ = dmrg(H, ψ₀; nsweeps, maxdim, cutoff, eigsolve_krylovdim, observer = custom_observer)
  end

  println("Final ground-state energy = $energy")
  println(repeat("*", 200), "\n")
  #******************************************************************************************************************************************
  #******************************************************************************************************************************************

  
  
  #******************************************************************************************************************************************
  #******************************************************************************************************************************************
  """Take measurements of the optimized ground-state wavefunction"""
  
  # Measure local observables (one-point functions)
  @timeit time_machine "one-point functions" begin
    Sx = expect(ψ, "Sx", sites = 1 : N)
    Sy = expect(ψ, "Sy", sites = 1 : N)
    Sz = expect(ψ, "Sz", sites = 1 : N)
  end

  
  # Measure spin correlation functions (two-point functions)
  @timeit time_machine "two-point functions" begin
    xxcorr = correlation_matrix(ψ, "Sx", "Sx", sites = 1 : N)
    zzcorr = correlation_matrix(ψ, "Sz", "Sz", sites = 1 : N)
    yycorr = -1.0 * correlation_matrix(ψ, "iSy", "iSy", sites = 1 : N)
  end


  # Measure the expectation values of the plaquette operators (six-point correlators) around each hexagon
  println(repeat("*", 200))
  println("Measuring the expectation values of the plaquette operators around each hexagon")
  plaquette_operator = Vector{String}(["Z", "X", "iY", "Z", "X", "iY"])
  plaquette_inds = [
    # --- Right edge  ---
    1  6  12  17  11   5;
    2  7  13  18  12   6;
    3  9  15  20  14   8;
    4  10 16  21  15   9;
    # --- Bulk region ---
    12 18 24  29  23  17;
    13 19 25  30  24  18;
    14 20 26  31  25  19;
    15 21 27  32  26  20;
    16 22 28  33  27  21;
    11 17 23  34  28  22;
    # --- Left edge  ---
    23 29 35  44  40  34;
    24 30 36  41  35  29;
    26 32 38  42  37  31;
    27 33 39  43  38  32;
  ]

  
  nplaquettes = size(plaquette_inds, 1)
  plaquette_vals = zeros(Float64, nplaquettes)

  @timeit time_machine "plaquette operators" begin
    for idx in 1:nplaquettes
      indices = plaquette_inds[idx, :]
      os_w = OpSum()
      os_w .+= plaquette_operator[1], indices[1], 
        plaquette_operator[2], indices[2], 
        plaquette_operator[3], indices[3], 
        plaquette_operator[4], indices[4], 
        plaquette_operator[5], indices[5], 
        plaquette_operator[6], indices[6]
      W = MPO(os_w, sites)

      # There is a minus sign becuase of the two "iY" operators
      plaquette_vals[idx] = -1.0 * real(inner(ψ', W, ψ))
    end
  end
  
  println(repeat("*", 200), "\n")
  #******************************************************************************************************************************************
  #******************************************************************************************************************************************
  

  #******************************************************************************************************************************************
  #****************************************************************************************************************************************** 
  println(repeat("*", 200))
  println("Summary of results: \n")
  
  # # Check the variance of the energy
  # @timeit time_machine "compaute the variance" begin
  #   H2 = inner(H, ψ, H, ψ)
  #   E₀ = inner(ψ', H, ψ)
  #   variance = H2 - E₀^2
  # end
  # println("Variance of the energy is $variance")
  # println("")
  
  # Check the expectation values of the plaquette operators
  println("\nExpectation values of the plaquette operators:")
  @show plaquette_vals
  

  # Check the bond dimension of the optimized MPS wavefunction
  println("\nBond dimension of the optimized MPS wavefunction:")
  @show linkdims(ψ)
 

  # # Check one-point functions
  # println("Expectation values of one-point functions <Sx>, <Sy>, and <Sz>:")
  # @show Sx
  # @show Sy
  # @show Sz

  # println(repeat("*", 200), "\n")
  #******************************************************************************************************************************************
  #******************************************************************************************************************************************

  
  @show time_machine
  h5open("data/interferometry_pbc_kappa$(κ)_bond300.h5", "cw") do file
    write(file, "psi", ψ)
    write(file, "E0", energy)
    write(file, "Ehist", custom_observer.ehistory)
    write(file, "Sx",  Sx)
    write(file, "Cxx", xxcorr)
    write(file, "Sy", Sy)
    write(file, "Cyy", yycorr)
    write(file, "Sz",  Sz)
    write(file, "Czz", zzcorr)
    write(file, "Plaquette", plaquette_vals)
    write(file, "Bond", linkdims(ψ))
  end

  return
end