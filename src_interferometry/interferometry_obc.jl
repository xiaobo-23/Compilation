# 12/10/2025
# Simuulate the interferometry setup of the 2D Kitaev model on a honeycomb lattice using DMRG
# The code constructs the Hamiltonian and obtain the ground-state wave function; used as the input for state compilation


using ITensors
using ITensorMPS
using HDF5
using MKL
using LinearAlgebra
using TimerOutputs


include("interferometry_lattice.jl")
include("Entanglement.jl")
include("TopologicalLoops.jl")
include("CustomObserver.jl")


# Set up parameters used in multithreading for BLAS/LAPACK and Block sparse multithreading
MKL_NUM_THREADS = 8
OPENBLAS_NUM_THREADS = 8
OMP_NUM_THREADS = 8
@show BLAS.get_config()
@show BLAS.get_num_threads()


# Set up the interferometry system with two constrictions
const Nx_unit = 8
const Ny_unit = 3
const Nx = 2 * Nx_unit
const Ny = Ny_unit + 1
const N = 58  # Total number of sites after removing sites for interferometry


const Jx::Float64 = 1.0
const Jy::Float64 = 1.0 
const Jz::Float64 = 1.0 
const κ::Float64 = -0.2
# const h::Float64 = 0.0
const time_machine = TimerOutput()  # Timing and profiling


let
  #***************************************************************************************************************
  #***************************************************************************************************************
  """
    Obtain the ground state of the interferometry setup using DMRG
  """

  header = repeat('#', 200)
  println(header)
  println(header)
  println("Obtain the ground state of the interferometry setup of the 2D Kitaev model using DMRG")
  println("")
  println("")
  
  
  """
    Set up the bonds on the interferometry lattice
    Use these bonds to set up the two-body Kitaev interactions in the Hamiltonian
  """
  
  println(repeat("*", 100))
  println("Setting up the bonds on the interferometry lattice")
  lattice = interferometry_lattice_obc(Nx, Ny, N)
  number_of_bonds = length(lattice)
  
  # println("\nPrinting bonds on the interferometry lattice:")
  # for (idx, bond) in enumerate(lattice)
  #   @show bond.s1, bond.s2
  # end
  println("")
  
  
  """
    Set up the wedges on the interferometry lattice
    Use these wedges to set up the three-spin interaction terms in the Hamiltonian
  """

  println(repeat("*", 100))
  println("Setting up the three-spin interaction on the interferometry lattice")
  wedge = interferometry_wedge(Nx, Ny, N)
  number_of_wedges = length(wedge)

  # println("\nPrinting wedges on the interferometry lattice:")
  # for idx in 1:length(wedge)
  #   if isassigned(wedge, idx)
  #     tmp = wedge[idx]
  #     @show idx, tmp.s1, tmp.s2, tmp.s3
  #   end
  # end 
  println("")


  """
    Construct the two-body interactions in the Hamiltonian
  """
  
  println(repeat("*", 100))
  println("Setting up two-body interactions in the Hamiltonian")
  

  # Define the constrictions on the interferometry lattice
  constriction₁ = [17, 20]
  constriction₂ = [39, 42]
  α = 4     # A scaling factor to make the interaction on the constriction stronger 


  # Set up the width profile and gauge to set up the x and y coordinates for each lattice site
  width_profile = Int[]
	for i in 1:5
		append!(width_profile, [3, 4, 4])
	end
	push!(width_profile, 3)

  
  x_gauge = Int[]
  for idx in 0:length(width_profile)
		append!(x_gauge, sum(width_profile[1:idx]))
	end
  println("\nThe gauge for x coordinates of each lattice site is:")
  @show x_gauge


  # Set up counters for different types of bonds 
  xbond::Int = 0
  ybond::Int = 0
  zbond::Int = 0


  # Loop through all the bonds in the lattice to set up the two-body interactions 
  os = OpSum()
  for b in lattice
    if (b.s1 == constriction₁[1] && b.s2 == constriction₁[2]) || (b.s1 == constriction₁[2] && b.s2 == constriction₁[1]) || 
       (b.s1 == constriction₂[1] && b.s2 == constriction₂[2]) || (b.s1 == constriction₂[2] && b.s2 == constriction₂[1])
      effective_Jx = α * Jx
      effective_Jy = α * Jy
      effective_Jz = α * Jz
    else
      effective_Jx = Jx
      effective_Jy = Jy
      effective_Jz = Jz
    end

    # Determine x coordinate of the first site in the bond
    x = 0
		for idx in 1 : length(x_gauge) - 1
			if b.s1 > x_gauge[idx] && b.s1 <= x_gauge[idx + 1]
				x = idx
				break
			end
		end
    # @show b.s1, x


    # Set up the two-body interaction terms based on the bond type
    if iseven(x)
      os .+= -effective_Jz, "Sz", b.s1, "Sz", b.s2
      zbond += 1
      # @info "Added Sz-Sz bond" term = ("Jz", effective_Jz, "Sz", b.s1, "Sz", b.s2)
    else
      if abs(b.s1 - b.s2) == Ny 
        os .+= -effective_Jx, "Sx", b.s1, "Sx", b.s2
        xbond += 1
        # @info "Added Sx-Sx bond" term = ("Jx", effective_Jx, "Sx", b.s1, "Sx", b.s2)
      elseif abs(b.s1 - b.s2) == Ny - 1
        os .+= -effective_Jy, "Sy", b.s1, "Sy", b.s2
        ybond += 1
        # @info "Added Sy-Sy bond" term = ("Jy", effective_Jy, "Sy", b.s1, "Sy", b.s2)
      end
    end
  end
  
  
  # Check whether the sum of all types of bonds is equal to the total number of bonds
  println("\nChecking the number of bonds in the Hamiltonian:")
  @show xbond, ybond, zbond, number_of_bonds
  if xbond + ybond + zbond != number_of_bonds
    error("The number of bonds in the Hamiltonian is not correct!")
  end
  println("")
 
  
  #***************************************************************************************************************
  #***************************************************************************************************************
  """
    Construct the three-body interactions in the Hamiltonian
  """
  println(repeat("*", 200))
  println("Setting up three-body interactions in the Hamiltonian")
  
  
  wedge_count::Int = 0
  for w in wedge
    # if w.s2 == constriction₁[1] || w.s2 == constriction₁[2] || 
    #   w.s2 == constriction₂[1] || w.s2 == constriction₂[2]
    #   effective_κ = α * κ
    # else
    #   effective_κ = κ
    # end
    effective_κ = κ

    
    # Determine the x coordinate of the second site and use the second site as the anchor point 
    x = 0
		for idx in 1 : length(x_gauge) - 1
			if w.s2 > x_gauge[idx] && w.s2 <= x_gauge[idx + 1]
				x = idx
				break
			end
		end
    # @show w.s2, x


    if abs(w.s1 - w.s3) == 1
      if isodd(x)
        os .+= effective_κ, "Sy", w.s1, "Sz", w.s2, "Sx", w.s3
        @info "Added three-spin term" term = ("Sy", w.s1, "Sz", w.s2, "Sx", w.s3, "kappa", effective_κ)
      else
        os .+= effective_κ, "Sx", w.s1, "Sz", w.s2, "Sy", w.s3
        @info "Added three-spin term" term = ("Sx", w.s1, "Sz", w.s2, "Sy", w.s3, "kappa", effective_κ)
      end
      wedge_count += 1
    else
      if isodd(x)
        if abs(w.s3 - w.s2) == Ny
          os .+= effective_κ, "Sz", w.s1, "Sy", w.s2, "Sx", w.s3
          @info "Added three-spin term" term = ("Sz", w.s1, "Sy", w.s2, "Sx", w.s3, "kappa", effective_κ)    
        else
          os .+= effective_κ, "Sz", w.s1, "Sx", w.s2, "Sy", w.s3
          @info "Added three-spin term" term = ("Sz", w.s1, "Sx", w.s2, "Sy", w.s3, "kappa", effective_κ)
        end
        wedge_count += 1
      else
        if abs(w.s2 - w.s1) == Ny 
          os .+= effective_κ, "Sx", w.s1, "Sy", w.s2, "Sz", w.s3 
          @info "Added three-spin term" term = ("Sx", w.s1, "Sy", w.s2, "Sz", w.s3, "kappa", effective_κ)
        else
          os .+= effective_κ, "Sy", w.s1, "Sx", w.s2, "Sz", w.s3
          @info "Added three-spin term" term = ("Sy", w.s1, "Sx", w.s2, "Sz", w.s3, "kappa", effective_κ)
        end
        wedge_count += 1
      end
    end
  end
  

  # Check to make sure the number of three-spin interaction terms is correct
  println("\nChecking the number of three-spin interaction terms in the Hamiltonian:")
  @show wedge_count
  if wedge_count != 122
    error("The number of three-spin interaction terms is incorrect!")
  end
  println("")
  #***************************************************************************************************************
  #*************************************************************************************************************** 
  
  
  #***************************************************************************************************************
  #***************************************************************************************************************
  """
    Obtain the ground-state wavefunction using DMRG
  """
  println(repeat("*", 100))
  println("Running DMRG simulations to find the ground-state wavefunction")

  
  # Initialize the wavefunction as a random MPS and set up the Hamiltonian as an MPO
  sites = siteinds("S=1/2", N)
  state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
  ψ₀ = randomMPS(sites, state, 8)
  H = MPO(os, sites)
  
  
  # Set up hyperparameters used in the DMRG simulations, including bond dimensions, cutoff etc.
  nsweeps = 2
  maxdim  = [20, 60, 100, 500, 800, 1000]
  cutoff  = [1E-10]
  eigsolve_krylovdim = 50
  # noise = [1E-6, 1E-7, 1E-8, 0.0]   # Add a noise term to prevent DMRG from getting stuck in local minima
  
  
  # Measure one-point functions of the initial state
  Sx₀ = expect(ψ₀, "Sx", sites = 1 : N)
  Sy₀ = -im * expect(ψ₀, "iSy", sites = 1 : N)
  Sz₀ = expect(ψ₀, "Sz", sites = 1 : N)


  # Construct a custom observer and stop the DMRG calculation early if criteria are met
  custom_observer = CustomObserver()
  @show custom_observer.etolerance
  @show custom_observer.minsweeps
  @timeit time_machine "dmrg simulation" begin
    energy, ψ = dmrg(H, ψ₀; nsweeps, maxdim, cutoff, eigsolve_krylovdim, observer = custom_observer)
  end

  println("Final ground-state energy = $energy")
  println("")
  #***************************************************************************************************************
  #***************************************************************************************************************

  #***************************************************************************************************************
  #***************************************************************************************************************
  """
    Measure various observables from the ground-state wavefunction
  """
  
  # Measure local observables (one-point functions)
  @timeit time_machine "one-point functions" begin
    Sx = expect(ψ, "Sx", sites = 1 : N)
    Sy = -im * expect(ψ, "iSy", sites = 1 : N)
    Sz = expect(ψ, "Sz", sites = 1 : N)
  end

  # Measure spin correlation functions (two-point functions)
  @timeit time_machine "two-point functions" begin
    xxcorr = correlation_matrix(ψ, "Sx", "Sx", sites = 1 : N)
    zzcorr = correlation_matrix(ψ, "Sz", "Sz", sites = 1 : N)
    yycorr = -1.0 * correlation_matrix(ψ, "iSy", "iSy", sites = 1 : N)
  end



  """
    Measure the expectation values of the plaquette operators (six-point correlators) on each hexagon
  """

  # Set up the plaquette operators and corresponding indices for all hexagons
  println(repeat("*", 100))
  println("Measuring the expectation values of the plaquette operators on each hexagon")
  plaquette = Vector{String}(["Z", "iY", "X", "Z", "iY", "X"])


  plaquette_indices = [
    1 5 9 12 8 4;
    2 6 10 13 9 5;
    3 7 11 14 10 6;
    9 13 16 19 15 12;
    10 14 17 20 16 13;
    17 21 24 27 23 20;
    18 22 25 28 24 21;
    23 27 31 34 30 26;
    24 28 32 35 31 27;
    25 29 33 36 32 28;
    31 35 38 41 37 34;
    32 36 39 42 38 35;
    39 43 46 49 45 42;
    40 44 47 50 46 43;
    45 49 53 56 52 48;
    46 50 54 57 53 49;
    47 51 55 58 54 50
  ]
  # plaquette_inds = PlaquetteListInterferometry(Nx_unit, Ny_unit, "rings", false)


  nplaquettes = size(plaquette_indices, 1)
  plaquette_vals = zeros(Float64, nplaquettes)
  for idx in 1:nplaquettes
    indices = plaquette_indices[idx, :]
    
    # Construct the MPO for the plaquette operator
    os_w = OpSum()
    os_w .+= plaquette[1], indices[1], 
      plaquette[2], indices[2], 
      plaquette[3], indices[3], 
      plaquette[4], indices[4], 
      plaquette[5], indices[5], 
      plaquette[6], indices[6]
    W = MPO(os_w, sites)

    # Compute the expectation value of the plaquette operator using MPS-MPO contraction
    plaquette_vals[idx] = -1.0 * real(inner(ψ', W, ψ))
  end

  println("The expectation values of the plaquette operators on each hexagon are:")
  for idx in 1:nplaquettes
    println("Plaquette $idx : ", plaquette_vals[idx])
  end
  println("")
  # ***************************************************************************************************************
  # ***************************************************************************************************************
  


  # #***************************************************************************************************************
  # #***************************************************************************************************************
  # println(repeat("*", 200))
  # println("Summary of results:")
  # println("")

  # # Check the variance of the energy
  # @timeit time_machine "compaute the variance" begin
  #   H2 = inner(H, ψ, H, ψ)
  #   E₀ = inner(ψ', H, ψ)
  #   variance = H2 - E₀^2
  # end
  # println("Variance of the energy is $variance")
  # println("")
  
  # # Check the expectation values of the plaquette operators
  # println("Expectation values of the plaquette operators:")
  # @show plaquette_vals
  # println("")

  # # Check one-point functions
  # println("Expectation values of one-point functions <Sx>, <Sy>, and <Sz>:")
  # @show Sx
  # @show Sy
  # @show Sz

  # println(repeat("*", 200))
  # println("")
  # #***************************************************************************************************************
  # #***************************************************************************************************************

  
  # @show time_machine
  # h5open("data/interferometry_kappa$(κ).h5", "cw") do file
  #   write(file, "psi", ψ)
  #   write(file, "E0", energy)
  #   write(file, "E0variance", variance)
  #   write(file, "Ehist", custom_observer.ehistory)
  #   write(file, "Bond", custom_observer.chi)
  #   # write(file, "Entropy", SvN)
  #   write(file, "Sx0", Sx₀)
  #   write(file, "Sx",  Sx)
  #   write(file, "Cxx", xxcorr)
  #   write(file, "Sy0", Sy₀)
  #   write(file, "Sy", Sy)
  #   write(file, "Cyy", yycorr)
  #   write(file, "Sz0", Sz₀)
  #   write(file, "Sz",  Sz)
  #   write(file, "Czz", zzcorr)
  #   write(file, "Plaquette", plaquette_vals)
  # end

  return
end