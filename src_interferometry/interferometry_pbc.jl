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
  #***************************************************************************************************************
  #***************************************************************************************************************
  println(repeat("*", 200))
  println(repeat("*", 200))
  println("Obtain the ground state of the interferometry setup of the 2D Kitaev model using DMRG")
  println(repeat("*", 200))
  println(repeat("*", 200))
  println("")

  
  # # Scaling factor for the Kitaev interaction to set up the interferometry
  # α = 1E-6

  
  # Set up the bonds on a honeycomb lattice for the two-spin interactions
  println(repeat("*", 200))
  println("Setting up the bonds on a honeycomb lattice")
  lattice = honeycomb_lattice_interferometry(Nx, Ny; yperiodic=true)
  number_of_bonds = length(lattice)
  @show number_of_bonds
  for (idx, bond) in enumerate(lattice)
    @show bond.s1, bond.s2
  end
  println(repeat("*", 200))
  println("")

  
  # Set up the wedge terms on a honeycomb lattice for the three-spin interactions
  println(repeat("*", 200))
  println("Setting up the wedgeds on a honeycomb lattice")
  wedge = honeycomb_wedge_interferometry(Nx, Ny; yperiodic=true)
  number_of_wedges = length(wedge)
  @show number_of_wedges
  for (idx, tmp) in enumerate(wedge)
    @show tmp.s1, tmp.s2, tmp.s3
  end 
  println(repeat("*", 200))
  println("")


  # Remove sites to set up the interferometry
  empty_sites = Set{Int64}([3, 6, 44, 47])
  y_periodic = true

  
  #***************************************************************************************************************
  #***************************************************************************************************************
  # Construct the Kitaev interaction and electron hoppings 
  os = OpSum()
  xbond::Int = 0
  ybond::Int = 0
  zbond::Int = 0
  
  println(repeat("*", 200))
  println("Setting up two-body interactions in the Hamiltonian")

  for b in lattice
    # Set up the hopping terms for spin-up and spin-down electrons
    # os .+= -t, "Cdagup", b.s1, "Cup", b.s2
    # os .+= -t, "Cdagup", b.s2, "Cup", b.s1
    # os .+= -t, "Cdagdn", b.s1, "Cdn", b.s2
    # os .+= -t, "Cdagdn", b.s2, "Cdn", b.s1
    
    # Set up the anisotropic two-body Kitaev interaction
    if b.s1 in empty_sites || b.s2 in empty_sites
      effective_Jx = α * Jx
      effective_Jy = α * Jy
      effective_Jz = α * Jz
    else
      effective_Jx = Jx
      effective_Jy = Jy
      effective_Jz = Jz
    end

    tmp_x = div(b.s1 - 1, Ny) + 1
    if iseven(tmp_x)
      os .+= -effective_Jz, "Sz", b.s1, "Sz", b.s2
      zbond += 1
      @info "Added Sz-Sz bond" term = ("Jz", effective_Jz, "Sz", b.s1, "Sz", b.s2)
    else
      if tmp_x == 1
        if abs(b.s1 - b.s2) == Ny 
          os .+= -effective_Jx, "Sx", b.s1, "Sx", b.s2
          xbond += 1
          @info "Added Sx-Sx bond" term = ("Jx", effective_Jx, "Sx", b.s1, "Sx", b.s2)
        else
          os .+= -effective_Jy, "Sy", b.s1, "Sy", b.s2
          ybond += 1
          @info "Added Sy-Sy bond" term = ("Jy", effective_Jy, "Sy", b.s1, "Sy", b.s2)
        end
      else
        if abs(b.s1 - b.s2) == Ny 
          os .+= -effective_Jy, "Sy", b.s1, "Sy", b.s2
          ybond += 1
          @info "Added Sy-Sy bond" term = ("Jy", effective_Jy, "Sy", b.s1, "Sy", b.s2)
        else
          os .+= -effective_Jx, "Sx", b.s1, "Sx", b.s2
          xbond += 1
          @info "Added Sx-Sx bond" term = ("Jx", effective_Jx, "Sx", b.s1, "Sx", b.s2)
        end
      end
    end
  end
  
  # Check whether number of bonds is correct 
  @show xbond, ybond, zbond, number_of_bonds
  if xbond + ybond + zbond != number_of_bonds
    error("The number of bonds in the Hamiltonian is not correct!")
  end
  println("")
 
  #***************************************************************************************************************
  #***************************************************************************************************************
  # Construct the three-spin interaction terms
  println(repeat("*", 200))
  println("Setting up three-body interactions in the Hamiltonian")
  
  wedge_count::Int = 0
  for w in wedge
    # Use the second term of each tuple as the anchor point to determine the coordinates of the wedge
    x_coordinate = div(w.s2 - 1, Ny) + 1
    y_coordinate = mod(w.s2 - 1, Ny) + 1

    if w.s1 in empty_sites || w.s2 in empty_sites || w.s3 in empty_sites
      effective_κ = α * κ
    else
      effective_κ = κ
    end


    # Set up the three-spin interaction terms for the odd columns
    if isodd(x_coordinate)
      if x_coordinate == 1
        if y_coordinate != Ny 
          os .+= effective_κ, "Sx", w.s1, "Sz", w.s2, "Sy", w.s3
          @info "Added three-spin term" term = ("Sx", w.s1, "Sz", w.s2, "Sy", w.s3, "kappa", effective_κ)
        else
          os .+= effective_κ, "Sy", w.s1, "Sz", w.s2, "Sx", w.s3
          @info "Added three-spin term" term = ("Sy", w.s1, "Sz", w.s2, "Sx", w.s3, "kappa", effective_κ)
        end
        wedge_count += 1
      else
        if abs(w.s1 - w.s2) == abs(w.s2 - w.s3) == Ny 
          os .+= effective_κ, "Sz", w.s1, "Sx", w.s2, "Sy", w.s3
          @info "Added three-spin term" term = ("Sz", w.s1, "Sx", w.s2, "Sy", w.s3, "kappa", effective_κ)
          wedge_count += 1
        elseif abs(w.s3 - w.s1) == 1
          os .+= effective_κ, "Sx", w.s1, "Sz", w.s2, "Sy", w.s3
          @info "Added three-spin term" term = ("Sx", w.s1, "Sz", w.s2, "Sy", w.s3, "kappa", effective_κ)
          wedge_count += 1
        elseif abs(w.s3 - w.s1) == Ny - 1
          os .+= effective_κ, "Sy", w.s1, "Sz", w.s2, "Sx", w.s3
          @info "Added three-spin term" term = ("Sy", w.s1, "Sz", w.s2, "Sx", w.s3, "kappa", effective_κ)
          wedge_count += 1
        else
          os .+= effective_κ, "Sz", w.s1, "Sy", w.s2, "Sx", w.s3
          @info "Added three-spin term" term = ("Sz", w.s1, "Sy", w.s2, "Sx", w.s3, "kappa", effective_κ)
          wedge_count += 1
        end
      end
    end


    # Set up the three-spin interaction terms for the even columns
    if iseven(x_coordinate)
      if x_coordinate == Nx 
        if abs(w.s3 - w.s1) == 1
          os .+= effective_κ, "Sy", w.s1, "Sz", w.s2, "Sx", w.s3
          @info "Added three-spin term" term = ("Sy", w.s1, "Sz", w.s2, "Sx", w.s3, "kappa", effective_κ)
        else
          os .+= effective_κ, "Sx", w.s1, "Sz", w.s2, "Sy", w.s3
          @info "Added three-spin term" term = ("Sx", w.s1, "Sz", w.s2, "Sy", w.s3, "kappa", effective_κ)
        end
        wedge_count += 1
      else
        if abs(w.s3 - w.s2) == abs(w.s2 - w.s1) == Ny
          os .+= effective_κ, "Sx", w.s1, "Sy", w.s2, "Sz", w.s3 
          @info "Added three-spin term" term = ("Sx", w.s1, "Sy", w.s2, "Sz", w.s3, "kappa", effective_κ)
          wedge_count += 1
        elseif abs(w.s3 - w.s1) == 1
          os .+= effective_κ, "Sy", w.s1, "Sz", w.s2, "Sx", w.s3
          @info "Added three-spin term" term = ("Sy", w.s1, "Sz", w.s2, "Sx", w.s3, "kappa", effective_κ)
          wedge_count += 1
        elseif abs(w.s3 - w.s1) == Ny - 1
          os .+= effective_κ, "Sx", w.s1, "Sz", w.s2, "Sy", w.s3
          @info "Added three-spin term" term = ("Sx", w.s1, "Sz", w.s2, "Sy", w.s3, "kappa", effective_κ)
          wedge_count += 1
        else
          os .+= effective_κ, "Sy", w.s1, "Sx", w.s2, "Sz", w.s3
          @info "Added three-spin term" term = ("Sy", w.s1, "Sx", w.s2, "Sz", w.s3, "kappa", effective_κ)
          wedge_count += 1
        end
      end
    end
  end
  

  # Check whether the number of three-spin interaction terms is correct 
  if wedge_count != 3 * N - 2 * 2 * Ny
    error("The number of three-spin interaction terms is incorrect!")
  end
   

  # Two optional parts: (1) adding a Zeeman field to each site; (2) adding loop perturbations to access a specific topological sector
  # # Add the Zeeman terms into the Hamiltonian, which breaks the integrability
  # if h > 1e-8
  #   for site in lattice_sites
  #     os .+= -h, "Sx", site
  #     os .+= -h, "Sy", site
  #     os .+= -h, "Sz", site
  #   end
  # end


  # # Add loop operators long the y direction of the cylinder to access a specific topological sector
  # loop_operator = ["Sx", "Sx", "Sz", "Sz", "Sz", "Sz"]            # Hard-coded for width-3 cylinders
  # loop_indices = LoopList_RightTwist(Nx_unit, Ny_unit, "rings", "y")  
  # @show loop_indices
  println(repeat("*", 200))
  println("")
  #***************************************************************************************************************
  #*************************************************************************************************************** 
  
  
  #***************************************************************************************************************
  #***************************************************************************************************************
  # Run DMRG simulations to find the ground-state wavefunction
  println(repeat("*", 200))
  println("Running DMRG simulations to find the ground-state wavefunction")

  # Initialize the wavefunction as a random MPS and set up the Hamiltonian as an MPO
  sites = siteinds("S=1/2", N)
  state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
  ψ₀ = randomMPS(sites, state, 10)
  H = MPO(os, sites)
  
  # Set up hyperparameters used in the DMRG simulations, including bond dimensions, cutoff etc.
  nsweeps = 10
  maxdim  = [20, 60, 100, 500, 800, 1000, 1500, 3000]
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
  println(repeat("*", 200))
  #***************************************************************************************************************
  #***************************************************************************************************************

  #***************************************************************************************************************
  #***************************************************************************************************************
  # Take measurements of the optimized ground-state wavefunction
  
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
  # Set up the plaquette operators and corresponding indices for all hexagons
  println(repeat("*", 200))
  println("Measuring the expectation values of the plaquette operators around each hexagon")
  plaquette_op = Vector{String}(["Z", "iY", "X", "Z", "iY", "X"])
  plaquette_inds = PlaquetteListInterferometry(Nx_unit, Ny_unit, "rings", false)
  
  for idx in 1 : size(plaquette_inds, 1)
    @show plaquette_inds[idx, :]
  end
  println("")

  nplaquettes = size(plaquette_inds, 1)
  plaquette_vals = zeros(Float64, nplaquettes)

  @timeit time_machine "plaquette operators" begin
    for idx in 1:nplaquettes
      indices  = plaquette_inds[idx, :]
      os_w = OpSum()
      os_w .+= plaquette_op[1], indices[1], 
        plaquette_op[2], indices[2], 
        plaquette_op[3], indices[3], 
        plaquette_op[4], indices[4], 
        plaquette_op[5], indices[5], 
        plaquette_op[6], indices[6]
      W = MPO(os_w, sites)

      # There is a minus sign becuase of the two "iY" operators
      plaquette_vals[idx] = -1.0 * real(inner(ψ', W, ψ))
    end
  end
  println("The expectation values of the plaquette operators around each hexagon are:")
  @show plaquette_vals

  println(repeat("*", 200))
  println("")
  #***************************************************************************************************************
  #***************************************************************************************************************
  

  #***************************************************************************************************************
  #***************************************************************************************************************
  println(repeat("*", 200))
  println("Summary of results:")
  println("")

  # Check the variance of the energy
  @timeit time_machine "compaute the variance" begin
    H2 = inner(H, ψ, H, ψ)
    E₀ = inner(ψ', H, ψ)
    variance = H2 - E₀^2
  end
  println("Variance of the energy is $variance")
  println("")
  
  # Check the expectation values of the plaquette operators
  println("Expectation values of the plaquette operators:")
  @show plaquette_vals
  println("")

  # Check one-point functions
  println("Expectation values of one-point functions <Sx>, <Sy>, and <Sz>:")
  @show Sx
  @show Sy
  @show Sz

  println(repeat("*", 200))
  println("")
  #***************************************************************************************************************
  #***************************************************************************************************************

  
  @show time_machine
  h5open("data/interferometry_kappa$(κ).h5", "cw") do file
    write(file, "psi", ψ)
    write(file, "E0", energy)
    write(file, "E0variance", variance)
    write(file, "Ehist", custom_observer.ehistory)
    write(file, "Bond", custom_observer.chi)
    # write(file, "Entropy", SvN)
    write(file, "Sx0", Sx₀)
    write(file, "Sx",  Sx)
    write(file, "Cxx", xxcorr)
    write(file, "Sy0", Sy₀)
    write(file, "Sy", Sy)
    write(file, "Cyy", yycorr)
    write(file, "Sz0", Sz₀)
    write(file, "Sz",  Sz)
    write(file, "Czz", zzcorr)
    write(file, "Plaquette", plaquette_vals)
  end

  return
end