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
include("interferometry_plaquettes.jl")
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
const N  = Nx * Ny - 6  # Total number of sites after removing sites for interferometry


const Jx::Float64 = 1.0
const Jy::Float64 = 1.0 
const Jz::Float64 = 1.0 
const κ::Float64 = -0.8
const time_machine = TimerOutput()  # Timing and profiling


let
  #***************************************************************************************************************
  #***************************************************************************************************************
  """Obtain the ground state of the interferometry setup using DMRG"""  
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

  # Define the constrictions on the interferometry lattice
  # Set up the width and gauge profiles so that we can compute the x and y coordinates for each lattice point
  constriction₁ = [17, 20]
  constriction₂ = [39, 42]
  
  width_profile = Int[]
	for i in 1:5
		append!(width_profile, [3, 4, 4])
	end
	push!(width_profile, 3)
  
  # constriction₁ = [33, 36]
  # constriction₂ = [55, 58]
  # width_profile = Int[3, 4, 4, 4, 4, 4, 4, 3, 4, 4, 3, 4, 4, 3, 4, 4, 3, 4, 4, 4, 4, 4, 4, 3]


  lattice = interferometry_lattice_obc(Nx, Ny, N, width_profile)
  number_of_bonds = length(lattice)
  
  # println("\nPrinting bonds on the interferometry lattice:")
  # for (idx, bond) in enumerate(lattice)
  #   @show idx, bond.s1, bond.s2
  # end
  println("")
  
  

  """
    Set up the wedges on the interferometry lattice
    Use these wedges to set up the three-spin interaction terms in the Hamiltonian
  """
  println(repeat("*", 100))
  println("Setting up the three-spin interaction on the interferometry lattice")
  wedge = interferometry_wedge(Nx, Ny, N, width_profile)
  number_of_wedges = length(wedge)

  # println("\nPrinting wedges on the interferometry lattice:")
  # for idx in 1:length(wedge)
  #   if isassigned(wedge, idx)
  #     tmp = wedge[idx]
  #     @show idx, tmp.s1, tmp.s2, tmp.s3
  #   end
  # end 
  println("")
  # @show length(wedge)


  #***************************************************************************************************************
  """Construct the two-body interactions in the Hamiltonian"""
  println(repeat("*", 100))
  println("Setting up two-body interactions in the Hamiltonian")
  
  
  # A scaling factor to make the interaction on the constriction stronger 
  α = 4   

  x_gauge = Int[]
  for idx in 0:length(width_profile)
		append!(x_gauge, sum(width_profile[1:idx]))
	end
  # println("\nThe gauge for x coordinates of each lattice site is:")
  # @show x_gauge

  
  # Set up counters for different types of bonds 
  xbond::Int = 0
  ybond::Int = 0
  zbond::Int = 0


  # Loop through all the bonds in the lattice and set up the two-body interactions 
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
      @info "Added Sz-Sz bond" term = ("Jz", effective_Jz, "Sz", b.s1, "Sz", b.s2)
    else
      if abs(b.s1 - b.s2) == Ny 
        os .+= -effective_Jx, "Sx", b.s1, "Sx", b.s2
        xbond += 1
        @info "Added Sx-Sx bond" term = ("Jx", effective_Jx, "Sx", b.s1, "Sx", b.s2)
      elseif abs(b.s1 - b.s2) == Ny - 1
        os .+= -effective_Jy, "Sy", b.s1, "Sy", b.s2
        ybond += 1
        @info "Added Sy-Sy bond" term = ("Jy", effective_Jy, "Sy", b.s1, "Sy", b.s2)
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
  """Construct three-spin interactions in the Hamiltonian"""
  if abs(κ) > 1e-8
    println(repeat("*", 200))
    println("Setting up three-body interactions in the Hamiltonian")
     
    wedge_count::Int = 0
    for w in wedge
      # Determine the x coordinate of the middle site and use the middle site as the anchor point
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
          os .+= κ, "Sy", w.s1, "Sz", w.s2, "Sx", w.s3
          @info "Added three-spin term" term = ("Sy", w.s1, "Sz", w.s2, "Sx", w.s3, "kappa", κ)
        else
          os .+= κ, "Sx", w.s1, "Sz", w.s2, "Sy", w.s3
          @info "Added three-spin term" term = ("Sx", w.s1, "Sz", w.s2, "Sy", w.s3, "kappa", κ)
        end
        wedge_count += 1
      else
        if isodd(x)
          if abs(w.s3 - w.s2) == Ny
            os .+= κ, "Sz", w.s1, "Sy", w.s2, "Sx", w.s3
            @info "Added three-spin term" term = ("Sz", w.s1, "Sy", w.s2, "Sx", w.s3, "kappa", κ)    
          else
            os .+= κ, "Sz", w.s1, "Sx", w.s2, "Sy", w.s3
            @info "Added three-spin term" term = ("Sz", w.s1, "Sx", w.s2, "Sy", w.s3, "kappa", κ)
          end
          wedge_count += 1
        else
          if abs(w.s2 - w.s1) == Ny 
            os .+= κ, "Sx", w.s1, "Sy", w.s2, "Sz", w.s3 
            @info "Added three-spin term" term = ("Sx", w.s1, "Sy", w.s2, "Sz", w.s3, "kappa", κ)
          else
            os .+= κ, "Sy", w.s1, "Sx", w.s2, "Sz", w.s3
            @info "Added three-spin term" term = ("Sy", w.s1, "Sx", w.s2, "Sz", w.s3, "kappa", κ)
          end
          wedge_count += 1
        end
      end

      @show wedge_count
    end
    

    # Check to make sure the number of three-spin interaction terms is correct
    println("\nChecking the number of three-spin interaction terms in the Hamiltonian:")
    @show wedge_count
    # if wedge_count != 122
    #   error("The number of three-spin interaction terms is incorrect!")
    # end
    # println("")
  end
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
  nsweeps = 12
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
  # @show custom_observer.etolerance
  # @show custom_observer.minsweeps
  @timeit time_machine "dmrg simulation" begin
    energy, ψ = dmrg(H, ψ₀; nsweeps, maxdim, cutoff, eigsolve_krylovdim, observer = custom_observer)
  end

  println("Final ground-state energy = $energy")
  println("")
  #***************************************************************************************************************
  #***************************************************************************************************************

  

  #***************************************************************************************************************
  #***************************************************************************************************************
  """Measure various observables from the ground-state wavefunction"""
  
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


  """Measure the expectation values of the plaquette operators (six-point correlators) on each hexagon"""
  println(repeat("*", 100))
  println("Measuring the expectation values of the plaquette operators on each hexagon")

  
  # Set up the operators in fixed order for each plaquette
  plaquette = Vector{String}(["Z", "iY", "X", "Z", "iY", "X"])


  # Set up a list of indices for each plaquette on the interferometry lattice 
  plaquette_refs = interferometry_plaquette_reference_obc(N, Nx_unit, width_profile, x_gauge) # Step 1: set up all the reference points
  plaquette_indices = interferometry_plaquette_obc(width_profile, x_gauge, plaquette_refs)    # Step 2: set up the indices for each plaquette based on the reference points


  # Compute the expectation values of the plaquette operators on each hexagon
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

  
  
  """Compute the contribution to the ground-state energy from each bond"""
  Etotal = 0.0

  # Loop through all the bonds in the lattice and measure the energy densities associated with each bond 
  E_bond = Any[]
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
    tmp_os = OpSum()
    
    if iseven(x)
      tmp_os .+= -effective_Jz, "Sz", b.s1, "Sz", b.s2
      # @info "Added Sz-Sz bond" term = ("Jz", effective_Jz, "Sz", b.s1, "Sz", b.s2)
    else
      if abs(b.s1 - b.s2) == Ny 
        tmp_os .+= -effective_Jx, "Sx", b.s1, "Sx", b.s2
        # @info "Added Sx-Sx bond" term = ("Jx", effective_Jx, "Sx", b.s1, "Sx", b.s2)
      elseif abs(b.s1 - b.s2) == Ny - 1
        tmp_os .+= -effective_Jy, "Sy", b.s1, "Sy", b.s2
        # @info "Added Sy-Sy bond" term = ("Jy", effective_Jy, "Sy", b.s1, "Sy", b.s2)
      end
    end

    tmp_H = MPO(tmp_os, sites)
    tmp_E = inner(ψ', tmp_H, ψ)
    # @show b.s1, b.s2, tmp_E
    push!(E_bond, [b.s1, b.s2, tmp_E])
    Etotal += tmp_E
  end

  # Convert the vector into a matrix to save it in the HDF5 file
  E_bond = stack(E_bond)
  # @show size(E_bond)
  println("")


  """Compute the energy contribution from each three-spin interaction term"""
  
  E_wedge = Any[]
  for w in wedge
    # Determine the x coordinate of the second site and use the second site as the anchor point 
    x = 0
    for idx in 1 : length(x_gauge) - 1
      if w.s2 > x_gauge[idx] && w.s2 <= x_gauge[idx + 1]
        x = idx
        break
      end
    end
    # @show w.s2, x
    
    tmp_os = OpSum()
    if abs(w.s1 - w.s3) == 1
      if isodd(x)
        tmp_os .+= κ, "Sy", w.s1, "Sz", w.s2, "Sx", w.s3
        # @info "Added three-spin term" term = ("Sy", w.s1, "Sz", w.s2, "Sx", w.s3, "kappa", κ)
      else
        tmp_os .+= κ, "Sx", w.s1, "Sz", w.s2, "Sy", w.s3
        # @info "Added three-spin term" term = ("Sx", w.s1, "Sz", w.s2, "Sy", w.s3, "kappa", κ)
      end
    else
      if isodd(x)
        if abs(w.s3 - w.s2) == Ny
          tmp_os .+= κ, "Sz", w.s1, "Sy", w.s2, "Sx", w.s3
          # @info "Added three-spin term" term = ("Sz", w.s1, "Sy", w.s2, "Sx", w.s3, "kappa", κ)    
        else
          tmp_os .+= κ, "Sz", w.s1, "Sx", w.s2, "Sy", w.s3
          # @info "Added three-spin term" term = ("Sz", w.s1, "Sx", w.s2, "Sy", w.s3, "kappa", κ)
        end
      else
        if abs(w.s2 - w.s1) == Ny 
          tmp_os .+= κ, "Sx", w.s1, "Sy", w.s2, "Sz", w.s3 
          # @info "Added three-spin term" term = ("Sx", w.s1, "Sy", w.s2, "Sz", w.s3, "kappa", κ)
        else
          tmp_os .+= κ, "Sy", w.s1, "Sx", w.s2, "Sz", w.s3
          # @info "Added three-spin term" term = ("Sy", w.s1, "Sx", w.s2, "Sz", w.s3, "kappa", κ)
        end
      end
    end

    
    # Set up the MPO for the three-spin term and compute its energy contribution
    tmp_H = MPO(tmp_os, sites)    
    tmp_E = inner(ψ', tmp_H, ψ)
    # @show w.s1, w.s2, w.s3, tmp_E
    push!(E_wedge, [w.s1, w.s2, w.s3, real(tmp_E)])
    Etotal += tmp_E
  end  

  # Convert the vector into a matrix to save it in the HDF5 file
  E_wedge = stack(E_wedge)
  # @show size(E_wedge)
  println("")


  # Check whether the total energy from bonds and wedges matches the ground-state energy from DMRG
  if abs(Etotal - energy) > 1e-8
    error("The total energy from bonds and wedges does not match the ground-state energy from DMRG!")
  end  
  @show Etotal, energy


  # """
  #   Check the variance of the energy
  # """
  # @timeit time_machine "compaute the variance" begin
  #   H2 = inner(H, ψ, H, ψ)
  #   E₀ = inner(ψ', H, ψ)
  #   variance = H2 - E₀^2
  # end
  # println("Variance of the energy is $variance")
  # println("")
  # ***************************************************************************************************************
  # ***************************************************************************************************************
  

  
  # #***************************************************************************************************************
  # #***************************************************************************************************************
  println(repeat("*", 100))
  println("Summary of results:")
  println("")

   
  println("Expectation values of the plaquette oprators on each hexagon are: ")
  for idx in 1:nplaquettes
    println("Plaquette $idx : ", plaquette_vals[idx])
  end
  println("")
  
  
  println("Bond dimensions of the ground-state wavefunction are: ")
  @show linkdims(ψ) 
  println("")


  # # Check one-point functions
  # println("Expectation values of one-point functions <Sx>, <Sy>, and <Sz>:")
  # @show Sx
  # @show Sy
  # @show Sz
  

  println(header)
  println(header)
  println("")
  #***************************************************************************************************************
  #***************************************************************************************************************

 
 
  """
    Save the ground-state wavefunction and various observables to an HDF5 file
  """
  h5open("data/interferometry_N$(Nx_unit)_kappa$(κ).h5", "cw") do file
    write(file, "psi", ψ)
    write(file, "E0", energy)
    write(file, "E0_bond", E_bond)
    write(file, "E0_wedge", E_wedge)
    # write(file, "E0variance", variance)
    # write(file, "Ehist", custom_observer.ehistory)
    # write(file, "Bond", custom_observer.chi)
    write(file, "chi", linkdims(ψ))
    write(file, "plaquette", plaquette_vals)
  end

  return
end