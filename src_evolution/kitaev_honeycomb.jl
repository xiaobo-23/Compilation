# 09/11/2025
# Running DMRG simulation for the Kitaev honeycomb model with three-spin interactions
# Obtaining the ground-state wave function as the starting point for quantum circuit compilation

using ITensors
using ITensorMPS
using HDF5
using MKL
using LinearAlgebra
using TimerOutputs
using MAT
using ITensors.NDTensors


include("honeycomb_lattice.jl")
include("Entanglement.jl")
include("TopologicalLoops.jl")
include("CustomObserver.jl")


# ---- BLAS / LAPACK threading ----
BLAS.set_num_threads(8)
ITensors.Strided.set_num_threads(1)            # avoid oversubscription
ITensors.enable_threaded_blocksparse(false)    # off when conserve_qns=false
@show BLAS.get_config()


# ---- Honeycomb lattice geometry ----
# Each unit cell contains two sites (A/B sublattices) along x.
const Nx_unit = 4
const Ny_unit = 3
const Nx = 2 * Nx_unit   # sites along x (2 per unit cell)
const Ny = Ny_unit       # sites along y (1 per unit cell)
const N  = Nx * Ny

# ---- Kitaev Hamiltonian couplings ---- 
const Jx, Jy, Jz = 1.0, 1.0, 1.0   # two-body bond strength
const κ          = 0.0             # three-spin (TRS-breaking) term strength


# ---- Profiling ----
const time_machine = TimerOutput()



let
  """
    Initialize a honeycomb lattice using zigzag geometry using the Cstyle ordering scheme
    Use PBC along the y direction and OBC along the x direction
  """
  x_periodic = false
  
  # Generate the list of two-body bonds in the honeycomb lattice
  if x_periodic
    lattice = honeycomb_lattice_rings_pbc(Nx, Ny; yperiodic=true)
  else
    lattice = honeycomb_lattice_Cstyle(Nx, Ny; yperiodic=true)
  end 
  number_of_bonds = length(lattice)
 
  # for (_, tmp) in enumerate(lattice)
  #   @show tmp.s1, tmp.s2
  # end

  # Generate the list of three-spin interaction terms in the honeycomb lattice
  wedge = honeycomb_Cstyle_wedge(Nx, Ny; yperiodic=true)

  # for (_, tmp) in enumerate(wedge)
  #   @show tmp.s1, tmp.s2, tmp.s3  
  # end
  

  
  """Construc the Kitaev Hamiltonian as an MPO using the OpSum interface"""
  os = OpSum()
  
  # Set up the two-body interaction terms in the Hamiltonian
  # Count the numbers of ⟨SxSx⟩, ⟨SySy⟩, ⟨SzSz⟩ bonds
  xbond, ybond, zbond = 0, 0, 0        

  for b in lattice
    xcoordinate = 2 * div(b.s1 - 1, 2 * Ny) + (iseven(b.s1) ? 2 : 1)
    ycoordinate = div(mod(b.s1 - 1, 2 * Ny), 2) + 1
    # @show b.s1, xcoordinate, ycoordinate

    if mod(xcoordinate, 2) == 0
      os .+= -Jy, "Y", b.s1, "Y", b.s2
      @show b.s1, b.s2, "Y"
      ybond += 1
    else
      if b.s2 - b.s1 == 1
        os .+= -Jx, "X", b.s1, "X", b.s2
        @show b.s1, b.s2, "X"
        xbond += 1
      else
        os .+= -Jz, "Z", b.s1, "Z", b.s2
        @show b.s1, b.s2, "Z"
        zbond += 1
      end
    end
  end
  # @show xbond, ybond, zbond


  # # Set up the three-spin interaction terms in the Hamiltonian
  # count = 0
  # for w in wedge 
  #   # Calculate the (x, y) coordinates of the site n based on C-style ordering
  #   tmp = div(w.s2 - 1, 2 * Ny)
  #   x = 2 * tmp + mod(w.s2 - 1, 2) + 1
  #   y = mod(div(w.s2 - 1, 2), Ny) + 1

  #   if mod(x, 2) == 1
  #     if w.s1 - w.s2 == 1 && w.s3 - w.s2 == 2 * Ny - 1
  #       os .+= κ, "Sx", w.s1, "Sy", w.s2, "Sz", w.s3
  #       @show w.s1, w.s2, w.s3, "Sx", "Sy", "Sz"
  #       count += 1
  #     end

  #     if w.s3 - w.s2 == 1 && w.s2 - w.s1 == 1
  #       os .+= κ, "Sz", w.s1, "Sy", w.s2, "Sx", w.s3
  #       @show w.s1, w.s2, w.s3, "Sz", "Sy", "Sx"
  #       count += 1
  #     end 

  #     if x != 1 && w.s2 - w.s1 == 2 * Ny - 1
  #       if w.s3 - w.s2 == 1
  #         os .+= κ, "Sy", w.s1, "Sz", w.s2, "Sx", w.s3
  #         @show w.s1, w.s2, w.s3, "Sy", "Sz", "Sx"
  #         count += 1  
  #       else
  #         os .+= κ, "Sy", w.s1, "Sx", w.s2, "Sz", w.s3
  #         @show w.s1, w.s2, w.s3, "Sy", "Sx", "Sz"
  #         count += 1  
  #       end
  #     end
  #   else
  #     if w.s3 - w.s2 == w.s2 - w.s1 == 1
  #       os .+= κ, "Sx", w.s1, "Sy", w.s2, "Sz", w.s3
  #       @show w.s1, w.s2, w.s3, "Sx", "Sy", "Sz"
  #       count += 1
  #     end

  #     if w.s2 - w.s3 == 1 && w.s2 - w.s1 == 2 * Ny - 1
  #       os .+= κ, "Sz", w.s1, "Sy", w.s2, "Sx", w.s3
  #       @show w.s1, w.s2, w.s3, "Sz", "Sy", "Sx"
  #       count += 1
  #     end

  #     if x != Nx && w.s3 - w.s2 == 2 * Ny - 1
  #       if w.s2 - w.s1 == 1
  #         os .+= κ, "Sx", w.s1, "Sz", w.s2, "Sy", w.s3
  #         @show w.s1, w.s2, w.s3, "Sx", "Sz", "Sy"
  #         count += 1
  #       else
  #         os .+= κ, "Sz", w.s1, "Sx", w.s2, "Sy", w.s3
  #         @show w.s1, w.s2, w.s3, "Sz", "Sx", "Sy"
  #         count += 1
  #       end
  #     end
  #   end
  # end
  # @show count 

  # if count != length(wedge)
  #   error("The number of three-spin interaction terms generated does not match the expected number.")
  # end


  
  # """
  #   Set up the initial MPS state and hyper parameters for the DMRG simulation
  #   The initial MPS state is chosen to be a random product state
  # """

  # # Set up the wave function as an MPS with random entries
  # sites = siteinds("S=1/2", N; conserve_qns=false)
  # state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
  # # ψ₀ = MPS(sites, state)
  # ψ₀ = random_mps(sites, state; linkdims=2)

  
  # # Set up the Hamiltonian as MPO
  # H = MPO(os, sites)
  
  
  # # Set up the parameters including bond dimensions and truncation error
  # nsweeps = 10
  # maxdim = [4, 16, 200, 500]
  # # maxdim  = [4, 8, 16, 32]
  # cutoff  = [1E-10]
  # eigsolve_krylovdim = 50
  # which_decomp = "svd"
  
  
  # # # Add noise terms to prevent DMRG from getting stuck in a local minimum
  # # noise = [1E-6, 1E-7, 0.0] 

  
  # # Measure local observables (one-point functions) before starting the DMRG simulation
  # Sx₀ = expect(ψ₀, "Sx", sites = 1 : N)
  # Sy₀ = expect(ψ₀, "iSy", sites = 1 : N)
  # Sz₀ = expect(ψ₀, "Sz", sites = 1 : N)


  # # Construct a custom observer and stop the DMRG calculation early if needed 
  # # custom_observer = DMRGObserver(; energy_tol=1E-9, minsweeps=2, energy_type=Float64)
  # custom_observer = CustomObserver()
  # # @show custom_observer.etolerance
  # # @show custom_observer.minsweeps
  # @timeit time_machine "dmrg simulation" begin
  #   energy, ψ = dmrg(H, ψ₀; nsweeps, maxdim, cutoff, which_decomp, eigsolve_krylovdim, observer = custom_observer)
  # end


  
  # """
  #   Measure local and non-local observables after finishing the DMRG simulation
  # """
  # # Measure local observables (one-point functions) 
  # @timeit time_machine "one-point functions" begin
  #   Sx = expect(ψ, "Sx", sites = 1 : N)
  #   Sy = expect(ψ, "iSy", sites = 1 : N)
  #   Sz = expect(ψ, "Sz", sites = 1 : N)
  # end

 
  # # Measure two-point correlation functions
  # @timeit time_machine "two-point functions" begin
  #   xxcorr = correlation_matrix(ψ, "Sx", "Sx", sites = 1 : N)
  #   yycorr = correlation_matrix(ψ, "Sy", "Sy", sites = 1 : N)
  #   zzcorr = correlation_matrix(ψ, "Sz", "Sz", sites = 1 : N)
  # end



  # # """Compute the expectation values of the noncontractible loop operator along the y direction with PBC"""
  # # # Generate the operators for the noncontractible loop operators along the y direction with PBC  
  # # loop_op = Vector{String}(["iY", "iY", "iY", "iY", "iY", "iY", "iY", "iY"])  # Hard-coded for width-three cylinders
  
  # # # Generate a list of indices for the noncontractible loop operators 
  # # loop_inds = zeros(Int, Nx_unit, 8)  # Each row corresponds to a loop operator along the y direction with PBC
  # # for idx in 1 : Nx_unit
  # #   loop_inds[idx, :] = 8 * (idx - 1) .+ (1 : 8)
  # # end
  # # # @show loop_inds

  # # # Compute the expectation values of the noncontractible loop operators
  # # loop_evals = zeros(Float64, Nx_unit)
  
  # # @timeit time_machine "loop operators" begin
  # #   for idx in 1 : Nx_unit
  # #       ## Construct loop operators along the y direction with PBC
  # #       os_wl = OpSum()
  # #       os_wl += loop_op[1], loop_inds[idx, 1], 
  # #           loop_op[2], loop_inds[idx, 2], 
  # #           loop_op[3], loop_inds[idx, 3], 
  # #           loop_op[4], loop_inds[idx, 4], 
  # #           loop_op[5], loop_inds[idx, 5], 
  # #           loop_op[6], loop_inds[idx, 6], 
  # #           loop_op[7], loop_inds[idx, 7], 
  # #           loop_op[8], loop_inds[idx, 8]

  # #       WL = MPO(os_wl, sites)
  # #       loop_evals[idx] = real(inner(ψ', WL, ψ))
  # #   end
  # # end
  # # @show loop_evals

  
  # # """Compute the expectation values of the plaquette operators defined on the hexagonal plaquettes"""
  # # # Generate the seqeunce of operators for the plaquette operators defined on the hexagonal plaquettes  
  # # plaquette_op = Vector{String}(["iY", "Z", "X", "X", "Z", "iY"])
  
  
  # # # Generate a list of indices for each hexagonal plaquette
  # # plaquette_inds = [1  2  7  6  11 12;
  # #                   3  4  9  2  7  8;
  # #                   5  6  11 4  9  10;
  # #                   7  8  13 12 17 18;
  # #                   9  10 15 8  13 14;
  # #                   11 12 17 10 15 16;
  # #                   13 14 19 18 23 24;
  # #                   15 16 21 14 19 20;
  # #                   17 18 23 16 21 22]
  # # @show plaquette_inds
  
 
  # # # Compute the expectation value of the plaquette operator defined on each hexagonal plaquette
  # # plaquette_evals = zeros(Float64, size(plaquette_inds, 1))
  # # @timeit time_machine "PLAQUETTE OPERATORS" begin
  # #   for idx in 1 : size(plaquette_inds, 1)
  # #     os_wp = OpSum()
  # #     os_wp += plaquette_op[1], plaquette_inds[idx, 1], 
  # #       plaquette_op[2], plaquette_inds[idx, 2], 
  # #       plaquette_op[3], plaquette_inds[idx, 3], 
  # #       plaquette_op[4], plaquette_inds[idx, 4], 
  # #       plaquette_op[5], plaquette_inds[idx, 5], 
  # #       plaquette_op[6], plaquette_inds[idx, 6]
      
  # #       WP = MPO(os_wp, sites)
  # #     plaquette_evals[idx] = -1.0 * real(inner(ψ', WP, ψ))
  # #   end
  # # end
  # # @show plaquette_evals
  
  
  # """
  #   Print out the results of the DMRG simulation and save the results to an HDF5 file for later use in quantum circuit compilation  
  # """


  # # Check the variance of the energy to see if the obtained state is close to an eigenstate of the Hamiltonian
  # @timeit time_machine "compaute the variance" begin
  #   H2 = real(inner(H, ψ, H, ψ))
  #   E₀ = real(inner(ψ', H, ψ))
  #   variance = H2 - E₀^2
  # end

  
  # println("\nGround-state energy: $E₀")
  # println("\nVariance of the energy is $variance")
  # println("\n")

  
  # # println("\nExpectation values of the plaquette operator on each hexagonal plaquette:")
  # # @show plaquette_evals
  # # println("\n")


  # # println("\nExpectation values of the loop operator defined along each loop:")
  # # @show loop_evals
  # # println("\n")


  # println("\nBond dimensions of the obtained MPS:")
  # @show linkdims(ψ)
  # println("\n")


  # # # Print out useful information of physical quantities
  # # println("")
  # # println("Visualize the optimization history of the energy and bond dimensions:")
  # # @show custom_observer.ehistory_full
  # # @show custom_observer.ehistory
  # # @show custom_observer.chi
  # # println("")



  # # Save the results to an HDF5 file for later use in quantum circuit compilation
  # @show time_machine
  # h5open("../data/kitaev_honeycomb_Lx4_Ly3_kappa$(κ).h5", "w") do file
  #   write(file, "psi", ψ)
  #   write(file, "E0", energy)
  #   write(file, "variance", variance)
  #   write(file, "chi", linkdims(ψ))
  #   # write(file, "Sz0", Sz₀)
  #   # write(file, "Sz",  Sz)
  #   # write(file, "Czz", zzcorr)
  #   # write(file, "plaquette", plaquette_evals)
  #   # write(file, "loop", loop_evals)
  # end


  return
end