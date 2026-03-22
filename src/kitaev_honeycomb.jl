## 09/11/2025
## Ground state preparation for the Kitaev honeycomb model on cylinders 
## Ongoing collaboratiion with Quantinuum

using ITensors
using ITensorMPS
using HDF5
using MKL
using LinearAlgebra
using TimerOutputs
using MAT
using ITensors.NDTensors


include("HoneycombLattice.jl")
include("Entanglement.jl")
include("TopologicalLoops.jl")
include("CustomObserver.jl")


# Set up parameters for multithreading for BLAS/LAPACK and Block sparse multithreading
MKL_NUM_THREADS = 8
OPENBLAS_NUM_THREADS = 8
OMP_NUM_THREADS = 8


# Monitor the number of threads used by BLAS and LAPACK
@show BLAS.get_config()
@show BLAS.get_num_threads()


const Nx_unit = 4
const Ny_unit = 3
const Nx = 2 * Nx_unit
const Ny = Ny_unit
const N = Nx * Ny


# Timing and profiling
const time_machine = TimerOutput()


let
  # Set up interactions strengths in the Kitaev Hamiltonian
  Jx, Jy, Jz = 1.0, 1.0, 1.0  # Strengths of the two-body interaction terms
  κ = -0.4                    # Strength of the three-spin interaction term

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
      os .+= -Jy, "Sy", b.s1, "Sy", b.s2
      @show b.s1, b.s2, "Sy"
      ybond += 1
    else
      if b.s2 - b.s1 == 1
        os .+= -Jx, "Sx", b.s1, "Sx", b.s2
        @show b.s1, b.s2, "Sx"
        xbond += 1
      else
        os .+= -Jz, "Sz", b.s1, "Sz", b.s2
        @show b.s1, b.s2, "Sz"
        zbond += 1
      end
    end
  end
  # @show xbond, ybond, zbond


  # Set up the three-spin interaction terms in the Hamiltonian
  count = 0
  for w in wedge 
    # Calculate the (x, y) coordinates of the site n based on C-style ordering
    tmp = div(w.s2 - 1, 2 * Ny)
    x = 2 * tmp + mod(w.s2 - 1, 2) + 1
    y = mod(div(w.s2 - 1, 2), Ny) + 1

    if mod(x, 2) == 1
      if w.s1 - w.s2 == 1 && w.s3 - w.s2 == 2 * Ny - 1
        os .+= κ, "Sx", w.s1, "Sy", w.s2, "Sz", w.s3
        @show w.s1, w.s2, w.s3, "Sx", "Sy", "Sz"
        count += 1
      end

      if w.s3 - w.s2 == 1 && w.s2 - w.s1 == 1
        os .+= κ, "Sz", w.s1, "Sy", w.s2, "Sx", w.s3
        @show w.s1, w.s2, w.s3, "Sz", "Sy", "Sx"
        count += 1
      end 

      if x != 1 && w.s2 - w.s1 == 2 * Ny - 1
        if w.s3 - w.s2 == 1
          os .+= κ, "Sy", w.s1, "Sz", w.s2, "Sx", w.s3
          @show w.s1, w.s2, w.s3, "Sy", "Sz", "Sx"
          count += 1  
        else
          os .+= κ, "Sy", w.s1, "Sx", w.s2, "Sz", w.s3
          @show w.s1, w.s2, w.s3, "Sy", "Sx", "Sz"
          count += 1  
        end
      end
    else
      if w.s3 - w.s2 == w.s2 - w.s1 == 1
        os .+= κ, "Sx", w.s1, "Sy", w.s2, "Sz", w.s3
        @show w.s1, w.s2, w.s3, "Sx", "Sy", "Sz"
        count += 1
      end

      if w.s2 - w.s3 == 1 && w.s2 - w.s1 == 2 * Ny - 1
        os .+= κ, "Sz", w.s1, "Sy", w.s2, "Sx", w.s3
        @show w.s1, w.s2, w.s3, "Sz", "Sy", "Sx"
        count += 1
      end

      if x != Nx && w.s3 - w.s2 == 2 * Ny - 1
        if w.s2 - w.s1 == 1
          os .+= κ, "Sx", w.s1, "Sz", w.s2, "Sy", w.s3
          @show w.s1, w.s2, w.s3, "Sx", "Sz", "Sy"
          count += 1
        else
          os .+= κ, "Sz", w.s1, "Sx", w.s2, "Sy", w.s3
          @show w.s1, w.s2, w.s3, "Sz", "Sx", "Sy"
          count += 1
        end
      end
    end
  end
  @show count 

  if count != length(wedge)
    error("The number of three-spin interaction terms generated does not match the expected number.")
  end


  
  # #*************************************************************************************************************************************
  # # Read in an initial state from a MATLAB file
  # #*************************************************************************************************************************************
  # chi = [2, 4, 8, 16, fill(32, 15)..., 16, 8, 4, 2]
  # input_file = matopen("../data/A_chi32_AFM.mat")
  # input_tensors = read(input_file)
  # @show typeof(input_tensors)
  # close(input_file)


  # tensor_array = Vector{ITensor}(undef, N) 
  # prev_index = Index(chi[1], "Link, l=1")
  # site_index = Index(2, "S=1/2, n=1")
  # # tensor_array[1] = ITensor(input_tensors["A"][1], prev_index, site_index)
  # # @show tensor_array[1]

  # tensor_array[1] = random_itensor(ComplexF64, site_index, prev_index)
  # @show tensor_array[1]
  # tensor_array[1][site_index => 1, prev_index => 1] = input_tensors["A"][1][1, 1, 1]
  # tensor_array[1][site_index => 1, prev_index => 2] = input_tensors["A"][1][1, 2, 1]
  # tensor_array[1][site_index => 2, prev_index => 1] = input_tensors["A"][1][1, 1, 2]
  # tensor_array[1][site_index => 2, prev_index => 2] = input_tensors["A"][1][1, 2, 2]

  # @show size(input_tensors["A"][1]), input_tensors["A"][1]
  # @show tensor_array[1]

  # # @show chi
  # for idx in 2 : N - 1
  #   next_index = Index(chi[idx], "Link, l=$(idx)")
  #   site_index = Index(2, "S=1/2, n=$(idx)")
  #   @show prev_index, next_index, site_index, chi[idx], size(input_tensors["A"][idx])
  #   tensor_array[idx] = ITensor(input_tensors["A"][idx], prev_index, next_index, site_index)
  #   prev_index = next_index
  #   # @show prev_index
  #   # @show i, j, k = siteinds(ψ₀)[idx], linkind(ψ₀, idx-1), linkind(ψ₀, idx)
  #   # @show i, j, k
  # end
  
  # site_index = Index(2, "S=1/2, n=$(N)")
  # # tensor_array[N] = ITensor(input_tensors["A"][N], site_index, prev_index)
  # # @show tensor_array[N]

  # tensor_array[N] = random_itensor(ComplexF64, site_index, prev_index)
  # @show tensor_array[N]
  # tensor_array[N][site_index => 1, prev_index => 1] = input_tensors["A"][N][1, 1, 1]
  # tensor_array[N][site_index => 1, prev_index => 2] = input_tensors["A"][N][2, 1, 1]
  # tensor_array[N][site_index => 2, prev_index => 1] = input_tensors["A"][N][1, 1, 2]
  # tensor_array[N][site_index => 2, prev_index => 2] = input_tensors["A"][N][2, 1, 2]

  # @show tensor_array[N]
  # @show size(input_tensors["A"][N]), input_tensors["A"][N]

  # ψ₀ = MPS(tensor_array)  
  # sites = siteinds(ψ₀)
  # H = MPO(os, sites)
  # E₀ = inner(ψ₀', H, ψ₀)
  # @show E₀

  # # Check the variance of the energy
  # @timeit time_machine "compaute the variance" begin
  #   H2 = inner(H, ψ₀, H, ψ₀)
  #   E₀ = inner(ψ₀', H, ψ₀)
  #   variance = H2 - E₀^2
  # end
  # println("")
  # println("")
  # println("Energy of the read-in state:")
  # @show E₀
  # println("Variance of the energy is $variance")
  # println("")
  # #*************************************************************************************************************************************
  # #*************************************************************************************************************************************

  
  
  #*****************************************************************************************************************************************************
  # Increase the maximum dimension of Krylov space used to locally solve the eigenvalues problem
  # Initialize wavefunction to a random MPS with same quantum number as `state`
  sites = siteinds("S=1/2", N; conserve_qns=false)
  state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
  # ψ₀ = MPS(sites, state)
  ψ₀ = random_mps(sites, state; linkdims=2)
  # @show ψ₀

  
  # Set up the Hamiltonian as MPO
  H = MPO(os, sites)
  
  
  # Set up the parameters including bond dimensions and truncation error
  nsweeps = 1
  maxdim = [4, 8, 128, 500]
  # maxdim  = [4, 8, 16, 32]
  cutoff  = [1E-10]
  eigsolve_krylovdim = 100
  which_decomp = "svd"
  
  
  # # Add noise terms to prevent DMRG from getting stuck in a local minimum
  # noise = [1E-6, 1E-7, 0.0] 

  
  # Measure local observables (one-point functions) before starting the DMRG simulation
  Sx₀ = expect(ψ₀, "Sx", sites = 1 : N)
  Sy₀ = expect(ψ₀, "iSy", sites = 1 : N)
  Sz₀ = expect(ψ₀, "Sz", sites = 1 : N)


  # Construct a custom observer and stop the DMRG calculation early if needed 
  # custom_observer = DMRGObserver(; energy_tol=1E-9, minsweeps=2, energy_type=Float64)
  custom_observer = CustomObserver()
  # @show custom_observer.etolerance
  # @show custom_observer.minsweeps
  @timeit time_machine "dmrg simulation" begin
    energy, ψ = dmrg(H, ψ₀; nsweeps, maxdim, cutoff, which_decomp, eigsolve_krylovdim, observer = custom_observer)
  end
  #*****************************************************************************************************************************************************


  
  # # Measure local observables (one-point functions) after finish the DMRG simulation
  # @timeit time_machine "one-point functions" begin
  #   Sx = expect(ψ, "Sx", sites = 1 : N)
  #   Sy = expect(ψ, "iSy", sites = 1 : N)
  #   Sz = expect(ψ, "Sz", sites = 1 : N)
  # end

  # # @timeit time_machine to "two-point functions" begin
  # #   xxcorr = correlation_matrix(ψ, "Sx", "Sx", sites = 1 : N)
  # #   yycorr = correlation_matrix(ψ, "Sy", "Sy", sites = 1 : N)
  # #   zzcorr = correlation_matrix(ψ, "Sz", "Sz", sites = 1 : N)
  # # end


  # # Generate the indices for all loop operators along the cylinder
  # loop_operator = Vector{String}(["iY", "X", "iY", "X", "iY", "X", "iY", "X"])  # Hard-coded for width-4 cylinders
  # loop_indices = LoopListArmchair(Nx_unit, Ny_unit, "armchair", "y")  
  # # @show loop_indices

  # # Generate the plaquette indices for all the plaquettes in the cylinder
  # plaquette_operator = Vector{String}(["iY", "Z", "X", "X", "Z", "iY"])
  # plaquette_indices = PlaquetteListArmchair(Nx_unit, Ny_unit, "armchair", false)
  # # @show plaquette_indices


  # # # Compute the eigenvalues of plaquette operators
  # # # normalize!(ψ)
  # # @timeit time_machine "plaquette operators" begin
  # #   W_operator_eigenvalues = zeros(Float64, size(plaquette_indices, 1))
    
  # #   # Compute the eigenvalues of the plaquette operator
  # #   for index in 1 : size(plaquette_indices, 1)
  # #     @show plaquette_indices[index, :]
  # #     os_w = OpSum()
  # #     os_w += plaquette_operator[1], plaquette_indices[index, 1], 
  # #       plaquette_operator[2], plaquette_indices[index, 2], 
  # #       plaquette_operator[3], plaquette_indices[index, 3], 
  # #       plaquette_operator[4], plaquette_indices[index, 4], 
  # #       plaquette_operator[5], plaquette_indices[index, 5], 
  # #       plaquette_operator[6], plaquette_indices[index, 6]
  # #     W = MPO(os_w, sites)
  # #     W_operator_eigenvalues[index] = -1.0 * real(inner(ψ', W, ψ))
  # #     # @show inner(ψ', W, ψ) / inner(ψ', ψ)
  # #   end
  # # end
  # # @show W_operator_eigenvalues
  
  # # # Compute the eigenvalues of the loop operators 
  # # # The loop operators depend on the width of the cylinder  
  # # @timeit time_machine "loop operators" begin
  # #   yloop_eigenvalues = zeros(Float64, size(loop_indices)[1])
    
  # #   # Compute eigenvalues of the loop operators in the direction with PBC.
  # #   for index in 1 : size(loop_indices)[1]
  # #     ## Construct loop operators along the y direction with PBC
  # #     os_wl = OpSum()
  # #     os_wl += loop_operator[1], loop_indices[index, 1], 
  # #       loop_operator[2], loop_indices[index, 2], 
  # #       loop_operator[3], loop_indices[index, 3], 
  # #       loop_operator[4], loop_indices[index, 4], 
  # #       loop_operator[5], loop_indices[index, 5], 
  # #       loop_operator[6], loop_indices[index, 6]

  # #     Wl = MPO(os_wl, sites)
  # #     yloop_eigenvalues[index] = real(inner(ψ', Wl, ψ))
  # #   end
  # # end

  
  # # Print out useful information of physical quantities
  # println("")
  # println("Visualize the optimization history of the energy and bond dimensions:")
  # @show custom_observer.ehistory_full
  # @show custom_observer.ehistory
  # @show custom_observer.chi
  # # @show number_of_bonds, energy / number_of_bonds
  # # @show N, energy / N
  # println("")

  

  # Check the variance of the energy
  @timeit time_machine "compaute the variance" begin
    H2 = inner(H, ψ, H, ψ)
    E₀ = inner(ψ', H, ψ)
    variance = H2 - E₀^2
  end
  println("\nGround-state energy: $E₀")
  println("\nVariance of the energy is $variance")
  println("\n")

  
  # # println("")
  # # println("Eigenvalues of the plaquette operator:")
  # # @show W_operator_eigenvalues
  # # println("")

  # # print("")
  # # println("Eigenvalues of the loop operator(s):")
  # # @show yloop_eigenvalues
  # # println("")

  # # println("")
  # # println("Eigenvalues of the twelve-point correlator near the first vacancy:")
  # # @show order_parameter
  # # println("")

  
  # @show time_machine
  # h5open("../data/kitaev_honeycomb_Lx4_Ly3.h5", "w") do file
  #   write(file, "psi", ψ)
  #   write(file, "E0", energy)
  #   write(file, "E0variance", variance)
  #   write(file, "Ehist", custom_observer.ehistory)
  #   write(file, "Bond", custom_observer.chi)
  #   # write(file, "Entropy", SvN)
  #   # write(file, "Sx0", Sx₀)
  #   # write(file, "Sx",  Sx)
  #   # write(file, "Cxx", xxcorr)
  #   # write(file, "Sy0", Sy₀)
  #   # write(file, "Sy", Sy)
  #   # write(file, "Cyy", yycorr)
  #   # write(file, "Sz0", Sz₀)
  #   # write(file, "Sz",  Sz)
  #   # write(file, "Czz", zzcorr)
  #   # write(file, "Plaquette", W_operator_eigenvalues)
  #   # write(file, "Loop", yloop_eigenvalues)
  # end

  return
end