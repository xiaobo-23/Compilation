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
  # Set up the interaction parameters for the Hamiltonian
  Jx, Jy, Jz = 1.0, 1.0, 1.0
  # alpha = 1E-4
  # h = 0.0

  # honeycomb lattice implemented in the ring ordering scheme
  x_periodic = false
  y_direction_twist = true
  
  # Construct a honeycomb lattice using armchair geometry
  # TO-DO: Implement the armchair geometery with periodic boundary condition
  if x_periodic
    lattice = honeycomb_lattice_rings_pbc(Nx, Ny; yperiodic=true)
    @show length(lattice)
  else
    lattice = honeycomb_lattice_Cstyle(Nx, Ny; yperiodic=true)
    # @show length(lattice)
  end 
  number_of_bonds = length(lattice)
  @show number_of_bonds
  
  # Select the position(s) of the vacancies
  # sites_to_delete = Set{Int64}([59])          
  # lattice_sites   = Set{Int64}()
  
  #***************************************************************************************************************
  #***************************************************************************************************************  
  # Construct the Hamiltonian
  # Construct the two-body interaction temrs in the Kitaev Hamiltonian
  os = OpSum()
  xbond = 0
  ybond = 0
  zbond = 0
  for b in lattice
    tmp_x = 2 * div(b.s1 - 1, 2 * Ny) + (iseven(b.s1) ? 2 : 1)
    tmp_y = div(mod(b.s1, 2*Ny) - 1, 2) + 1
    @show b.s1, b.s2, tmp_x, tmp_y

    if mod(tmp_x, 2) == 0
      os .+= -Jy, "Sy", b.s1, "Sy", b.s2
      # @show b.s1, b.s2, "Sy"
      ybond += 1
    else
      if tmp_y == 1
        if abs(b.s2 - b.s1) == 2 * Ny - 1
          os .+= -Jz, "Sz", b.s1, "Sz", b.s2
          @show b.s1, b.s2, "Sz"
          zbond += 1
        else
          os .+= -Jx, "Sx", b.s1, "Sx", b.s2
          # @show b.s1, b.s2, "Sx"
          xbond += 1
        end
      else 
        if b.s1 < b.s2
          os .+= -Jx, "Sx", b.s1, "Sx", b.s2
          # @show b.s1, b.s2, "Sx"
          xbond += 1
        else
          os .+= -Jz, "Sz", b.s1, "Sz", b.s2
          @show b.s1, b.s2, "Sz"
          zbond += 1
        end
      end
    end
  end
  @show xbond, ybond, zbond
  #***************************************************************************************************************
  #***************************************************************************************************************  


  # # Generate the indices for all loop operators along the cylinder
  # loop_operator = Vector{String}(["iY", "X", "iY", "X", "iY", "X", "iY", "X"])  # Hard-coded for width-4 cylinders
  # loop_indices = LoopListArmchair(Nx_unit, Ny_unit, "armchair", "y")  
  # # @show loop_indices

  # # Generate the plaquette indices for all the plaquettes in the cylinder
  # plaquette_operator = Vector{String}(["iY", "Z", "X", "X", "Z", "iY"])
  # plaquette_indices = PlaquetteListArmchair(Nx_unit, Ny_unit, "armchair", false)
  # # @show plaquette_indices

  
  #******************************************************************************************************
  # Read in an initial state from a MATLAB file
  #******************************************************************************************************
  chi = [2, 4, 8, 16, fill(32, 15)..., 16, 8, 4, 2]
  input_file = matopen("../data/A_chi32.mat")
  input_tensors = read(input_file)
  @show typeof(input_tensors)
  close(input_file)



  tensor_array = Vector{ITensor}(undef, N) 
  prev_index = Index(chi[1], "Link, l=1")
  site_index = Index(2, "S=1/2, n=1")
  # tensor_array[1] = ITensor(input_tensors["A"][1], prev_index, site_index)
  # @show tensor_array[1]

  tensor_array[1] = random_itensor(ComplexF64, site_index, prev_index)
  @show tensor_array[1]
  tensor_array[1][site_index => 1, prev_index => 1] = input_tensors["A"][1][1, 1, 1]
  tensor_array[1][site_index => 1, prev_index => 2] = input_tensors["A"][1][1, 2, 1]
  tensor_array[1][site_index => 2, prev_index => 1] = input_tensors["A"][1][1, 1, 2]
  tensor_array[1][site_index => 2, prev_index => 2] = input_tensors["A"][1][1, 2, 2]

  @show size(input_tensors["A"][1]), input_tensors["A"][1]
  @show tensor_array[1]

  # @show chi
  for idx in 2 : N - 1
    next_index = Index(chi[idx], "Link, l=$(idx)")
    site_index = Index(2, "S=1/2, n=$(idx)")
    @show prev_index, next_index, site_index, chi[idx], size(input_tensors["A"][idx])
    tensor_array[idx] = ITensor(input_tensors["A"][idx], site_index, prev_index, next_index)
    prev_index = next_index
    # @show prev_index
    # @show i, j, k = siteinds(ψ₀)[idx], linkind(ψ₀, idx-1), linkind(ψ₀, idx)
    # @show i, j, k
  end
  
  site_index = Index(2, "S=1/2, n=$(N)")
  # tensor_array[N] = ITensor(input_tensors["A"][N], site_index, prev_index)
  # @show tensor_array[N]

  tensor_array[N] = random_itensor(ComplexF64, site_index, prev_index)
  @show tensor_array[N]
  tensor_array[N][site_index => 1, prev_index => 1] = input_tensors["A"][N][1, 1, 1]
  tensor_array[N][site_index => 1, prev_index => 2] = input_tensors["A"][N][2, 1, 1]
  tensor_array[N][site_index => 2, prev_index => 1] = input_tensors["A"][N][1, 1, 2]
  tensor_array[N][site_index => 2, prev_index => 2] = input_tensors["A"][N][2, 1, 2]

  @show tensor_array[N]
  @show size(input_tensors["A"][N]), input_tensors["A"][N]

  ψ₀ = MPS(tensor_array)  
  #*****************************************************************************************************
  #*****************************************************************************************************
  sites = siteinds(ψ₀)
  H = MPO(os, sites)
  E₀ = inner(ψ₀', H, ψ₀)
  @show E₀

  # Check the variance of the energy
  @timeit time_machine "compaute the variance" begin
    H2 = inner(H, ψ₀, H, ψ₀)
    E₀ = inner(ψ₀', H, ψ₀)
    variance = H2 - E₀^2
  end
  println("")
  @show E₀
  println("Variance of the energy is $variance")
  println("")
  
  #*****************************************************************************************************
  #*****************************************************************************************************  
  # Increase the maximum dimension of Krylov space used to locally solve the eigenvalues problem.
  # sites = siteinds("S=1/2", N; conserve_qns=false)
  
  # Initialize wavefunction to a random MPS of bond-dimension 20 with same quantum number as `state`
  # state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
  # ψ₀ = MPS(sites, state)
  # ψ₀ = random_mps(sites, state; linkdims=2)
  # @show ψ₀


  # # Set up the parameters including bond dimensions and truncation error
  # nsweeps = 1
  # maxdim  = [4, 8, 16, 32]
  # cutoff  = [1E-10]
  # eigsolve_krylovdim = 50
  
  # # Add noise terms to prevent DMRG from getting stuck in a local minimum
  # noise = [1E-6, 1E-7, 0.0]
  # #*****************************************************************************************************
  # #*****************************************************************************************************

  # #*****************************************************************************************************
  # #*****************************************************************************************************
  # # Measure one-point functions of the initial state
  # Sx₀ = expect(ψ₀, "Sx", sites = 1 : N)
  # Sy₀ = expect(ψ₀, "iSy", sites = 1 : N)
  # Sz₀ = expect(ψ₀, "Sz", sites = 1 : N)
  # #*****************************************************************************************************
  # #*****************************************************************************************************
  

  # # Construct a custom observer and stop the DMRG calculation early if needed 
  # # custom_observer = DMRGObserver(; energy_tol=1E-9, minsweeps=2, energy_type=Float64)
  # custom_observer = CustomObserver()
  # @show custom_observer.etolerance
  # @show custom_observer.minsweeps
  # @timeit time_machine "dmrg simulation" begin
  #   energy, ψ = dmrg(H, ψ₀; nsweeps, maxdim, cutoff, eigsolve_krylovdim, observer = custom_observer)
  # end
  
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

  # # Check the variance of the energy
  # @timeit time_machine "compaute the variance" begin
  #   H2 = inner(H, ψ, H, ψ)
  #   E₀ = inner(ψ', H, ψ)
  #   variance = H2 - E₀^2
  # end
  # println("")
  # @show E₀
  # println("Variance of the energy is $variance")
  # println("")
  
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
  # # h5open("../data/Kitaev_FM_Lx$(Nx_unit)_Ly$(Ny_unit)_chi$(maximum(maxdim)).h5", "w") do file
  # #   write(file, "psi", ψ)
  # #   write(file, "NormalizedE0", energy / number_of_bonds)
  # #   write(file, "E0", energy)
  # #   write(file, "E0variance", variance)
  # #   write(file, "Ehist", custom_observer.ehistory)
  # #   write(file, "Bond", custom_observer.chi)
  # #   # write(file, "Entropy", SvN)
  # #   write(file, "Sx0", Sx₀)
  # #   write(file, "Sx",  Sx)
  # #   # write(file, "Cxx", xxcorr)
  # #   write(file, "Sy0", Sy₀)
  # #   write(file, "Sy", Sy)
  # #   # write(file, "Cyy", yycorr)
  # #   write(file, "Sz0", Sz₀)
  # #   write(file, "Sz",  Sz)
  # #   # write(file, "Czz", zzcorr)
  # #   # write(file, "Plaquette", W_operator_eigenvalues)
  # #   # write(file, "Loop", yloop_eigenvalues)
  # # end

  return
end