## 10/7/2025
## Compiling initial state of the Kitaev model on a cylinder geometry
## Optimize parameters of two-qubit gates

using ITensors
using ITensorMPS
using HDF5
using MKL
using LinearAlgebra
using TimerOutputs
using MAT
using Random


include("compute_cost_function.jl")
include("update_gates.jl")
include("compilation_initialization.jl")

# Set up parameters for multithreading and parallelization
MKL_NUM_THREADS = 8
OPENBLAS_NUM_THREADS = 8
OMP_NUM_THREADS = 8


# Monitor the number of threads used by BLAS and LAPACK
@show BLAS.get_config()
@show BLAS.get_num_threads()


const N  = 12  # Total number of qubits
const J₁ = 1.0
const τ = 1.0
const cutoff = 1e-12
const nsweeps = 3
# const time_machine = TimerOutput()  # Timing and profiling


let
  #*****************************************************************************************************
  #*****************************************************************************************************
  println(repeat("#", 200))
  println("Optimize two-qubit gates to approximate the time evolution operator")
  
  
  # Initialize the original random MPS
  Random.seed!(123)
  sites = siteinds("S=1/2", N; conserve_qns=false)
  state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
  ψ₀ = random_mps(sites, state; linkdims=8)   # Initialize the original random MPS
  # ψ₀ = MPS(sites, state)                    # Initialize a Néel state MPS
  # @show ψ₀

  
  # Measure local observables (one-point functions)
  Sx₀, Sy₀, Sz₀ = zeros(Float64, N), zeros(ComplexF64, N), zeros(Float64, N)
  Sx₀ = expect(ψ₀, "Sx", sites = 1 : N)
  Sy₀ = -im*expect(ψ₀, "iSy", sites = 1 : N)
  Sz₀ = expect(ψ₀, "Sz", sites = 1 : N)
  #*****************************************************************************************************
  #*****************************************************************************************************
  
  
  
  #*****************************************************************************************************
  #*****************************************************************************************************
  # Construct the sequence of target gates and randomly intiialized gates to be optimized
  # Define pairs of qubit indices for two-qubit gates 
  indices_pairs = [[7, 8], [9, 10], [11, 12]]       
  
  # Target two-qubit gate sequence
  gates = heisenberg_gates_single_layer(indices_pairs, J₁, τ, sites)    
  # @show gates

  # Initial random two-qubit gate sequence
  optimization_gates = random_gates_single_layer(indices_pairs, sites)   
  # # @show length(optimization_gates)
  #*****************************************************************************************************
  #*****************************************************************************************************
  
  


  #*****************************************************************************************************
  #*****************************************************************************************************
  # Create the target MPS by applying the sequence of two-qubit gates to the original MPS
  ψ_T = deepcopy(ψ₀)                        
  ψ_T = apply(gates, ψ_T; cutoff=cutoff)
  normalize!(ψ_T)
  
  # Sx_R, Sz_R = zeros(Float64, N), zeros(Float64, N)
  Sx_R = expect(ψ_T, "Sx", sites = 1 : N)
  Sz_R = expect(ψ_T, "Sz", sites = 1 : N)
  println("")
  println("After applying the sequence of two-qubit gates:")
  @show Sx₀
  @show Sx_R
  println("")
  #*****************************************************************************************************
  #*****************************************************************************************************


  #*****************************************************************************************************
  #*****************************************************************************************************
  # Optimize the set of two-qubit gates using an iterative sweeping procedure
  cost_function, reference = Vector{Float64}(undef, nsweeps), Vector{Float64}(undef, nsweeps)
  optimization_trace, fidelity_trace = Float64[], Float64[]
  
  for iteration in 1 : nsweeps
    
    # Update each two-qubit gate in the forward order
    println(repeat("#", 200))
    println("Iteration = $iteration: Forward Sweep")
    for idx in 1 : length(indices_pairs)
      idx₁, idx₂ = indices_pairs[idx][1], indices_pairs[idx][2]
      @show idx₁, idx₂
      updated_gate, tmp_trace, tmp_cost = update_single_gate(
        ψ₀, ψ_T, optimization_gates, idx, idx₁, idx₂, cutoff
      )
      optimization_gates[idx] = updated_gate
      append!(optimization_trace, tmp_trace)
      append!(fidelity_trace, tmp_cost)
    end
    println(repeat("#", 200))
    println("")
    println("")
    

    # Update each two-qubit gate in the backward order
    println(repeat("#", 200))
    println("Iteration = $iteration: Backward Sweep")
    for idx in length(indices_pairs):-1:1
      idx₁, idx₂ = indices_pairs[idx][1], indices_pairs[idx][2]
      @show idx₁, idx₂
      idx₁, idx₂ = indices_pairs[idx][1], indices_pairs[idx][2]
      @show idx₁, idx₂
      updated_gate, tmp_trace, tmp_cost = update_single_gate(
        ψ₀, ψ_T, optimization_gates, idx, idx₁, idx₂, cutoff
      )
      optimization_gates[idx] = updated_gate
      append!(optimization_trace, tmp_trace)
      append!(fidelity_trace, tmp_cost)
    end
    println(repeat("#", 200))
    println("")


    # Compute the cost function after each sweep
    cost_function[iteration] = compute_cost_function(ψ₀, ψ_T, optimization_gates, cutoff)
    reference[iteration] = compute_cost_function(ψ₀, ψ_T, gates, cutoff)
  end

  
  println("\nResults after optimization:")
  @show optimization_trace
  @show fidelity_trace
  @show cost_function
  # @show reference 
  
  # output_filename = "../data/compilation_generic_N$(N)_v3.h5"
  # h5open(output_filename, "w") do file
  #   write(file, "cost function", cost_function)
  #   write(file, "reference", reference)
  #   write(file, "optimization trace", optimization_trace)
  #   write(file, "fidelity trace", fidelity_trace)
  # end
  
  return
end