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


# Set up parameters for multithreading and parallelization
MKL_NUM_THREADS = 8
OPENBLAS_NUM_THREADS = 8
OMP_NUM_THREADS = 8


# Monitor the number of threads used by BLAS and LAPACK
@show BLAS.get_config()
@show BLAS.get_num_threads()


const N  = 12  # Total number of qubits
const J₁ = 1.0
const τ = 1
const cutoff = 1e-12
const nsweeps = 2
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
  # Construct a sequence of two-qubit gates as the target unitaries
  indices_pairs = [[7, 8], [9, 10], [11, 12]]  # Define pairs of qubit indices for two-qubit gates 
  gates = ITensor[]
  for idx in 1 : length(indices_pairs)
    idx₁, idx₂ = indices_pairs[idx][1], indices_pairs[idx][2]
    s₁ = sites[idx₁]
    s₂ = sites[idx₂]

    # Define a two-qubit gate, using the Heisenberg interaction as an example 
    hj = 1/2 * J₁ * op("S+", s₁) * op("S-", s₂) + 1/2 * J₁ * op("S-", s₁) * op("S+", s₂) 
      + J₁ * op("Sz", s₁) * op("Sz", s₂)
    Gj = exp(-im * τ/2 * hj)
    push!(gates, Gj)
  end
  # @show gates

  
  # Construct a sequence of two-qubit gates as the initial random unitaries to be optimized 
  optimization_gates = ITensor[]
  for idx in 1 : length(indices_pairs)
    idx₁, idx₂ = indices_pairs[idx][1], indices_pairs[idx][2]
    s₁ = sites[idx₁]
    s₂ = sites[idx₂]

    # SVD a random tensor to obtain a random unitary by setting all the singular values to 1 
    G_opt = randomITensor(s₁', s₂', s₁, s₂)
    U, S, V = svd(G_opt, (s₁', s₂'))
    G_random = U * delta(inds(S)[1], inds(S)[2]) * dag(V)
    push!(optimization_gates, G_random)
  end
  # @show length(optimization_gates)


  
  # Apply the sequence of two-qubit gates to the original MPS
  ψ_R = deepcopy(ψ₀)                        
  ψ_R = apply(gates, ψ_R; cutoff=cutoff)
  normalize!(ψ_R)
  
  
  # Sx_R, Sz_R = zeros(Float64, N), zeros(Float64, N)
  Sx_R = expect(ψ_R, "Sx", sites = 1 : N)
  Sz_R = expect(ψ_R, "Sz", sites = 1 : N)
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
        ψ₀, ψ_R, optimization_gates, idx, idx₁, idx₂, cutoff
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
        ψ₀, ψ_R, optimization_gates, idx, idx₁, idx₂, cutoff
      )
      optimization_gates[idx] = updated_gate
      append!(optimization_trace, tmp_trace)
      append!(fidelity_trace, tmp_cost)
    end
    println(repeat("#", 200))
    println("")


    # Compute the cost function after each sweep
    cost_function[iteration] = compute_cost_function(ψ₀, ψ_R, optimization_gates, cutoff)
    reference[iteration] = compute_cost_function(ψ₀, ψ_R, gates, cutoff)
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