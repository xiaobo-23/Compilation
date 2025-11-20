## 11/1/2025
## Functions used to update a target two-qubit gate within a set of two-qubit gates
## Use Evenbly-Vidal algorithms to compute the environment tensor and update the target gate


using ITensors
using ITensorMPS
using MKL
using LinearAlgebra
using Random

include("compute_cost_function.jl")


# Define a function to update a single two-qubit gate using Evenbly-Vidal algorithm
function update_single_gate(psi_ket::MPS, psi_bra::MPS, gates_set::Vector{ITensor}, 
  idx::Int64, idx₁::Int64, idx₂::Int64, input_cutoff::Float64 = 1e-14)
    
    # Set up the gate set without the target gate
    gates_copy = deepcopy(gates_set)
    target = gates_copy[idx]

    # idx₁, idx₂ = indices_pairs[idx][1], indices_pairs[idx][2]
    # @show idx₁, idx₂

    # Remove the target gate from the set of gates and check whether it is removed properly
    deleteat!(gates_copy, idx)
    if target in gates_copy
      error("The gate to be optimized is still in the temporary gate set!")
    end
    

    # Apply the gate set without the target gate to the initial MPS
    if length(gates_copy) != 0
      psi_intermediate = apply(gates_copy, psi_ket; cutoff=input_cutoff)
      normalize!(psi_intermediate)
    else
      psi_intermediate = psi_ket
    end

  
    # Set specific site indices to be primed
    prime!(psi_bra[idx₁], tags = "Site")
    prime!(psi_bra[idx₂], tags = "Site")
    i₁, i₂ = siteind(psi_intermediate, idx₁), siteind(psi_intermediate, idx₂)
    j₁, j₂ = siteind(psi_bra, idx₁), siteind(psi_bra, idx₂)
    @show i₁, i₂, j₁, j₂
    # println("")


    # Compute the environment tensor T for the target two-qubit gate from scratch
    T = ITensor(1)
    psi_intermediate_copy = orthogonalize(psi_intermediate, length(psi_intermediate))
    psi_bra_copy = orthogonalize(psi_bra, length(psi_bra))
    
    for j in 1:length(psi_intermediate_copy)
      T *= psi_intermediate_copy[j] 
      T *= dag(psi_bra_copy[j])
    end
    @show inds(T)
    noprime!(psi_bra)
   
    
    # Compute the product of the target gate with its environment tensor & compute the cost function before updating the target gate 
    trace = real((T * target)[1])
    cost = compute_cost_function(psi_ket, psi_bra, gates_set, input_cutoff)
    @show trace, cost 
    println("")

    
    # Perform SVD (USV†) on the environment tensors 
    U, S, V = svd(T, (i₁, i₂))


    # Update the target two-qubit gate using the Evenbly-Vidal formula
    updated_T = dag(V) * delta(inds(S)[1], inds(S)[2]) * dag(U)
    # @show inds(updated_T)
    
    
    # Return the updated two-qubit gate, the trace of the product of the target gate and environment tensor
    # and the cost function
    return updated_T, trace, cost
end