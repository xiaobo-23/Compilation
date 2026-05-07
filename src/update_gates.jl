# 11/1/2025
# Functions used to update a target two-qubit gate within a set of two-qubit gates
# Use Evenbly-Vidal algorithms to compute the environment tensor and update the target gate

using ITensors
using ITensorMPS
using MKL
using LinearAlgebra
using Random
using Printf

include("compute_cost_function.jl")


const PauliX = ComplexF64[0  1;  1  0]
const PauliY = ComplexF64[0 -im; im  0]
const PauliZ = ComplexF64[1  0;  0 -1]

const PAULI_PRODUCTS = Dict{String, Matrix{ComplexF64}}(
    "Rxx" => kron(PauliX, PauliX),
    "Ryy" => kron(PauliY, PauliY),
    "Rzz" => kron(PauliZ, PauliZ),
)


# Define a function to update a single two-qubit gate using Evenbly-Vidal algorithm
function update_single_gate(psi_ket::MPS, psi_bra::MPS, gates_set::Vector{ITensor}, 
  idx::Int64, idx₁::Int64, idx₂::Int64, input_cutoff::Float64 = 1e-10)
    
    # Remove the target gate from the set of gates and check whether it is removed properly
    target = gates_set[idx]
    gates_copy = ITensor[gates_set[i] for i in eachindex(gates_set) if i != idx]
    if target in gates_copy
    	error("The gate to be optimized is still in the temporary gate set!")
    end

    # idx₁, idx₂ = indices_pairs[idx][1], indices_pairs[idx][2]
    # @show idx₁, idx₂


    # Apply the gate set without the target gate to the initial MPS
    psi_intermediate = isempty(gates_copy) ? 
        deepcopy(psi_ket) :
        normalize!(apply(gates_copy, psi_ket; cutoff=input_cutoff))
  
    
    
        # Set specific site indices to be primed
    prime!(psi_bra[idx₁], tags = "Site")
    prime!(psi_bra[idx₂], tags = "Site")
    i₁, i₂ = siteind(psi_intermediate, idx₁), siteind(psi_intermediate, idx₂)
    j₁, j₂ = siteind(psi_bra, idx₁), siteind(psi_bra, idx₂)
    # @show i₁, i₂, j₁, j₂
    # println("")


    # Compute the environment tensor T for the target two-qubit gate from scratch
    psi_intermediate_copy = orthogonalize(psi_intermediate, length(psi_intermediate))
    psi_bra_copy = orthogonalize(psi_bra, length(psi_bra))
    

    T = ITensor(1)
    for j in eachindex(psi_intermediate_copy)
      T *= psi_intermediate_copy[j] 
      T *= dag(psi_bra_copy[j])
    end
    noprime!(psi_bra)
   
    
    # Compute the product of the target gate with its environment tensor & compute the cost function before updating the target gate 
    trace = real((T * target)[1])
    cost = compute_cost_function(psi_ket, psi_bra, gates_set, input_cutoff)
    # @show trace, cost 
    # println("")

    
    # Perform SVD (USV†) on the environment tensors 
    U, S, V = svd(T, (i₁, i₂))


    # Update the target two-qubit gate using the Evenbly-Vidal formula
    updated_T = dag(V) * delta(inds(S)[1], inds(S)[2]) * dag(U)
    # @show inds(updated_T)
    
    
    # Return the updated two-qubit gate, the trace of the product of the target gate and environment tensor
    # and the cost function
    return updated_T, trace, cost
end



# Define a function to update one single-qubit gate using Evenbly-Vidal algorithm
function update_single_qubit_gate(psi_ket::MPS, psi_bra::MPS, gates_set::Vector{ITensor}, 
  idx::Int64, idx₁::Int64, input_cutoff::Float64 = 1e-10)
    
    # Remove the target gate from the set of gates and check whether it is removed properly
    target = gates_set[idx]
    gates_copy = ITensor[gates_set[i] for i in eachindex(gates_set) if i != idx]
    if target in gates_copy
    	error("The gate to be optimized is still in the temporary gate set!")
    end

    # Apply the gate set without the target gate to the initial MPS
    psi_intermediate = isempty(gates_copy) ? 
        deepcopy(psi_ket) :
        normalize!(apply(gates_copy, psi_ket; cutoff=input_cutoff))


	# Set specific site indices to be primed
	prime!(psi_bra[idx₁], tags = "Site")
	i₁ = siteind(psi_intermediate, idx₁)
	j₁ = siteind(psi_bra, idx₁)
	# @show i₁, j₁


	# Compute the environment tensor T for the target single-qubit gate from scratch
	T = ITensor(1)
	psi_intermediate_copy = orthogonalize(psi_intermediate, length(psi_intermediate))
	psi_bra_copy = orthogonalize(psi_bra, length(psi_bra))

	for j in eachindex(psi_intermediate_copy)
		T *= psi_intermediate_copy[j] 
		T *= dag(psi_bra_copy[j])
	end
	# @show inds(T)
	noprime!(psi_bra)


	# Compute the product of the target gate with its environment tensor & compute the cost function before updating the target gate 
	trace = real((T * target)[1])
	cost = compute_cost_function(psi_ket, psi_bra, gates_set, input_cutoff)
	# @show trace, cost 
	# println("")


	# Perform SVD (USV†) on the environment tensors 
	U, S, V = svd(T, (i₁))


	# Update the target two-qubit gate using the Evenbly-Vidal formula
	updated_T = dag(V) * delta(inds(S)[1], inds(S)[2]) * dag(U)
	# @show inds(updated_T)


	# Return the updated two-qubit gate, the trace of the product of the target gate and environment tensor
	# and the cost function
	return updated_T, trace, cost
end



# Define a function to update a single two-qubit gate using Evenbly-Vidal algorithm
function update_Pauli(psi_ket::MPS, psi_bra::MPS, gates_set::Vector{ITensor}, 
    idx::Int64, idx₁::Int64, idx₂::Int64, input_sites, gate_name::String, input_cutoff::Float64 = 1e-10)
    
    
    # Set up the gate set without the target gate	
    P = PAULI_PRODUCTS[gate_name]	
    target = gates_set[idx]


    # Remove the target gate from the set of gates and check whether it is removed properly
    gates_copy = ITensor[gates_set[i] for i in eachindex(gates_set) if i != idx]
    if target in gates_copy
    	error("The gate to be optimized is still in the temporary gate set!")
    end
    

    # Apply the gate set without the target gate to the initial MPS
    psi_intermediate = isempty(gates_copy) ? 
        deepcopy(psi_ket) :
        normalize!(apply(gates_copy, psi_ket; cutoff=input_cutoff))


  
    # Set specific site indices to be primed
    prime!(psi_bra[idx₁], tags = "Site")
    prime!(psi_bra[idx₂], tags = "Site")
    i₁, i₂ = siteind(psi_intermediate, idx₁), siteind(psi_intermediate, idx₂)
    j₁, j₂ = siteind(psi_bra, idx₁), siteind(psi_bra, idx₂)
    # @show i₁, i₂, j₁, j₂
    # println()


    # Compute the environment tensor T for the target gate from scratch
    psi_intermediate_copy = orthogonalize(psi_intermediate, length(psi_intermediate))
    psi_bra_copy = orthogonalize(psi_bra, length(psi_bra))
    
    E_T = ITensor(1)
    for j in eachindex(psi_intermediate_copy)
        E_T *= psi_intermediate_copy[j] 
        E_T *= dag(psi_bra_copy[j])
    end
    noprime!(psi_bra)
   

	trace = real((E_T * target)[1])
    cost = compute_cost_function(psi_ket, psi_bra, gates_set, input_cutoff)
    # @show trace, cost 


	# Compute the product of the target gate with its environment tensor & compute the cost function before updating the target gate 
	C_row = combiner(i₂, i₁)
	C_col = combiner(j₂, j₁)
	matrix_T = matrix(C_row * E_T * C_col, combinedind(C_row), combinedind(C_col))
	
	
	# Update the input angle based on the coefficients 
	# One of them should give the maximum value and the other gives the minimum value
	coeff_A = imag(tr(matrix_T * P))
	coeff_B = real(tr(matrix_T))
	θ₁ = atan(coeff_A, coeff_B)
	θ₂ = θ₁ + π
	

	# Update the target gate using native gate constructor in ITensorMPS.jl
	updated_T1 = op(input_sites, gate_name, idx₁, idx₂; ϕ=θ₁)
	updated_T2 = op(input_sites, gate_name, idx₁, idx₂; ϕ=θ₂)


	if real((E_T * updated_T1)[1]) > real((E_T * updated_T2)[1])
		updated_T = updated_T1
	else
		updated_T = updated_T2
	end
	# @show real((E_T * updated_T)[1]), real((E_T * updated_T1)[1]), real((E_T * updated_T2)[1])
	# println("")


    # Return the updated gate, the trace of the product of the target gate and environment tensor and the cost function
    return updated_T, trace, cost
end



# Define a wrapper function to update a single Rzz gate
function update_Rzz(
    psi_ket::MPS,
    psi_bra::MPS,
    gates_set::Vector{ITensor},
    idx::Int64,
    idx₁::Int64,
    idx₂::Int64,
    input_sites,
    input_cutoff::Float64 = 1e-10,
)
    return update_Pauli(
        psi_ket,
        psi_bra,
        gates_set,
        idx,
        idx₁,
        idx₂,
        input_sites,
        "Rzz",
        input_cutoff,
    )
end