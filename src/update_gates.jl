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


PauliZ = [1  0; 0  -1]
KroneckerZ = kron(PauliZ, PauliZ)
# display(KroneckerZ)


# Define a function to update a single two-qubit gate using Evenbly-Vidal algorithm
function update_single_gate(psi_ket::MPS, psi_bra::MPS, gates_set::Vector{ITensor}, 
  idx::Int64, idx₁::Int64, idx₂::Int64, input_cutoff::Float64 = 1e-10)
    
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
    # @show inds(T)
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



# Define a function to update one single-qubit gate using Evenbly-Vidal algorithm
function update_single_qubit_gate(psi_ket::MPS, psi_bra::MPS, gates_set::Vector{ITensor}, 
  idx::Int64, idx₁::Int64, input_cutoff::Float64 = 1e-10)
    
	# Set up the gate set without the target gate
	gates_copy = deepcopy(gates_set)
	target = gates_copy[idx]

	# idx₁ = indices_pairs[idx][1]
	# @show idx₁

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
	i₁ = siteind(psi_intermediate, idx₁)
	j₁ = siteind(psi_bra, idx₁)
	@show i₁, j₁


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
	@show trace, cost 
	println("")


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
function update_Rzz(psi_ket::MPS, psi_bra::MPS, gates_set::Vector{ITensor}, 
  idx::Int64, idx₁::Int64, idx₂::Int64, input_sites, input_cutoff::Float64 = 1e-10)
    
    # Set up the gate set without the target gate
    gates_copy = deepcopy(gates_set)
    target = gates_copy[idx]


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
      psi_intermediate = deepcopy(psi_ket)
    end

  
    # Set specific site indices to be primed
    prime!(psi_bra[idx₁], tags = "Site")
    prime!(psi_bra[idx₂], tags = "Site")
    i₁, i₂ = siteind(psi_intermediate, idx₁), siteind(psi_intermediate, idx₂)
    j₁, j₂ = siteind(psi_bra, idx₁), siteind(psi_bra, idx₂)
    @show i₁, i₂, j₁, j₂
    # println()


    # Compute the environment tensor T for the target two-qubit gate from scratch
    E_T = ITensor(1)
    psi_intermediate_copy = orthogonalize(psi_intermediate, length(psi_intermediate))
    psi_bra_copy = orthogonalize(psi_bra, length(psi_bra))
    
    for j in 1:length(psi_intermediate_copy)
      E_T *= psi_intermediate_copy[j] 
      E_T *= dag(psi_bra_copy[j])
    end
    # @show inds(E_T)
    noprime!(psi_bra)
   

	trace = real((E_T * target)[1])
    cost = compute_cost_function(psi_ket, psi_bra, gates_set, input_cutoff)
    @show trace, cost 


	# Compute the product of the target gate with its environment tensor & compute the cost function before updating the target gate 
	C_row = combiner(i₂, i₁)
	C_col = combiner(j₂, j₁)
	matrix_T = matrix(C_row * E_T * C_col, combinedind(C_row), combinedind(C_col))
	# show(IOContext(stdout, :limit=>false), "text/plain", matrix_T)
	# println()

	
	# Update the input angle based on the coefficients; one of them should give the maximum value and the other gives the minimum value
	coeff_A = imag(sum(matrix_T .* KroneckerZ))
	coeff_B = real(tr(matrix_T))
	θ₁ = atan(coeff_A, coeff_B)
	θ₂ = θ₁ + π
	@show coeff_A, coeff_B, coeff_A/coeff_B, θ₁, θ₂
	# println()


	# For debugging: search for the optimal angle by brute-force way in the range of [-2π, 2π] and check whether the optimal angle obtained from the coefficients is correct
	Fidelity_values = []
	for θ in range(-2π, 2π; length = 1000)
		updated_T = op(input_sites, "Rzz", idx₁, idx₂; ϕ=θ)
		push!(Fidelity_values, real((E_T * updated_T)[1]))
	end
	# @show Fidelity_values	
	@show maximum(Fidelity_values)
	
	# For Dubugging: search for the optimal angle by computing the cost function in a different way
	F_values = []
	for θ in range(-2π, 2π; length = 1000)
		tmp = coeff_B * cos(θ) + coeff_A * sin(θ)
		push!(F_values, tmp)
	end
	# @show F_values
	@show maximum(F_values)


	# Update the target Rzz gate using native Rzz gate constructor in ITensorMPS.jl
	updated_T1 = op(input_sites, "Rzz", idx₁, idx₂; ϕ=θ₁)
	updated_T2 = op(input_sites, "Rzz", idx₁, idx₂; ϕ=θ₂)


	if real((E_T * updated_T1)[1]) > real((E_T * updated_T2)[1])
		updated_T = updated_T1
	else
		updated_T = updated_T2
	end
	@show real((E_T * updated_T)[1]), real((E_T * updated_T1)[1]), real((E_T * updated_T2)[1])
	println()
	
	# Double check the optimal angle by computing the cost function after updating the target Rzz gate
	# F₁ = coeff_B * cos(θ₁) + coeff_A * sin(θ₁)
	# F₂ = coeff_B * cos(θ₂) + coeff_A * sin(θ₂)
	# @show F₁, real((E_T * updated_T1)[1])
	# @show F₂, real((E_T * updated_T2)[1])


	# @show inds(updated_T1)
	# @show inds(updated_T2)
	# @show j₁, j₂, i₁, i₂
	# println("")


    # Return the updated Rzz gate, the trace of the product of the target gate and environment tensor and the cost function
    return updated_T, trace, cost
end