# 11/1/2025
# Functions used to update a target two-qubit gate within a set of two-qubit gates
# Use Evenbly-Vidal algorithms to compute the environment tensor and update the target gate


using ITensors
using ITensorMPS
using MKL
using LinearAlgebra
using Random

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
    # @show i₁, i₂, j₁, j₂
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
   

	trace = real((T * target)[1])
    cost = compute_cost_function(psi_ket, psi_bra, gates_set, input_cutoff)
    @show trace, cost 


	# Compute the product of the target gate with its environment tensor & compute the cost function before updating the target gate 
    # primed_inds   = filterinds(T; plev=1) 
	# unprimed_inds = filterinds(T; plev=0)
	# C_row = combiner(primed_inds...)
	# C_col = combiner(unprimed_inds...)

	C_row = combiner(i₁, i₂)
	C_col = combiner(j₁, j₂)
	matrix_T = matrix(C_row * T * C_col, combinedind(C_row), combinedind(C_col))
	# @show size(matrix_T)
	# @show inds(matrix_T)

	for i in 1:4
		for j in 1:4
			@printf("%10.4f + %10.4fi   ", real(matrix_T[i,j]), imag(matrix_T[i,j]))
		end
		println()
	end
	

	coeff_A = real(sum(matrix_T .* KroneckerZ))
	coeff_B = real(tr(matrix_T))
	θ₁ = atan(coeff_A, coeff_B)
	θ₂ = θ₁ + π
	
	@show coeff_A, coeff_B, coeff_A/coeff_B, θ₁, θ₂
	println("")


	# Update the target Rzz gate using customized formula for Rzz gates
	# Rzz_trace = []
	# for θ in range(θ₁ - π, θ₂ + π; length = 100)
	# 	a = exp(-im * θ/2)
	# 	b = exp( im * θ/2)
	# 	mat = [
	# 		a  0  0  0
	# 		0  b  0  0
	# 		0  0  b  0
	# 		0  0  0  a
	# 	]
	# 	updated_T = itensor(mat, j₁, j₂, i₁, i₂)
	# 	push!(Rzz_trace, real((T * updated_T)[1]))
	# end
	# @show Rzz_trace


	function zz_matrix_from_itensor()
		ZZi = op("Z", i₁) * op("Z", i₂)
		Crow = combiner(j₂, j₁)
		Ccol = combiner(i₂, i₁)
		M = matrix(Crow * ZZi * Ccol, combinedind(Crow), combinedind(Ccol))

		return M
	end

	function hand_zz()
		return Diagonal([1.0, -1.0, -1.0, 1.0])
	end


	M_it = zz_matrix_from_itensor()
	M_hand = hand_zz()
	@show M_it
	@show M_hand
	@show norm(M_it - M_hand)
	@show norm(M_it + M_hand)




	Rzz_trace = []
	for θ in range(-2π, 2π; length = 200)
		updated_T = op(input_sites, "Rzz", idx₁, idx₂; ϕ=θ)
		push!(Rzz_trace, real((T * updated_T)[1]))
	end
	@show maximum(Rzz_trace)

	# Update the target Rzz gate using native Rzz gate constructor in ITensorMPS.jl
	updated_T1 = op(input_sites, "Rzz", idx₁, idx₂; ϕ=θ₁)
	updated_T2 = op(input_sites, "Rzz", idx₁, idx₂; ϕ=θ₂)


	if real((T * updated_T1)[1]) > real((T * updated_T2)[1])
		updated_T = updated_T1
	else
		updated_T = updated_T2
	end
	@show real((T * updated_T)[1]), real((T * updated_T1)[1]), real((T * updated_T2)[1])
	# @show inds(updated_T1)
	# @show inds(updated_T2)
	# @show j₁, j₂, i₁, i₂
	println("")


    # Return the updated two-qubit gate, the trace of the product of the target gate and environment tensor
    # and the cost function
    return updated_T, trace, cost
end