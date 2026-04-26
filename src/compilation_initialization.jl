## 11/20/2025
## Functions to initialize the target two-qubit gates and the randomized initial two-qubit gates

using ITensors
using ITensorMPS
using LinearAlgebra
using Printf

# function ITensors.op(::OpName"Rzz", ::SiteType"S=1/2",
#                      s1::Index, s2::Index; θ::Real)
#     a = exp(-im * θ/2)
#     b = exp( im * θ/2)
#     mat = [
#         a  0  0  0
#         0  b  0  0
#         0  0  b  0
#         0  0  0  a
#     ]
#     return itensor(mat, s2', s1', s2, s1)
# end



# Function to generate a single-layer of two-qubit gates using the Heisenberg interaction
function heisenberg_gates_single_layer(input_pairs::Vector{Vector{Int64}}, 
  J::Float64, Δτ::Float64, input_sites)
  gates = ITensor[]
  
  for idx in eachindex(input_pairs)
    idx₁, idx₂ = input_pairs[idx][1], input_pairs[idx][2]
    s₁ = input_sites[idx₁]
    s₂ = input_sites[idx₂]

    # Define a two-qubit gate, using the Heisenberg interaction as an example 
    hj = 1/2 * J * op("S+", s₁) * op("S-", s₂) + 1/2 * J * op("S-", s₁) * op("S+", s₂) 
      + J * op("Sz", s₁) * op("Sz", s₂)
    Gj = exp(-im * Δτ/2 * hj)
    push!(gates, Gj)
  end

  return gates
end



# Function to generate multi-layers of two-qubit gates using the Heisenberg interaction
function heisenberg_gates_multi_layers(pairs_array::Vector{Vector{Vector{Int64}}}, 
  J::Float64, Δτ::Float64, input_sites)
  circuit_depth = length(pairs_array)
  output_gates = []

  for idx in 1 : circuit_depth
    gates_layer = heisenberg_gates_single_layer(pairs_array[idx], J, Δτ, input_sites)
    push!(output_gates, gates_layer)
  end

  return output_gates
end



# Function to generate a single-layer of random SU(4) gates as the initial unitaries
function random_gates_single_layer(input_pairs::Vector{Vector{Int64}}, input_sites)
  gates = ITensor[]
  for idx in eachindex(input_pairs)
    idx₁, idx₂ = input_pairs[idx][1], input_pairs[idx][2]
    s₁ = input_sites[idx₁]
    s₂ = input_sites[idx₂]

    # SVD a random tensor to obtain a random unitary by setting all the singular values to 1 
    G_opt = randomITensor(s₁', s₂', s₁, s₂)
    U, S, V = svd(G_opt, (s₁', s₂'))
    G_random = U * delta(inds(S)[1], inds(S)[2]) * dag(V)
    push!(gates, G_random)
  end

  return gates
end 


# Function to generate multi-layers of random SU(4) gates as the initial unitaries
function random_gates_multi_layers(pairs_array::Vector{Vector{Vector{Int64}}}, input_sites)
  circuit_depth = length(pairs_array)
  output_gates = []

  for idx in 1 : circuit_depth
    gates_layer = random_gates_single_layer(pairs_array[idx], input_sites)
    push!(output_gates, gates_layer)
  end

  return output_gates
end



# Function to generate a single-layer of mixed single-qubit & SU(4) gates as the initial unitaries
function single_layer_mixed(input_pairs::Vector{Vector{Int64}}, input_sites)
	gates = ITensor[]
	for idx in eachindex(input_pairs)
		if length(input_pairs[idx]) == 1
			idx₁ = input_pairs[idx][1]
			s₁ = input_sites[idx₁]

			# SVD a random tensor to obtain a random unitary by setting all the singular values to 1 
			G_opt = randomITensor(s₁', s₁)
			U, S, V = svd(G_opt, (s₁',))
			# @show inds(U), inds(S), inds(V)
			G_random = U * delta(inds(S)[1], inds(S)[2]) * dag(V)
			# @show inds(G_random)
		elseif length(input_pairs[idx]) == 2
			idx₁, idx₂ = input_pairs[idx][1], input_pairs[idx][2]
			s₁ = input_sites[idx₁]
			s₂ = input_sites[idx₂]

			# SVD a random tensor to obtain a random unitary by setting all the singular values to 1 
			G_opt = randomITensor(s₁', s₂', s₁, s₂)
			U, S, V = svd(G_opt, (s₁', s₂'))
			# @show inds(U), inds(S), inds(V)
			G_random = U * delta(inds(S)[1], inds(S)[2]) * dag(V)
			# @show inds(G_random)
		end

		push!(gates, G_random)
	end

	return gates
end 



# Function to generate a single-layer of mixed single-qubit & Rzz gates as the initial unitaries
function single_layer_mixed_Rzz(input_pairs::Vector{Vector{Int64}}, input_sites)
	gates = ITensor[]
	for pair in input_pairs
		if length(pair) == 1
			idx₁ = pair[1]
			s₁ = input_sites[idx₁]

			# SVD a random tensor to obtain a random unitary by setting the S matrix to be an identity matrix
			G_opt = randomITensor(s₁', s₁)
			U, S, V = svd(G_opt, (s₁'))
			G_random = U * delta(inds(S)[1], inds(S)[2]) * dag(V)

			# G_random = op("Id", s₁)
		elseif length(pair) == 2
			ϕ = π/2 * rand()
			G_random = op(input_sites, "Rzz", pair[1], pair[2]; ϕ=ϕ)
			# @show G_random


			# """Check the exponent convention of the Rzz gate in ITensorMPS.jl"""
			# i₁, i₂ = input_sites[pair[1]], input_sites[pair[2]]
			# ϕ_test = 0.3
			# G_random = op(input_sites, "Rzz", pair[1], pair[2]; ϕ=ϕ_test)
			# @show inds(G_random)
			# @show i₁, i₂

			# rows, cols = combiner(i₁, i₂), combiner(i₁', i₂')
			# G_random_matrix = matrix(rows * G_random * cols, combinedind(rows), combinedind(cols))
			# show(IOContext(stdout, :limit=>false), "text/plain", G_random_matrix)
			# println()

			# expected_factor1 = Diagonal([exp(-im*ϕ_test*z) for z in [1, -1, -1, 1]])
			# expected_factor2 = Diagonal([exp(-im*ϕ_test*z/2) for z in [1, -1, -1, 1]])

			# show(IOContext(stdout, :limit=>false), "text/plain", expected_factor1)
			# println()

			# @show isapprox(G_random_matrix, expected_factor1; atol=1e-10)  
			# @show isapprox(G_random_matrix, expected_factor2; atol=1e-10)


			# """Check the ordering of basis states"""
			# ZI = op("Z", i₁) * op("I", i₂)
			# rows = combiner(i₂, i₁)
			# cols = combiner(i₂', i₁')
			# ZI_matrix = matrix(rows * ZI * cols, combinedind(rows), combinedind(cols))
			# show(IOContext(stdout, :limit=>false), "text/plain", ZI_matrix)
			# println()


			# """Set up the Rzz gate using custom constructor"""
			# G_random_custom = op("RzzCustom", input_sites, idx₁, idx₂; θ=ϕ)
			# @show G_random_custom
		end

		push!(gates, G_random)
	end

	return gates
end 



# Function to generate multi-layers mixed single-qubit & SU(4) gates as the initial unitaries
# function multi_layers_mixed(pairs_array::Vector{Vector{Vector{Int64}}}, input_sites)
# 	circuit_depth = length(pairs_array)
# 	output_gates = []

# 	for idx in 1 : circuit_depth
# 		gates_layer = single_layer_mixed(pairs_array[idx], input_sites)
# 		push!(output_gates, gates_layer)
# 	end

# 	return output_gates
# end



# Function to generate multi-layers mixed single-qubit & Rzz gates as the initial unitaries
function multi_layers_mixed_Rzz(pairs_array::Vector{Vector{Vector{Int64}}}, input_sites)
	circuit_depth = length(pairs_array)
	output_gates = []

	for idx in 1 : circuit_depth
		gates_layer = single_layer_mixed_Rzz(pairs_array[idx], input_sites)
		push!(output_gates, gates_layer)
	end
 
	return output_gates
end



# Function to generate multi-layers mixed single-qubit & SU(4) gates as the initial unitaries
# function multi_layers_mixed(pairs_array::Vector{Vector{Vector{Int64}}}, input_sites)
# 	return [single_layer_mixed(pairs, input_sites) for pairs in pairs_array]
# end



# Function to generate multi-layers mixed single-qubit & Rzz gates as the initial unitaries
# function multi_layers_mixed_Rzz(pairs_array::Vector{Vector{Vector{Int64}}}, input_sites)
# 	return [single_layer_mixed_Rzz(pairs, input_sites) for pairs in pairs_array]
# end