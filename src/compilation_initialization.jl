## 11/20/2025
## Functions to initialize the target two-qubit gates and the randomized initial two-qubit gates

using ITensors
using ITensorMPS
using LinearAlgebra
using Printf

# function ITensors.op(::OpName"RzzCustom", ::SiteType"S=1/2",
#                      s1::Index, s2::Index; θ::Real)
#     a = exp(-im * θ)
#     b = exp( im * θ)
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
		elseif length(pair) == 2
			# idx₁, idx₂ = pair[1], pair[2]
			angle = π/2 * rand()
			G_random = op(input_sites, "Rzz", pair[1], pair[2]; ϕ=angle)
			# @show G_random

			# tmp_matrix = matrix(combiner(inds(G_random)[3], inds(G_random)[4]) * G_random * combiner(inds(G_random)[1], inds(G_random)[2]))
			# for i in axes(tmp_matrix, 1)
			# 	for j in axes(tmp_matrix, 2)
			# 		@printf("%12.6f + %12.6fi   ", real(tmp_matrix[i,j]), imag(tmp_matrix[i,j]))
			# 	end
			# 	println()
			# end

			# G_random_custom = op("RzzCustom", input_sites, idx₁, idx₂; θ=angle)
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