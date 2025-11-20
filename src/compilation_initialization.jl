## 11/20/2025
## Functions to initialize the target two-qubit gates and the randomized initial two-qubit gates

using ITensors
using ITensorMPS


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



# Function to generate a single-layer of random two-qubit gates as the initial unitaries to be optimized
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


# Function to generate multi-layers of random two-qubit gates as the initial unitaries to be optimized
function random_gates_multi_layers(pairs_array::Vector{Vector{Vector{Int64}}}, input_sites)
  circuit_depth = length(pairs_array)
  output_gates = []

  for idx in 1 : circuit_depth
    gates_layer = random_gates_single_layer(pairs_array[idx], input_sites)
    push!(output_gates, gates_layer)
  end

  return output_gates
end