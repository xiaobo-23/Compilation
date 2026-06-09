## 11/1/2025
## Compute the cost function given a set of two-qubit gates and two matrix product states
## Works for both a single layer of two-qubit gates and multiple layers of two-qubit gates  

using ITensors
using ITensorMPS


# Define a function to compute the cost function given two MPS and a set of unitaries
# and a single layer of two-qubit gates as input
function compute_cost_function(psi_ket::MPS, psi_bra::MPS, input_gates::Vector{ITensor}, input_cutoff::Float64 = 1e-14)
  
  # Throw an error if the lengths of the two MPS are not the same
  if length(psi_bra) != length(psi_ket)
    error("The lengths of the two MPS must be the same!")
  end
  
  
  # Apply the input gates to the left MPS
  intermediate_psi = apply(input_gates, psi_ket; cutoff=input_cutoff)
  normalize!(intermediate_psi)

  
  # Compute the inner product between the two MPS after applying the gates; benchmark purposes only
  psi_bra = orthogonalize(psi_bra, length(psi_bra))
  intermediate_psi = orthogonalize(intermediate_psi, length(intermediate_psi))
  
  inner_product = ITensor(1)
  for idx in eachindex(intermediate_psi)
    inner_product *= (intermediate_psi[idx] * dag(psi_bra[idx]))
  end
  
  
  # @show real(inner_product[1]) ≈ real(inner(intermediate_psi, psi_bra))
  return real(inner(intermediate_psi, psi_bra))
end


# Define the function to compute the cost function using two matrix product states
# and multiple layers of two-qubit gates as input
# function compute_cost_function_multi_layers(psi_ket::MPS, psi_bra::MPS, input_gates::Vector{Any}, 
#   input_cutoff::Float64 = 1e-14)

#   circuit_depth = length(input_gates)
#   for idx in 1 : circuit_depth
#     psi_ket = apply(input_gates[idx], psi_ket; cutoff=input_cutoff)
#   end
#   normalize!(psi_ket)

#   return real(inner(psi_bra, psi_ket))
# end



# ------- Multi-layer cost ---------------------------------------------------------------------------
# Two methods (one for the legacy `Vector{Any}` callers, one for the
# type-clean `Vector{Vector{ITensor}}` callers) sharing a single body.
# Once the legacy callers are migrated, replace both with the one definition
# in `_compute_cost_function_multi_layers_impl`.

function _compute_cost_function_multi_layers_impl(psi_ket::MPS, psi_bra::MPS,
                                                  input_gates,
                                                  input_cutoff::Float64)
    for layer in input_gates
        psi_ket = apply(layer, psi_ket; cutoff = input_cutoff)
    end
    normalize!(psi_ket)
    return real(inner(psi_bra, psi_ket))
end


# Legacy: `Vector{Any}` containing `Vector{ITensor}` layers.
# Used by compile_su4.jl, compile_rzz.jl.
function compute_cost_function_multi_layers(psi_ket::MPS, psi_bra::MPS,
                                            input_gates::Vector{Any},
                                            input_cutoff::Float64 = 1e-12)
    return _compute_cost_function_multi_layers_impl(psi_ket, psi_bra, input_gates, input_cutoff)
end


# Type-clean: `Vector{Vector{ITensor}}`.
# Used by compile_pauli.jl.
function compute_cost_function_multi_layers(psi_ket::MPS, psi_bra::MPS,
                                            input_gates::Vector{Vector{ITensor}},
                                            input_cutoff::Float64 = 1e-12)
    return _compute_cost_function_multi_layers_impl(psi_ket, psi_bra, input_gates, input_cutoff)
end