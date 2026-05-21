# 5/21/2026
# Functions to compute the environment tensor at a gate and update that
# gate via Evenbly-Vidal SVD (for SQ unitaries) or analytic argmax
# (for single-parameter Rzz).

using ITensors, ITensorMPS
using MKL
using LinearAlgebra
using Random
using Printf

include("compute_cost_function.jl")
include("cached_environment.jl")


const PauliX = ComplexF64[0  1;  1  0]
const PauliY = ComplexF64[0 -im; im  0]
const PauliZ = ComplexF64[1  0;  0 -1]

const PAULI_PRODUCTS = Dict{String, Matrix{ComplexF64}}(
    "Rxx" => kron(PauliX, PauliX),
    "Ryy" => kron(PauliY, PauliY),
    "Rzz" => kron(PauliZ, PauliZ),
)


"""
    build_env(ups, dns, ψ_left, ψ_right, k, idx_pairs) -> ITensor

Build the environment tensor for gate `k` directly from `ψ_left` and
`ψ_right`. The cached `ups[k]` / `dns[k]` already contain every gate of
the current layer EXCEPT the one at gate k's site(s), with all bond
Index objects inherited from ψ_left/ψ_right — so we can drop bare
`ψ_left` tensors in at the gate's sites without any divide-out.

Index convention of the returned tensor:
  - SQ at site i:      open indices (s_i, s_i').
  - NN Rzz at (i, j):  open indices (s_i, s_j, s_i', s_j').
"""
function build_env(ups, dns, ψ_left, ψ_right, k, idx_pairs)
    if length(idx_pairs[k]) == 1
        # Single-qubit gate
        i      = idx_pairs[k][1]
        bra_i  = prime(dag(ψ_right[i]); tags = "Site")
        return ups[k] * ψ_left[i] * bra_i * dns[k]
    else
        # Two-qubit Rzz gate at NN sites (i, j = i+1)
        i, j   = idx_pairs[k][1], idx_pairs[k][2]
        @assert abs(i - j) == 1 "Non-adjacent (i, j) = ($i, $j) not supported"
        bra_i  = prime(dag(ψ_right[i]); tags = "Site")
        bra_j  = prime(dag(ψ_right[j]); tags = "Site")
        return ups[k] * ψ_left[i] * ψ_left[j] * bra_i * bra_j * dns[k]
    end
end


# """
#     build_env(ups, dns, ψ_left, ψ_intermediate, ψ_right,
#               optimization_gates, k, idx_pairs) -> ITensor

# Build the environment tensor for gate `k` using cached `ups[k]`, `dns[k]`, and
# local tensors at the gate's target site(s).

# Single-qubit gate at site i:
#     E_T = ups[k] · ψ_left[i] · prime(dag(ψ_right[i]); "Site") · dns[k]
#     (works because SQ apply preserves bonds, so ψ_left and ψ_intermediate
#     share boundary bond indices.)

# Two-qubit Rzz gate at sites (i, j):
#     Divide out the current Rzz from ψ_intermediate at sites (i, j) via Rzz†,
#     then contract with the cached ups/dns and primed bra at the target sites.

# Returns:
#   - For SQ: an ITensor with 2 open indices (s_i, s_i')
#   - For Rzz: an ITensor with 4 open indices (s_i, s_j, s_i', s_j')
# """
# function build_env(ups, dns, ψ_left, ψ_intermediate, ψ_right,
#                    optimization_gates, k, idx_pairs)
#     if length(idx_pairs[k]) == 1
#         i = idx_pairs[k][1]
#         bra_i = prime(dag(ψ_right[i]); tags = "Site")
#         return ups[k] * ψ_left[i] * bra_i * dns[k]
#     else
#         i, j      = idx_pairs[k][1], idx_pairs[k][2]
#         Rzz_dag   = swapprime(dag(optimization_gates[k]), 0 => 1)

#         @assert i < j "build_env LR branch requires i < j, got (i=$i, j=$j)"
#         if abs(i - j) == 1
#             T_divided = noprime(ψ_intermediate[i] * ψ_intermediate[j] * Rzz_dag)
#             bra_i     = prime(dag(ψ_right[i]); tags = "Site")
#             bra_j     = prime(dag(ψ_right[j]); tags = "Site")

#             return ups[k] * T_divided * bra_i * bra_j * dns[k]
#         else
#             # Build joint tensor through the middle.
#             T_joint = ψ_intermediate[i]
#             for m in (i+1) : (j-1)
#                 T_joint = T_joint * ψ_intermediate[m]
#             end
#             T_joint = T_joint * ψ_intermediate[j]
            
#             # Apply Rzz† to divide out the target gate from sites (i, j).
#             T_divided = noprime(T_joint * Rzz_dag)
            
#             # Build middle environment from ψ_right side (mirroring the above).
#             middle_right = ITensor(1)
#             for m in (i+1) : (j-1)
#                 middle_right = middle_right * dag(ψ_right[m])
#             end
#             bra_i     = prime(dag(ψ_right[i]); tags = "Site")
#             bra_j     = prime(dag(ψ_right[j]); tags = "Site")

#             return ups[k] * T_divided * bra_i * middle_right * bra_j * dns[k]
#         end
#     end
# end



"""
    update_single_qubit_from_env(E_T, s) -> ITensor

Given the environment tensor `E_T` for a single-qubit gate at site `s`
(with open indices (s, s')), return the optimal unitary by SVD.

The polar decomposition of E_T gives the nearest unitary, which is the
fidelity-maximizing update for this gate.
"""
function update_single_qubit_from_env(E_T, s)
    U, S, V = svd(E_T, (s,))
    
    return dag(V) * delta(inds(S)[1], inds(S)[2]) * dag(U)
end


"""
    update_Rzz_from_env(E_T, sites, i, j) -> ITensor

Given the environment tensor `E_T` for an Rzz(ϕ) gate at sites (i, j)
(with open indices (s_i, s_j, s_i', s_j')), return the new Rzz operator
with the optimal angle.

Rzz(ϕ) is single-parameter, so the optimum is analytic:
    ϕ_opt = atan2( Im tr(E_T · Z⊗Z), Re tr(E_T) )
which maximizes Re Tr(Rzz(ϕ) · E_T) = cos(ϕ)·Re tr(E_T) + sin(ϕ)·Im tr(E_T·Z⊗Z).
"""
function update_Rzz_from_env(E_T, sites, i, j)
    P = PAULI_PRODUCTS["Rzz"]	

    s_i, s_j = sites[i], sites[j]
    C_row    = combiner(s_j,   s_i)              # unprimed (ket) side
    C_col    = combiner(s_j',  s_i')             # primed (bra) side
    matrix_T = matrix(C_row * E_T * C_col, combinedind(C_row), combinedind(C_col))

    
    # Update the input angle based on the coefficients 
	# One of them should give the maximum value and the other gives the minimum value
	coeff_A = imag(tr(matrix_T * P))
	coeff_B = real(tr(matrix_T))
	θ₁ = atan(coeff_A, coeff_B)
	θ₂ = θ₁ + π
	

	# Update the target gate using native gate constructor in ITensorMPS.jl
	updated_T1 = op(sites, "Rzz", i, j; ϕ=θ₁)
	updated_T2 = op(sites, "Rzz", i, j; ϕ=θ₂)


	if real((E_T * updated_T1)[1]) > real((E_T * updated_T2)[1])
		updated_T = updated_T1
	else
		updated_T = updated_T2
	end

    return updated_T
end