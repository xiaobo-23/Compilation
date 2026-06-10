# 5/21/2026
# Functions to compute the environment tensor at a gate and update that
# gate via Evenbly-Vidal SVD (for SQ unitaries) or analytic argmax
# (for single-parameter Rzz).

using ITensors, ITensorMPS
using LinearAlgebra


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


# ------------------------------------------------------------------------------------------
# Per-layer optimization driver shared by the forward and backward passes
# ------------------------------------------------------------------------------------------
"""
    optimize_layer!(optimization_gates, idx_pairs, ψ_left, ψ_right, sites, N;
                    default_iters, stop_criteria, layer_idx,
                    min_iters = 5, debug_refs = nothing) -> Float64

Optimize every gate of one circuit layer by alternating left→right and
right→left Evenbly–Vidal sub-sweeps over the cached `ups`/`dns` environments,
holding `ψ_left` (layers below, applied to ψ₀) and `ψ_right` (layers above,
folded into ψ_T from the bra side) fixed.

The global overlap Re⟨ψ_T|U|ψ₀⟩ is read off the environment for free: at the
last update of each right→left sub-sweep (gate k = 1), `E_T` contains every
other gate of the circuit at its current value, so `real((E_T * new_gate)[1])`
equals the overlap of the full current circuit, up to the truncations used to
build ψ_left/ψ_right. Early stopping compares consecutive readouts once
`iter_idx > min_iters`.

Mutates `optimization_gates` in place — pass `circuit_gates[layer_idx]`, which
aliases the same vector. Returns the final fidelity readout (valid at return
time; later layer updates change the circuit).

`debug_refs = (ψ₀, ψ_T, circuit_gates, cutoff)` additionally evaluates the
from-scratch `compute_cost_function_multi_layers` each inner iteration and
logs both values with their discrepancy — O(nlayers) MPS applies per call,
debugging only; leave as `nothing` in production runs.
"""
function optimize_layer!(optimization_gates, idx_pairs, ψ_left, ψ_right, sites, N;
                         default_iters::Int, stop_criteria::Real, layer_idx::Int,
                         min_iters::Int = 5, debug_refs = nothing)
    M = length(idx_pairs)

    # Precompute the left and right environments for each gate.
    ups = Vector{ITensor}(undef, M)
    dns = Vector{ITensor}(undef, M)

    init_ups_left!(ups, ψ_left, ψ_right, idx_pairs)
    precompute_dns!(dns, ψ_left, ψ_right, optimization_gates, idx_pairs, N)

    # Optimize all gates in the current layer by sweeping
    fidelity₁ = fidelity₂ = 0.0
    for iter_idx in 1:default_iters
        # Forward sub-sweep from left to right
        for k in 1 : M
            E_T = build_env(ups, dns, ψ_left, ψ_right, k, idx_pairs)

            new_gate = if length(idx_pairs[k]) == 1
                update_single_qubit_from_env(E_T, sites[idx_pairs[k][1]])
            else
                update_Rzz_from_env(E_T, sites, idx_pairs[k][1], idx_pairs[k][2])
            end
            optimization_gates[k] = new_gate

            if k < M
                extend_ups!(ups, ψ_left, ψ_right, optimization_gates, idx_pairs, k)
            end
        end


        # Backward sub-sweep from right to left
        for k in M : -1 : 1
            E_T = build_env(ups, dns, ψ_left, ψ_right, k, idx_pairs)

            new_gate = if length(idx_pairs[k]) == 1
                update_single_qubit_from_env(E_T, sites[idx_pairs[k][1]])
            else
                update_Rzz_from_env(E_T, sites, idx_pairs[k][1], idx_pairs[k][2])
            end
            optimization_gates[k] = new_gate

            if k > 1
                contract_dns_from_right!(dns, ψ_left, ψ_right, optimization_gates, idx_pairs, k)
            else
                # Global overlap for free: E_T at k = 1 contains every other gate at its
                # freshest value, so contracting with the new gate closes ⟨ψ_T|U|ψ₀⟩.
                fidelity₂ = real((E_T * new_gate)[1])

                # DEBUG: cross-check the environment readout fidelity₂ = real((E_T * gate)[1])
                # against an independent, essentially exact contraction of the SAME network
                # ⟨ψ_right| L |ψ_left⟩. Identical inputs ⇒ must agree to machine precision;
                # any larger deviation means the ups/dns environments are stale or mis-indexed.
                # fidelity_direct = real(inner(ψ_right, apply(optimization_gates, ψ_left; cutoff = 1e-15)))
                # abs(fidelity₂ - fidelity_direct) < 1e-10 || @warn "Env readout ≠ direct layer contraction — check ups/dns" layer_idx iter_idx fidelity₂ fidelity_direct
                # @info "Env readout vs direct contraction" layer_idx iter_idx fidelity₂ fidelity_direct
            end
        end


        # DEBUG: cross-check the environment readout against the from-scratch cost
        # function. Expect a discrepancy of O(nlayers × cutoff) — the two values follow
        # different truncation paths — shrinking with the cutoff, NOT machine precision.
        if debug_refs !== nothing
            ψ₀_dbg, ψ_T_dbg, circuit_dbg, cutoff_dbg = debug_refs
            fidelity_scratch = compute_cost_function_multi_layers(ψ₀_dbg, ψ_T_dbg, circuit_dbg, cutoff_dbg)
            @info "Compared cost function with direct computation" fidelity_sweep = fidelity_scratch fidelity_env = fidelity₂ discrepancy = abs(fidelity_scratch - fidelity₂)
        end

        if iter_idx > min_iters && abs(fidelity₂ - fidelity₁) < stop_criteria
            println("The change of the cost function is smaller than the stopping criteria. Stop the optimization of gates at layer $(layer_idx).")
            println([fidelity₁, abs(fidelity₂ - fidelity₁)])
            break
        end
        fidelity₁ = fidelity₂
    end

    return fidelity₂
end