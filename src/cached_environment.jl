# 5/21/2026
# Helper functions for incremental left/right environment caching used by
# the cached-environment sweep optimization.
#
# Conventions:
#   - ups[k] = contraction of (current_state[m] · dag(ψ_right[m])) over
#             sites m = 1..(idx_pairs[k][1] - 1).
#   - dns[k] = contraction of (current_state[m] · dag(ψ_right[m])) over
#             sites m = (idx_pairs[k][end] + 1)..N.
#
#   current_state[m] is built on the fly:
#     - If site m is not covered by any gate in this layer, use ψ_left[m].
#     - If a gate g acts on site m alone (SQ), use noprime(g · ψ_left[m]).
#     - If a gate g acts on sites (m, m+1) (NN Rzz), use the joint
#       tensor noprime(g · ψ_left[m] · ψ_left[m+1]) for both sites
#       together (it is immediately contracted with the two bra tensors
#       into ups/dns, so we never need to SVD-split it).
#
# All bond Index objects come from ψ_left (and ψ_right on the bra side),
# so contractions are always legal — no apply()-induced bond ID drift.

using ITensors
using ITensorMPS
using MKL
using LinearAlgebra
using Random
using Printf

include("compute_cost_function.jl")


"""
    init_ups_left!(ups, ψ_left, ψ_right, idx_pairs)

Initialize `ups[1]` by absorbing all sites strictly before the first qubit
of gate 1. These sites are not covered by any gate in this layer, so
ψ_left is used as-is (no gate to apply).

* Loop is empty (and `ups[1] = ITensor(1)`) when `idx_pairs[1][1] == 1`.
* Otherwise it contracts sites `1..(idx_pairs[1][1] - 1)`.
"""
function init_ups_left!(ups, ψ_left, ψ_right, idx_pairs)
    first_site = idx_pairs[1][1]

    new_ups = ITensor(1)
    for m in 1 : (first_site - 1)
        new_ups = new_ups * ψ_left[m] * dag(ψ_right[m])
    end
    ups[1] = new_ups
end


"""
    init_dns_right!(dns, ψ_left, ψ_right, idx_pairs, N)

Initialize `dns[M]` by absorbing all sites strictly after the last qubit
of gate M. These sites are not covered by any gate in this layer, so
ψ_left is used as-is.

* Loop is empty (and `dns[M] = ITensor(1)`) when `idx_pairs[M][end] == N`.
* Otherwise it contracts sites `(idx_pairs[M][end] + 1)..N`.
"""
function init_dns_right!(dns, ψ_left, ψ_right, idx_pairs, N)
    M = length(idx_pairs)
    last_site = idx_pairs[M][end]

    new_dns = ITensor(1)
    for m in N : -1 : (last_site + 1)
        new_dns = new_dns * ψ_left[m] * dag(ψ_right[m])
    end
    dns[M] = new_dns
end


"""
    extend_ups!(ups, ψ_left, ψ_right, optimization_gates, idx_pairs, k)

After updating gate `k` (during the forward sweep), build `ups[k+1]` from
`ups[k]` by absorbing:
  1. gate k's site(s), with gate k applied on the fly to ψ_left, then
     contracted with the corresponding bra tensors.
  2. any gap sites between gate k's last qubit and gate k+1's first qubit
     (no gate active on those sites).

Should only be called for `k < M`.
"""
function extend_ups!(ups, ψ_left, ψ_right, optimization_gates, idx_pairs, k)
    new_ups = ups[k]

    # ── Absorb gate k's sites with the latest gate applied on the fly. ──
    if length(idx_pairs[k]) == 1
        i = idx_pairs[k][1]
        new_ups = new_ups *
                  noprime(optimization_gates[k] * ψ_left[i]) *
                  dag(ψ_right[i])
    else
        i, j = idx_pairs[k][1], idx_pairs[k][2]
        @assert abs(i - j) == 1 "Non-adjacent (i, j) = ($i, $j) not supported"
        local_joint = noprime(optimization_gates[k] * ψ_left[i] * ψ_left[j])
        new_ups = new_ups *
                  local_joint *
                  dag(ψ_right[i]) *
                  dag(ψ_right[j])
    end

    # ── Absorb any gap sites between gate k and gate k+1. ───────────────
    last_site_k     = idx_pairs[k][end]
    first_site_next = idx_pairs[k+1][1]
    for m in (last_site_k + 1) : (first_site_next - 1)
        new_ups = new_ups * ψ_left[m] * dag(ψ_right[m])
    end

    ups[k+1] = new_ups
end


"""
    contract_dns_from_right!(dns, ψ_left, ψ_right, optimization_gates, idx_pairs, k)

After updating gate `k` (during the backward sweep), build `dns[k-1]`
from `dns[k]` by absorbing:
  1. gate k's site(s), with gate k applied on the fly to ψ_left.
  2. any gap sites between gate k-1's last qubit and gate k's first qubit.

Should only be called for `k > 1`.
"""
function contract_dns_from_right!(dns, ψ_left, ψ_right, optimization_gates, idx_pairs, k)
    new_dns = dns[k]

    # ── Absorb gate k's sites with the latest gate applied on the fly. ──
    if length(idx_pairs[k]) == 1
        i = idx_pairs[k][1]
        new_dns = noprime(optimization_gates[k] * ψ_left[i]) *
                  dag(ψ_right[i]) *
                  new_dns
    else
        i, j = idx_pairs[k][1], idx_pairs[k][2]
        @assert abs(i - j) == 1 "Non-adjacent (i, j) = ($i, $j) not supported"
        local_joint = noprime(optimization_gates[k] * ψ_left[i] * ψ_left[j])
        new_dns = local_joint *
                  dag(ψ_right[i]) *
                  dag(ψ_right[j]) *
                  new_dns
    end

    # ── Absorb any gap sites between gate k-1 and gate k. ───────────────
    first_site_k    = idx_pairs[k][1]
    last_site_prev  = idx_pairs[k-1][end]
    for m in (first_site_k - 1) : -1 : (last_site_prev + 1)
        new_dns = ψ_left[m] * dag(ψ_right[m]) * new_dns
    end

    dns[k-1] = new_dns
end


"""
    precompute_dns!(dns, ψ_left, ψ_right, optimization_gates, idx_pairs, N)

Fill `dns[1..M]` once at the start of the layer optimization, before any
sweeping. Initializes `dns[M]` via `init_dns_right!`, then walks leftward
via `contract_dns_from_right!` to fill `dns[M-1], ..., dns[1]`.

Called ONCE per layer (not per iter_idx) — the backward sweep rebuilds
dns in-place each iter_idx.
"""
function precompute_dns!(dns, ψ_left, ψ_right, optimization_gates, idx_pairs, N)
    M = length(idx_pairs)
    init_dns_right!(dns, ψ_left, ψ_right, idx_pairs, N)
    for k in M : -1 : 2
        contract_dns_from_right!(dns, ψ_left, ψ_right, optimization_gates, idx_pairs, k)
    end
end