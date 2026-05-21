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


# ═════════════════════════════════════════════════════════════════════════
# Helper functions for incremental left/right environment caching.
#
# Conventions:
#   - ups[k] = contraction of (ψ_intermediate[m] · dag(ψ_right[m])) over
#             sites m = 1..(idx_pairs[k][1] - 1).
#   - dns[k] = contraction of (ψ_intermediate[m] · dag(ψ_right[m])) over
#             sites m = (idx_pairs[k][end] + 1)..N.
#
#   - Both arrays have size M = length(idx_pairs).
#   - ups[1] = ITensor(1) only if idx_pairs[1][1] == 1; otherwise it
#     absorbs sites 1..(idx_pairs[1][1] - 1).
#   - dns[M] = ITensor(1) only if idx_pairs[M][end] == N; otherwise it
#     absorbs sites (idx_pairs[M][end] + 1)..N.
# ═════════════════════════════════════════════════════════════════════════


"""
    init_ups_left!(ups, ψ_intermediate, ψ_right, idx_pairs)

Initialize `ups[1]` by absorbing all sites strictly before the first qubit
of gate 1.

* Loop is empty (and `ups[1] = ITensor(1)`) when `idx_pairs[1][1] == 1`.
* Otherwise it contracts sites `1..(idx_pairs[1][1] - 1)`.

Should be called once at the start of each forward sweep — the sites it
covers are not touched by any gate in this layer, so the result is constant
across `iter_idx` iterations and could equivalently be cached once per layer.
"""
function init_ups_left!(ups, ψ_intermediate, ψ_right, idx_pairs)
    first_site = idx_pairs[1][1]
    
    new_ups = ITensor(1)
    for m in 1 : (first_site - 1)
        new_ups = new_ups * ψ_intermediate[m] * dag(ψ_right[m])
    end
    ups[1] = new_ups
end


"""
    init_dns_right!(dns, ψ_intermediate, ψ_right, idx_pairs, N)

Initialize `dns[M]` by absorbing all sites strictly after the last qubit
of gate M.

* Loop is empty (and `dns[M] = ITensor(1)`) when `idx_pairs[M][end] == N`.
* Otherwise it contracts sites `(idx_pairs[M][end] + 1)..N`.
"""
function init_dns_right!(dns, ψ_intermediate, ψ_right, idx_pairs, N)
    M = length(idx_pairs)
    last_site = idx_pairs[M][end]
    
    new_dns = ITensor(1)
    for m in N : -1 : (last_site + 1)
        new_dns = new_dns * ψ_intermediate[m] * dag(ψ_right[m])
    end
    dns[M] = new_dns
end


"""
    extend_ups!(ups, ψ_intermediate, ψ_right, idx_pairs, k)

After updating gate `k` (during the forward sweep), build `ups[k+1]` from
`ups[k]` by absorbing all sites from gate k's first qubit (inclusive) to
gate k+1's first qubit (exclusive).

For NN brickwall: absorbs gate k's two sites.
For LR gates: absorbs gate k's full span (e.g. 6 sites for a (1, 6) gate).
For patterns with gaps between gates: also absorbs the gap sites.

Should only be called for `k < M`.
"""
function extend_ups!(ups, ψ_intermediate, ψ_right, idx_pairs, k)
    first_site_k    = idx_pairs[k][1]
    first_site_next = idx_pairs[k+1][1]
    
    new_ups = ups[k]
    for m in first_site_k : (first_site_next - 1)
        new_ups = new_ups * ψ_intermediate[m] * dag(ψ_right[m])
    end
    ups[k+1] = new_ups
end


"""
    contract_dns_from_right!(dns, ψ_intermediate, ψ_right, idx_pairs, k)

After updating gate `k` (during the backward sweep), build `dns[k-1]` from
`dns[k]` by absorbing all sites from gate k-1's last qubit + 1 (inclusive)
to gate k's last qubit (inclusive), going leftward.

For NN brickwall: absorbs gate k's two sites.
For LR gates: absorbs gate k's full span.
For patterns with gaps: also absorbs gap sites between gate k-1 and gate k.

Should only be called for `k > 1`.
"""
function contract_dns_from_right!(dns, ψ_intermediate, ψ_right, idx_pairs, k)
    last_site_k    = idx_pairs[k][end]
    last_site_prev = idx_pairs[k-1][end]
    
    new_dns = dns[k]
    for m in last_site_k : -1 : (last_site_prev + 1)
        new_dns = new_dns * ψ_intermediate[m] * dag(ψ_right[m])
    end
    dns[k-1] = new_dns
end


"""
    precompute_dns!(dns, ψ_intermediate, ψ_right, idx_pairs, N)

Fill `dns[1..M]` once at the start of the layer optimization, before any
sweeping begins. Initializes `dns[M]` via `init_dns_right!`, then walks
leftward via `contract_dns_from_right!` to fill `dns[M-1], ..., dns[1]`.

Called ONCE per layer (not per iter_idx) — the result remains correct
across iter_idx iterations because the backward sweep rebuilds dns
in-place each time.
"""
function precompute_dns!(dns, ψ_intermediate, ψ_right, idx_pairs, N)
    M = length(idx_pairs)
    init_dns_right!(dns, ψ_intermediate, ψ_right, idx_pairs, N)
    for k in M : -1 : 2
        contract_dns_from_right!(dns, ψ_intermediate, ψ_right, idx_pairs, k)
    end
end


"""
    refresh_intermediate!(ψ_intermediate, new_gate, ψ_left, idx_pairs, k; cutoff = 1e-8)

After updating gate `k` to `new_gate`, refresh `ψ_intermediate` at the gate's
site(s) so it reflects the new gate (with the OLD gate's effect removed).

This uses bare `ψ_left[i]` (not the previous `ψ_intermediate[i]`), so we
don't need to undo the old gate via Rzz†.

* SQ gate at site i: `ψ_intermediate[i] = noprime(new_gate · ψ_left[i])`.
* NN Rzz at adjacent sites (i, j=i+1): SVD-split the joint tensor
  `noprime(new_gate · ψ_left[i] · ψ_left[j])` back into MPS form.
* LR Rzz: NOT IMPLEMENTED. Currently throws an error. (For long-range gates
  this requires SWAP-based application or careful manual handling of the
  intermediate bonds.)
"""
function refresh_intermediate!(ψ_intermediate, new_gate, ψ_left, idx_pairs, k;
                                cutoff::Real = 1e-8)
    if length(idx_pairs[k]) == 1
        # ── Single-qubit gate ────────────────────────────────────────────
        i = idx_pairs[k][1]
        ψ_intermediate[i] = noprime(new_gate * ψ_left[i])
    else
        i, j = idx_pairs[k][1], idx_pairs[k][2]
        if abs(i - j) == 1
            # ── Nearest-neighbor Rzz ─────────────────────────────────────
            T_new = noprime(new_gate * ψ_left[i] * ψ_left[j])
            
            s_i        = siteind(ψ_left, i)
            inds_for_U = i > 1 ? (s_i, linkind(ψ_left, i - 1)) : (s_i,)
            U, S, V    = svd(T_new, inds_for_U; cutoff)
            ψ_intermediate[i] = U
            ψ_intermediate[j] = S * V
        else
            error("refresh_intermediate! does not yet support long-range " *
                  "Rzz gates (i=$i, j=$j). Use NN brickwall for now.")
        end
    end
end