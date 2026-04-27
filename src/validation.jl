# 4/27/2026
# Compile the interferometer ground state on a quantum device
# This file contains the code to compute the expectation values of the plaquette operators on the optimized state and the target state

using ITensors
using ITensorMPS

include("plaquette.jl")

const PLAQUETTE_OPS = ("iY", "Z", "X", "X", "Z", "iY")


"""
    plaquette_mpo(p_sites, sites)

    Build the Kitaev plaquette MPO `Wp = (iY) Z X X Z (iY)` on the six ordered
    sites in `p_sites`.
"""

function plaquette_mpo(p_sites, sites)
    os = OpSum()
    os += Tuple(Iterators.flatten(zip(PLAQUETTE_OPS, p_sites)))
    return MPO(os, sites)
end


"""
    validate_plaquettes(circuit_gates, sites, state, ψ_T; width = 4, cutoff = 1e-10) -> NamedTuple

    Apply `circuit_gates` to the product state defined by `state` on `sites`
    and return the plaquette expectation values ⟨Wp⟩ on both the compiled and
    target MPS for a width-`width` honeycomb cylinder.

    In the Kitaev spin-liquid ground state every ⟨Wp⟩ = +1, so closeness of
    `wp_opt` to `+1` is the local flux-sector check that should pass even at
    moderate global fidelity.

    Returns `(; ψ_opt, wp_opt, wp_target, plaquettes)`.
"""

function validate_plaquettes(circuit_gates, sites, state, ψ_T; 
        width::Integer = 4, cutoff::Real = 1e-10)
    
    # Apply the optimized circuit to the initial product state.
    ψ_opt = MPS(sites, state)
    for layer in circuit_gates
        ψ_opt = apply(layer, ψ_opt; cutoff)
    end
    normalize!(ψ_opt)

    # Measure ⟨Wp⟩ on every plaquette, on both states.
    plaquettes = hexagonal_plaquettes(length(sites), width)
    wp(ψ, p)   = -real(inner(ψ', plaquette_mpo(p, sites), ψ))

    wp_opt    = [wp(ψ_opt, p) for p in plaquettes]
    wp_target = [wp(ψ_T,   p) for p in plaquettes]

    return (; ψ_opt, wp_opt, wp_target, plaquettes)
end



"""
    measure_plaquettes(ψ::MPS, sites; width = 4) -> NamedTuple
 
    Measure the Kitaev plaquette expectation values ⟨Wp⟩ on a pre-existing MPS
    `ψ` for a width-`width` honeycomb cylinder.
    
    Use this when `ψ` has already been prepared (e.g. by flux-sector projection
    of the initial product state) and you only need to measure, not compile.
    
    Returns `(; wp, plaquettes)`.
"""

function measure_plaquettes(ψ::MPS, sites; width::Integer = 4)
    plaquettes = hexagonal_plaquettes(length(sites), width)
    wp = [-real(inner(ψ', plaquette_mpo(p, sites), ψ)) for p in plaquettes]
    return (; wp, plaquettes)
end