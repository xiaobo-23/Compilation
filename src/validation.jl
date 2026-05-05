# 4/27/2026
# Compile the interferometer ground state on a quantum device
# This file contains the code to compute the expectation values of the plaquette operators on the optimized state and the target state

using ITensors
using ITensorMPS

include("plaquette.jl")
include("honeycomb.jl")

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




"""
    energy_mpo(sites; Nx, Ny, Jx = 1.0, Jy = 1.0, Jz = 1.0, κ = 0.0,
               yperiodic::Bool = true) -> MPO

Build the Kitaev honeycomb Hamiltonian as an MPO on the C-style site labelling:

    H = -Jx Σ Sxᵢ Sxⱼ  - Jy Σ Syᵢ Syⱼ  - Jz Σ Szᵢ Szⱼ
        + κ  Σ Sᵅᵢ Sᵝⱼ Sᵞₖ      (three-spin "wedge" terms)
        
Bond and wedge dispatch matches `src_evolution/kitaev_honeycomb.jl`, so the MPO
is consistent with the ground-state MPS stored in `data/kitaev_honeycomb_*.h5`.
Set `κ = 0.0` to skip the wedge construction.
"""

function energy_mpo(sites; Nx::Integer, Ny::Integer, 
                    Jx::Real = 1.0, Jy::Real = 1.0, Jz::Real = 1.0, 
                    κ::Real  = 0.0, yperiodic::Bool = true)

    @assert length(sites) == Nx * Ny "sites length $(length(sites)) ≠ Nx*Ny = $(Nx*Ny)"
    
    
    # Set up the Hamiltonian with two-body and three-spin terms as an MPO
    os = OpSum()
    

    # ── two-body Kitaev bonds ─────────────────────────────────────────────────
    bonds = honeycomb_lattice_Cstyle(Nx, Ny; yperiodic=yperiodic)
    xbond, ybond, zbond = 0, 0, 0


    for b in bonds
        xcoord = 2 * div(b.s1 - 1, 2 * Ny) + (iseven(b.s1) ? 2 : 1)

        if iseven(xcoord)
            os .+= -Jy, "Sy", b.s1, "Sy", b.s2
            # @show b.s1, b.s2, "Sy"
            ybond += 1
        elseif b.s2 - b.s1 == 1
            os .+= -Jx, "Sx", b.s1, "Sx", b.s2
            # @show b.s1, b.s2, "Sx"
            xbond += 1
        else
            os .+= -Jz, "Sz", b.s1, "Sz", b.s2
            # @show b.s1, b.s2, "Sz"
            zbond += 1
        end
    end
    xbond + ybond + zbond == length(bonds) || error(
            "Setting up $(xbond + ybond + zbond) bonds instead of $(length(bonds)) inconsistent.")


    # ── three-spin (wedge) terms ─────────────────────────────────────────
    if abs(κ) > 1e-12
        wedge = honeycomb_Cstyle_wedges(Nx, Ny; yperiodic=yperiodic)
        count = 0
        
        for w in wedge 
            tmp = div(w.s2 - 1, 2 * Ny)
            x = 2 * tmp + mod(w.s2 - 1, 2) + 1
            # y = mod(div(w.s2 - 1, 2), Ny) + 1

            if isodd(x)
                if w.s1 - w.s2 == 1 && w.s3 - w.s2 == 2 * Ny - 1
                    os .+= κ, "Sx", w.s1, "Sy", w.s2, "Sz", w.s3
                    # @show w.s1, w.s2, w.s3, "Sx", "Sy", "Sz"
                    count += 1
                end

                if w.s3 - w.s2 == 1 && w.s2 - w.s1 == 1
                    os .+= κ, "Sz", w.s1, "Sy", w.s2, "Sx", w.s3
                    # @show w.s1, w.s2, w.s3, "Sz", "Sy", "Sx"
                    count += 1
                end 

                if x != 1 && w.s2 - w.s1 == 2 * Ny - 1
                    if w.s3 - w.s2 == 1
                        os .+= κ, "Sy", w.s1, "Sz", w.s2, "Sx", w.s3
                        # @show w.s1, w.s2, w.s3, "Sy", "Sz", "Sx"
                        count += 1  
                    else
                        os .+= κ, "Sy", w.s1, "Sx", w.s2, "Sz", w.s3
                        # @show w.s1, w.s2, w.s3, "Sy", "Sx", "Sz"
                        count += 1  
                    end
                end
            else
                if w.s3 - w.s2 == w.s2 - w.s1 == 1
                    os .+= κ, "Sx", w.s1, "Sy", w.s2, "Sz", w.s3
                    # @show w.s1, w.s2, w.s3, "Sx", "Sy", "Sz"
                    count += 1
                end

                if w.s2 - w.s3 == 1 && w.s2 - w.s1 == 2 * Ny - 1
                    os .+= κ, "Sz", w.s1, "Sy", w.s2, "Sx", w.s3
                    # @show w.s1, w.s2, w.s3, "Sz", "Sy", "Sx"
                    count += 1
                end

                if x != Nx && w.s3 - w.s2 == 2 * Ny - 1
                    if w.s2 - w.s1 == 1
                        os .+= κ, "Sx", w.s1, "Sz", w.s2, "Sy", w.s3
                        # @show w.s1, w.s2, w.s3, "Sx", "Sz", "Sy"
                        count += 1
                    else
                        os .+= κ, "Sz", w.s1, "Sx", w.s2, "Sy", w.s3
                        # @show w.s1, w.s2, w.s3, "Sz", "Sx", "Sy"
                        count += 1
                    end
                end
            end
        end
        # @show count

        count == length(wedges) || error(
            "Wedge dispatch covered $count of $(length(wedges)) wedges — bond classification is inconsistent.")
    end


    return MPO(os, sites)
end