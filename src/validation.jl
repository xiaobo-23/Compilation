# 4/27/2026
# Measurement primitives for compiled wave functions of the Kitaev honeycomb
# model: plaquette expectation values, energy/variance against the Kitaev
# Hamiltonian, and helpers `validate_circuit` / `validate_reference` that
# bundle the per-state measurements for the post-optimization report.


using ITensors
using ITensorMPS

include("plaquette.jl")
include("honeycomb.jl")


# Note: ITensor's "iY" is the real matrix i·Y, so (iY)·(iY) = -I and the
# MPO built from these ops equals **-Wp**. Recover ⟨Wp⟩ via -real(inner(…)).
const PLAQUETTE_OPS = ("iY", "Z", "X", "X", "Z", "iY")


"""
    plaquette_mpo(p_sites, sites) -> MPO

Build the (signed) Kitaev plaquette MPO on the six ordered sites in `p_sites`.
Returned MPO equals `-Wp` by the iY-convention; take `-real(inner(ψ', _, ψ))`
to recover ⟨Wp⟩.
"""
function plaquette_mpo(p_sites, sites)
    os = OpSum()
    os += Tuple(Iterators.flatten(zip(PLAQUETTE_OPS, p_sites)))
    return MPO(os, sites)
end



"""
    measure_plaquettes(ψ::MPS, sites; Ny::Integer) -> NamedTuple

Per-plaquette ⟨Wp⟩ on a honeycomb cylinder of circumference `Ny`. In the
flux-free Kitaev ground state every ⟨Wp⟩ = +1, so closeness of `wp` to `+1`
is the local flux-sector check that should pass even at moderate global fidelity.

Returns `(; wp, plaquettes)`.
"""
function measure_plaquettes(ψ::MPS, sites; Ny::Integer)
    plaquettes = hexagonal_plaquettes(length(sites), Ny)
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
    bonds = honeycomb_lattice_Cstyle(Nx, Ny; yperiodic)
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
        wedge = honeycomb_Cstyle_wedge(Nx, Ny; yperiodic)
        count = 0
        
        for w in wedge 
            tmp = div(w.s2 - 1, 2 * Ny)
            x = 2 * tmp + mod(w.s2 - 1, 2) + 1

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

        count == length(wedge) || error(
            "Wedge dispatch covered $count of $(length(wedge)) wedges — bond classification is inconsistent.")
    end

    return MPO(os, sites)
end



"""
    measure_energy(ψ::MPS, H::MPO) -> NamedTuple

Return `(; E, variance)` where `E = ⟨ψ|H|ψ⟩` and
`variance = ⟨ψ|H²|ψ⟩ - E²`. The variance is small when ψ is close to an
eigenstate of H — useful as a quality check independent of fidelity.
"""
function measure_energy(ψ::MPS, H::MPO)
    E  = real(inner(ψ', H, ψ))
    H2 = real(inner(H, ψ, H, ψ))
    return (; E, variance = H2 - E^2)
end



"""
    validate_circuit(circuit_gates, sites, state; 
    Ny::Integer, Hamiltonian, cutoff::Real = 1e-10) -> NamedTuple

Apply `circuit_gates` to the product state defined by `state` on `sites`,
then measure on the compiled MPS:
  - per-plaquette ⟨Wp⟩ (flux-sector check; +1 in the Kitaev ground state),
  - energy `E = ⟨ψ|H|ψ⟩` and variance `⟨H²⟩ - E²` against the Kitaev
    Hamiltonian built with `(Jx, Jy, Jz, κ)`.

Returns `(; ψ_opt, E_opt, var_opt, wp_opt, plaquettes)`.
"""
function validate_circuit(circuit_gates, ψ_initial::MPS; 
    Ny::Integer, Hamiltonian, cutoff::Real = 1e-10)
    
    # Apply the optimized circuit to the initial product state.
    ψ_opt = deepcopy(ψ_initial)
    for layer in circuit_gates
        ψ_opt = apply(layer, ψ_opt; cutoff)
    end
    normalize!(ψ_opt)

    
    # Measure ⟨Wp⟩ on every plaquette on the compiled MPS
    sites = siteinds(ψ_opt)
    plaquettes = hexagonal_plaquettes(length(sites), Ny)
    wp(ψ, p)   = -real(inner(ψ', plaquette_mpo(p, sites), ψ))
    wp_opt    = [wp(ψ_opt, p) for p in plaquettes]


    # Energy & variance of the compiled state
    en      = measure_energy(ψ_opt, Hamiltonian)
    E_opt   = en.E
    var_opt = en.variance

    
    return (; E_opt, var_opt, wp_opt, plaquettes)
end



"""
    validate_reference(ψ_T; Ny::Integer, Hamiltonian) -> NamedTuple

Return the plaquette expectation values ⟨Wp⟩ on the reference MPS and the energy and 
variance of the reference state.

In the Kitaev spin-liquid ground state every ⟨Wp⟩ = +1, so closeness of
`wp_target` to `+1` is the local flux-sector check that should pass even at
moderate global fidelity.

Returns `(; E_target, var_target, wp_target, plaquettes)`.
"""
function validate_reference(ψ_T; Ny::Integer, Hamiltonian)
    
    sites = siteinds(ψ_T)
        
    # Measure ⟨Wp⟩ on every plaquette on the target MPS
    plaquettes = hexagonal_plaquettes(length(sites), Ny)
    wp(ψ, p)   = -real(inner(ψ', plaquette_mpo(p, sites), ψ))
    wp_target = [wp(ψ_T,   p) for p in plaquettes]


    # Energy & variance 
    en      = measure_energy(ψ_T, Hamiltonian)
    E_target   = en.E
    var_target = en.variance

    
    return (; E_target, var_target, wp_target, plaquettes)
end