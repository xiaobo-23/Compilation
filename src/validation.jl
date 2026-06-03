# 4/27/2026
# Geometry-agnostic measurement & validation for compiled Kitaev wave functions.
#
# Both geometries (honeycomb cluster and interferometer) are validated through
# the SAME code path. Everything geometry-specific is bundled by a builder into
# one "geometry context" NamedTuple:
#
#     geom = (; H, plaquettes, plaquette_ops)
#
#   - H             :: MPO                  the Kitaev Hamiltonian for energy
#   - plaquettes    :: Vector{Vector{Int}} six-site index list per hexagon
#   - plaquette_ops :: NTuple/Vector        Pauli string matched to the vertex order
#
# Use `honeycomb_geometry(...)` or `interferometer_geometry(...)` to build it,
# then `validate_circuit` / `validate_reference` are identical for both.

using ITensors
using ITensorMPS

include("plaquette.jl")                # hexagonal_plaquettes, interferometry_plaquette_{reference_obc,obc}
include("honeycomb_lattice.jl")        # LatticeBond/WedgeBond/Lattice + honeycomb lattice/wedge builders
include("hamiltonian.jl")              # cluster_energy_mpo, interferometer_energy_mpo, cumulative_gauge, …



# ------- Plaquette operator strings ----------------------------------------------------------------------
# Each string is paired with its geometry's vertex ordering and must stay in
# sync with the corresponding plaquette generator. ITensor's "iY" = i·σʸ keeps
# the MPO real; two "iY" give i² = −1, handled by `plaquette_sign`.
const PLAQUETTE_OPS            = ("iY", "Z", "X", "X", "Z", "iY")   # honeycomb cylinder
const PLAQUETTE_INTERFEROMETER = ("iY", "Z", "X", "iY", "Z", "X")   # interferometer 



# ------- Plaquette measurement ---------------------------------------------------------------------------
"""
    plaquette_sign(ops) -> Float64

Sign correction for the `iY` convention: `(iY)ⁿ = iⁿ`, and `n` (number of
`iY` factors) is even ⇒ ±1. Equals −1 for the standard two-`iY` strings.
"""
plaquette_sign(ops) = real(im ^ count(==("iY"), ops))


"""
    plaquette_mpo(p_sites, sites, ops) -> MPO

Build the plaquette MPO `∏ₖ ops[k](p_sites[k])` on `sites`. The six operators
act on distinct sites, so the product is order-independent; what matters is the
(`ops[k]` → `p_sites[k]`) pairing, which is fixed by the geometry's generator.
"""
function plaquette_mpo(p_sites, sites, ops)
    os = OpSum()
    add!(os, 1.0, Iterators.flatten(zip(ops, p_sites))...)
    return MPO(os, sites)
end


"""
    measure_plaquettes(ψ, sites, plaquettes, ops; imag_tol = 1e-8) -> Vector{Float64}

Per-plaquette ⟨Wp⟩ for any geometry. In the flux-free Kitaev ground state every
⟨Wp⟩ = +1, so closeness to +1 is a local flux-sector check that holds even at
moderate global fidelity. Warns if any contraction has a non-negligible
imaginary part (Wp is Hermitian, so it should be real).
"""
function measure_plaquettes(ψ::MPS, sites, plaquettes, ops; imag_tol::Real = 1e-8)
    s    = plaquette_sign(ops)
    vals = Vector{Float64}(undef, length(plaquettes))
    for (k, p) in enumerate(plaquettes)
        z = inner(ψ', plaquette_mpo(p, sites, ops), ψ)
        abs(imag(z)) < imag_tol ||
            @warn "Plaquette $k has non-negligible imaginary part" imag(z)
        vals[k] = s * real(z)
    end
    return vals
end


# ── Energy & variance ─────────────────────────────────────────────────────
"""
    measure_energy(ψ, H) -> Float64

Energy expectation `⟨ψ|H|ψ⟩` (real part). Cheap — one MPO–MPS contraction —
safe to call every sweep.
"""
measure_energy(ψ::MPS, H::MPO) = real(inner(ψ', H, ψ))


"""
    measure_variance(ψ, H) -> Float64

Energy variance `⟨ψ|H²|ψ⟩ − ⟨ψ|H|ψ⟩²` (real part). Small when ψ is near an
eigenstate of H. Expensive (contracts H twice) — call only for the final
report, never inside the per-sweep loop.
"""
function measure_variance(ψ::MPS, H::MPO)
    E  = real(inner(ψ', H, ψ))
    H2 = real(inner(H, ψ, H, ψ))
    return H2 - E^2
end


# ── Geometry context builders ─────────────────────────────────────────────
"""
    honeycomb_geometry(sites; Nx, Ny, Jx=1, Jy=1, Jz=1, κ=0, yperiodic=true) -> NamedTuple

Validation context for the honeycomb cylinder. Returns `(; H, plaquettes,
plaquette_ops)`.
"""
function honeycomb_geometry(sites; Nx::Integer, Ny::Integer,
        Jx::Real = 1.0, Jy::Real = 1.0, Jz::Real = 1.0,
        κ::Real = 0.0, yperiodic::Bool = true)
    H = cluster_energy_mpo(sites; Nx, Ny, Jx, Jy, Jz, κ, yperiodic)
    return (; H,
              plaquettes    = hexagonal_plaquettes(length(sites), Ny),
              plaquette_ops = PLAQUETTE_OPS)
end


"""
    interferometer_geometry(sites; Nx, Ny, Nx_unit, width_profile, constrictions,
                            Jx=1, Jy=1, Jz=1, κ=0, α=4) -> NamedTuple

Validation context for the interferometer (OBC). `width_profile` and
`constrictions` MUST match the DMRG run that produced the target MPS. Returns
`(; H, plaquettes, plaquette_ops)`.
"""
function interferometer_geometry(sites; Nx::Integer, Ny::Integer, Nx_unit::Integer,
        width_profile::AbstractVector{Int}, constrictions,
        Jx::Real = 1.0, Jy::Real = 1.0, Jz::Real = 1.0,
        κ::Real = 0.0, α::Real = 4.0)
    H       = interferometer_energy_mpo(sites; Nx, Ny, width_profile, constrictions,
                                        Jx, Jy, Jz, κ, α)
    x_gauge             = cumulative_gauge(width_profile)
    refs                = interferometry_plaquette_reference_obc(length(sites), Nx_unit, width_profile, x_gauge)
    plaquette_indices   = interferometry_plaquette_obc(width_profile, x_gauge, refs)
    return (; H,
              plaquettes    = [plaquette_indices[i, :] for i in 1:size(plaquette_indices, 1)],
              plaquette_ops = PLAQUETTE_INTERFEROMETER)
end


# ── Validators (geometry-agnostic; take a geometry context) ───────────────
"""
    validate_circuit(circuit_gates, ψ_initial, geom; cutoff = 1e-6) -> NamedTuple

Apply `circuit_gates` to `ψ_initial`, then measure per-plaquette ⟨Wp⟩ and
energy `E_opt = ⟨ψ|geom.H|ψ⟩`. Variance is opt-in via `measure_variance`.

Returns `(; ψ_opt, E_opt, wp_opt, plaquettes)`.
"""
function validate_circuit(circuit_gates, ψ_initial::MPS, geom; cutoff::Real = 1e-6)
    ψ_opt = deepcopy(ψ_initial)
    for layer in circuit_gates
        ψ_opt = apply(layer, ψ_opt; cutoff)
    end
    normalize!(ψ_opt)

    sites  = siteinds(ψ_opt)
    wp_opt = measure_plaquettes(ψ_opt, sites, geom.plaquettes, geom.plaquette_ops)
    E_opt  = measure_energy(ψ_opt, geom.H)
    return (; ψ_opt, E_opt, wp_opt, plaquettes = geom.plaquettes)
end


"""
    validate_reference(ψ_T, geom) -> NamedTuple

Per-plaquette ⟨Wp⟩ and energy `E_target` on the reference MPS. Variance is
opt-in via `measure_variance`.

Returns `(; E_target, wp_target, plaquettes)`.
"""
function validate_reference(ψ_T::MPS, geom)
    sites     = siteinds(ψ_T)
    wp_target = measure_plaquettes(ψ_T, sites, geom.plaquettes, geom.plaquette_ops)
    E_target  = measure_energy(ψ_T, geom.H)
    return (; E_target, wp_target, plaquettes = geom.plaquettes)
end