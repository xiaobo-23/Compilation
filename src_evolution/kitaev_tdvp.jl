# 6/3/2026
# Compare TEBD vs TDVP for real-time evolution of the Kitaev honeycomb cluster
# (two-body terms only). Both evolve the SAME initial state under the SAME
# Hamiltonian; we track energy conservation for each and their mutual overlap.

# ------- Imports -----------------------------------------------------------
using ITensors
using ITensorMPS            # provides `tdvp` (post ITensorTDVP merge); else `using ITensorTDVP`
using HDF5
using MKL
using LinearAlgebra
using Printf

include("hamiltonian.jl")    # hamiltonian_cluster(sites; Nx, Ny, Jx, Jy, Jz, yperiodic)
include("build_gates.jl")    # build_two_body_gates(sites; Nx, Ny, Jx, Jy, Jz, yperiodic, dt)

# ------- BLAS threading ----------------------------------------------------
BLAS.set_num_threads(8)
@info "BLAS configuration" vendor=BLAS.vendor() threads=BLAS.get_num_threads()

# ------- Lattice + couplings (must match the loaded ground state) ----------
const Nx_unit = 4
const Ny_unit = 3
const Nx = 2 * Nx_unit           # 8 columns
const Ny = Ny_unit               # 3 sites per ring
const N  = Nx * Ny               # 24
const Jx, Jy, Jz = 1.0, 1.0, 1.0

# ------- Evolution hyperparameters -----------------------------------------
const dt        = 0.05
const t_max     = 1.0
const nsteps    = round(Int, t_max / dt)
const cutoff_ev = 1e-10
const maxdim_ev = 256

let
    # ---- Load the same ground state for both evolutions ------------------
    ψ0, sites = h5open("../data/kitaev_honeycomb_Lx4_Ly3_kappa0.0.h5", "r") do f
        ψ = read(f, "psi", MPS)
        (ψ, siteinds(ψ))
    end
    @assert length(ψ0) == N "loaded MPS length $(length(ψ0)) ≠ N=$N"
    @info "Loaded initial state" N=length(ψ0) maxlinkdim=maxlinkdim(ψ0)

    # ---- Shared Hamiltonian (MPO) ---------------------------------------
    H = hamiltonian_cluster(sites; Nx, Ny, Jx, Jy, Jz, yperiodic=true)

    # ---- TEBD gates from the SAME dispatch as H (2nd-order Trotter) ------
    half      = build_two_body_gates(sites; Nx, Ny, Jx, Jy, Jz, yperiodic=true, dt=dt)
    tebd_step = vcat(half, reverse(half))     # symmetric Strang, O(dt³) per step

    # ---- Independent copies for the two methods -------------------------
    ψ_tebd = deepcopy(ψ0)
    ψ_tdvp = deepcopy(ψ0)

    energy(ψ) = real(inner(ψ', H, ψ))
    E0 = energy(ψ0)

    # ---- Time evolve, measuring energy + mutual overlap each step -------
    @printf "%-4s %-7s | %-13s %-9s | %-13s %-9s | %-11s\n" "step" "t" "E_tebd" "ΔE/|E0|" "E_tdvp" "ΔE/|E0|" "|⟨T|V⟩|"
    @printf "%-4d %-7.3f | %-13.8f %-9.1e | %-13.8f %-9.1e | %-11.8f\n" 0 0.0 E0 0.0 E0 0.0 1.0

    for step in 1:nsteps
        # TEBD: one 2nd-order Trotter step
        ψ_tebd = apply(tebd_step, ψ_tebd; cutoff=cutoff_ev, maxdim=maxdim_ev)
        normalize!(ψ_tebd)

        # TDVP: one step of exp(-i H dt).  tdvp(H, τ, ψ) = exp(τ·H)·ψ  ⇒  τ = -im·dt
        ψ_tdvp = tdvp(H, -im*dt, ψ_tdvp; cutoff=cutoff_ev, maxdim=maxdim_ev,
                      normalize=true, outputlevel=0)

        E_t = energy(ψ_tebd)
        E_v = energy(ψ_tdvp)
        ov  = abs(inner(ψ_tebd, ψ_tdvp))
        @printf "%-4d %-7.3f | %-13.8f %-9.1e | %-13.8f %-9.1e | %-11.8f\n" step step*dt E_t abs((E_t-E0)/abs(E0)) E_v abs((E_v-E0)/abs(E0)) ov
    end

    return
end