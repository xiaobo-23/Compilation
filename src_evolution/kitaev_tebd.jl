# 5/10/2026
# Time evolution of the Kitaev honeycomb model with three-spin interactions using the TEBD algrithm


# ------- Imports libraries -------------------------------------------------------------------------------------------
using ITensors
using ITensorMPS
using HDF5
using MKL
using LinearAlgebra
using TimerOutputs
using Printf


# ------- Local helpers -----------------------------------------------------------------------------------------------
include("honeycomb_lattice.jl")
include("build_gates.jl")
include("hamiltonian.jl")



# ------- BLAS / LAPACK threading -------------------------------------------------------------------------------------
const NTHREADS = 8
BLAS.set_num_threads(NTHREADS)

@show BLAS.get_config()
@show BLAS.get_num_threads()



# ------- Lattice: 4 × 3 × 2 honeycomb cluster on a cylinder ----------------------------------------------------------
const Nx_unit = 4                     # Unit cells along x
const Ny_unit = 3                     # Unit cells along y (cylinder width)
const Nx      = 2 * Nx_unit           # Sublattice-resolved columns of sites
const Ny      = Ny_unit               # Sites per ring along y
const N       = Nx * Ny               # Total qubits = 24



# ------- Kitaev couplings --------------------------------------------------------------------------------------------
const Jx = 1.0
const Jy = 1.0
const Jz = 1.0
const κ  = -0.2                      # Three-spin interaction strength



# ------- TEBD hyperparameters ----------------------------------------------------------------------------------------
const dt          = 0.05             # Trotter step
const t_max       = 1.0              # Total real-time evolution
const nsteps      = round(Int, t_max / dt)
const cutoff_tebd = 1e-14            # MPS truncation cutoff per gate apply
const maxdim_tebd = 500              # Maximum bond dimension during TEBD



# ------- Profiling ---------------------------------------------------------------------------------------------------
const time_machine = TimerOutput()



let
    # -------- Read in the ground-state wavefunction as an MPS ------------------------------------------------------------
    file = h5open("../data/kitaev_honeycomb_Lx4_Ly3_kappa0.0.h5", "r")
    ψ_T = read(file, "psi", MPS)
    sites = siteinds(ψ_T)
    close(file)


    if length(ψ_T) != N
        error("Loaded MPS has length $(length(ψ_T)); expected N=$N for $(Nx_unit)×$(Ny_unit)×2 cluster")
    end
    @info "Loaded MPS" length=length(ψ_T) maxlinkdim=maxlinkdim(ψ_T)



    # # Check the variance of the energy to see if the obtained state is close to an eigenstate of the Hamiltonian
    # @timeit time_machine "compaute the variance" begin
    #   H2 = inner(H, ψ_T, H, ψ_T)
    #   E₀ = inner(ψ_T', H, ψ_T)
    #   variance = H2 - E₀^2
    # end

    # println("\nGround-state energy: $E₀")
    # println("\nVariance of the energy is $variance")
    # println("\n")
  
  
    
    # -------- Set up gates for time evolution using 2nd-order Trotter decomposition --------------------------------------
    step = build_tebd_step(sites; Nx, Ny, Jx, Jy, Jz, dt, yperiodic=true)
    
    
    
    # -------- Time evolve the state using TEBD ----------------------------------------------------------------------------
    H  = hamiltonian_cluster(sites; Nx, Ny, Jx, Jy, Jz, yperiodic=true)
    E0 = measure_energy(ψ_T, H)
    @printf "step %3d  t=%.4f  E=%+.8f  ΔE/|E0|=%.2e  χ=%d\n" 0 0.0 E0 0.0 maxlinkdim(ψ_T)
    for step in 1:nsteps
        ψ_T = apply(step, ψ_T; cutoff=cutoff_tebd)
        normalize!(ψ_T)


        E = measure_energy(ψ_T, H)
        @printf "step %3d  t=%.4f  E=%+.8f  ΔE/|E0|=%.2e  χ=%d\n" step step*dt E abs((E-E0)/abs(E0)) maxlinkdim(ψ_T)
    end



    # # Save the results to an HDF5 file for later use in quantum circuit compilation
    # @show time_machine
    # h5open("../data/kitaev_honeycomb_kappa-0.4_Lx6_Ly4.h5", "w") do file
    #   write(file, "psi", ψ)
    #   write(file, "E0", energy)
    #   write(file, "variance", variance)
    #   write(file, "chi", linkdims(ψ))
    #   # write(file, "Sz0", Sz₀)
    #   # write(file, "Sz",  Sz)
    #   # write(file, "Czz", zzcorr)
    #   # write(file, "plaquette", plaquette_evals)
    #   write(file, "loop", loop_evals)
    # end


  return
end