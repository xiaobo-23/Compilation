# 5/10/2026
# Time evolution of the Kitaev honeycomb model with three-spin interactions using the TEBD algrithm

# ---- Imports ----
using ITensors
using ITensorMPS
using HDF5
using MKL
using LinearAlgebra
using TimerOutputs


# ---- Local helpers ----
include("HoneycombLattice.jl")
include("Entanglement.jl")
include("TopologicalLoops.jl")
include("CustomObserver.jl")          # DMRG observer; kept for ground-state comparisons


# ---- BLAS / LAPACK threading ----
# NOTE: ENV["MKL_NUM_THREADS"] etc. only take effect if set BEFORE `using MKL`.
# The runtime call below works regardless of load order.
const NTHREADS = 8
BLAS.set_num_threads(NTHREADS)

@show BLAS.get_config()
@show BLAS.get_num_threads()


## ---- Lattice: 4 × 3 × 2 honeycomb cluster on a cylinder (24 qubits, YPBC) ----
const Nx_unit = 4                     # Unit cells along x
const Ny_unit = 3                     # Unit cells along y (cylinder width)
const Nx      = 2 * Nx_unit           # Sublattice-resolved columns of sites
const Ny      = Ny_unit               # Sites per ring along y
const N       = Nx * Ny               # Total qubits = 24


## ---- Kitaev couplings ----
const Jx = 1.0
const Jy = 1.0
const Jz = 1.0
const κ  = -0.4                       # Three-spin interaction strength


## ---- TEBD hyperparameters ----
const dt          = 0.05              # Trotter step
const t_max       = 1.0               # Total real-time evolution
const nsteps      = round(Int, t_max / dt)
const cutoff_tebd = 1e-10             # MPS truncation cutoff per gate apply
const maxdim_tebd = 256               # Maximum bond dimension during TEBD


## ---- Profiling ----
const time_machine = TimerOutput()


let
  #-------------------------------------------------------------------------------------------------------
  # Read in the ground-state wave function an MPS
  #-------------------------------------------------------------------------------------------------------
  file = h5open("../data/kitaev_honeycomb_kappa-0.4_Lx4_Ly3.h5", "r")
  ψ_T = read(file, "psi", MPS)
  # @show typeof(ψ_T)
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
  
  
  #-------------------------------------------------------------------------------------------------------
  # Set up gates for time evolution using 2nd-order Trotter decomposition
  #-------------------------------------------------------------------------------------------------------

  

  
  # """Construc the Kitaev Hamiltonian as an MPO using the OpSum interface"""
  # os = OpSum()
  
  # # Set up the two-body interaction terms in the Hamiltonian
  # # Count the numbers of ⟨SxSx⟩, ⟨SySy⟩, ⟨SzSz⟩ bonds
  # xbond, ybond, zbond = 0, 0, 0        

  # for b in lattice
  #   xcoordinate = 2 * div(b.s1 - 1, 2 * Ny) + (iseven(b.s1) ? 2 : 1)
  #   ycoordinate = div(mod(b.s1 - 1, 2 * Ny), 2) + 1
  #   # @show b.s1, xcoordinate, ycoordinate

  #   if mod(xcoordinate, 2) == 0
  #     os .+= -Jy, "Sy", b.s1, "Sy", b.s2
  #     @show b.s1, b.s2, "Sy"
  #     ybond += 1
  #   else
  #     if b.s2 - b.s1 == 1
  #       os .+= -Jx, "Sx", b.s1, "Sx", b.s2
  #       @show b.s1, b.s2, "Sx"
  #       xbond += 1
  #     else
  #       os .+= -Jz, "Sz", b.s1, "Sz", b.s2
  #       @show b.s1, b.s2, "Sz"
  #       zbond += 1
  #     end
  #   end
  # end
  # # @show xbond, ybond, zbond


  # # Set up the three-spin interaction terms in the Hamiltonian
  # count = 0
  # for w in wedge 
  #   # Calculate the (x, y) coordinates of the site n based on C-style ordering
  #   tmp = div(w.s2 - 1, 2 * Ny)
  #   x = 2 * tmp + mod(w.s2 - 1, 2) + 1
  #   y = mod(div(w.s2 - 1, 2), Ny) + 1

  #   if mod(x, 2) == 1
  #     if w.s1 - w.s2 == 1 && w.s3 - w.s2 == 2 * Ny - 1
  #       os .+= κ, "Sx", w.s1, "Sy", w.s2, "Sz", w.s3
  #       @show w.s1, w.s2, w.s3, "Sx", "Sy", "Sz"
  #       count += 1
  #     end

  #     if w.s3 - w.s2 == 1 && w.s2 - w.s1 == 1
  #       os .+= κ, "Sz", w.s1, "Sy", w.s2, "Sx", w.s3
  #       @show w.s1, w.s2, w.s3, "Sz", "Sy", "Sx"
  #       count += 1
  #     end 

  #     if x != 1 && w.s2 - w.s1 == 2 * Ny - 1
  #       if w.s3 - w.s2 == 1
  #         os .+= κ, "Sy", w.s1, "Sz", w.s2, "Sx", w.s3
  #         @show w.s1, w.s2, w.s3, "Sy", "Sz", "Sx"
  #         count += 1  
  #       else
  #         os .+= κ, "Sy", w.s1, "Sx", w.s2, "Sz", w.s3
  #         @show w.s1, w.s2, w.s3, "Sy", "Sx", "Sz"
  #         count += 1  
  #       end
  #     end
  #   else
  #     if w.s3 - w.s2 == w.s2 - w.s1 == 1
  #       os .+= κ, "Sx", w.s1, "Sy", w.s2, "Sz", w.s3
  #       @show w.s1, w.s2, w.s3, "Sx", "Sy", "Sz"
  #       count += 1
  #     end

  #     if w.s2 - w.s3 == 1 && w.s2 - w.s1 == 2 * Ny - 1
  #       os .+= κ, "Sz", w.s1, "Sy", w.s2, "Sx", w.s3
  #       @show w.s1, w.s2, w.s3, "Sz", "Sy", "Sx"
  #       count += 1
  #     end

  #     if x != Nx && w.s3 - w.s2 == 2 * Ny - 1
  #       if w.s2 - w.s1 == 1
  #         os .+= κ, "Sx", w.s1, "Sz", w.s2, "Sy", w.s3
  #         @show w.s1, w.s2, w.s3, "Sx", "Sz", "Sy"
  #         count += 1
  #       else
  #         os .+= κ, "Sz", w.s1, "Sx", w.s2, "Sy", w.s3
  #         @show w.s1, w.s2, w.s3, "Sz", "Sx", "Sy"
  #         count += 1
  #       end
  #     end
  #   end
  # end
  # @show count 

  # if count != length(wedge)
  #   error("The number of three-spin interaction terms generated does not match the expected number.")
  # end




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