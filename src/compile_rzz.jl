# 4/20/2026
# Compiling the Kitaev model on the interferometer geometry
# Optimize parameters of single-qubit gates and two-qubit Rzz gates


using ITensors, ITensorMPS
using LinearAlgebra, MKL
using HDF5
using Random
using Printf



include("compute_cost_function.jl")
include("gates_initialization.jl")
include("validation.jl")
include("cached_environment.jl")
include("gates_update.jl")



# ------- Set up parameters for multithreading and parallelization -----------------------------------
BLAS.set_num_threads(8)
@info "BLAS configuration" vendor=BLAS.vendor() config=BLAS.get_config() threads=BLAS.get_num_threads()



# ------- Compilation parameters for the Kitaev cluster ----------------------------------------------
# const model = (; Nx = 8, Ny = 3, Jx = 1.0, Jy = 1.0, Jz = 1.0, κ = -0.4, yperiodic = true)
# const N = model.Nx * model.Ny            # Total number of qubits
# const cutoff = 1e-4
# const nsweeps = 100
# const default_iters = 25                 # Number of iterations for optimizing each layer of two-qubit gates in the sweeping procedure
# const stop_criteria = 1e-4               # Stopping criteria for the optimization of two-qubit gates; if the change of the cost function is smaller than this value, stop the optimization
# const per_stage_stop_criteria = 1e-6     



# ------- Compilation parameters for the interferometer ----------------------------------------------
const Nx_unit = 9                        # honeycomb unit cells along x
const Ny_unit = 3                        # honeycomb unit cells along y
const model = (;
    Nx       = 2 * Nx_unit,              # 18 lattice columns
    Ny       = Ny_unit + 1,              # 4  lattice rows
    Nx_unit  = Nx_unit,                  # passed through for the plaquette refs
    Jx = 1.0, Jy = 1.0, Jz = 1.0,
    κ = -0.2,                            # ← MUST match the interferometer DMRG run
    α = 4.0,                             # ← MUST match the interferometer DMRG run
    width_profile = [3, 4, 4, 3, 4, 4, 3, 4, 4, 4, 4, 3, 4, 4, 3, 4, 4, 3],
    constrictions = ([17, 20], [47, 50]),
)
const N = model.Nx * model.Ny - 6        # 66 total sites
const cutoff = 1e-4
const nsweeps = 100
const default_iters = 25                 # Number of iterations for optimizing each layer of two-qubit gates in the sweeping procedure
const stop_criteria = 1e-6               # Stopping criteria for the optimization of two-qubit gates; if the change of the cost function is smaller than this value, stop the optimization
const per_stage_stop_criteria = 1e-8     



let
	# -----------------------------------------------------------------------------------------
	# Set up and optimize single-qubit & two-qubit gates to variationally
	# compile the wave function of the Kitaev model on a cylinder.
	# -----------------------------------------------------------------------------------------
	println(repeat("-", 100))
	println("Variational circuit compilation: ground state preparation for the interferometer based on the Kitaev honeycomb model")
	println(repeat("-", 100), "\n")


	# Load the target ground-state MPS for the Kitaev model on a honeycomb lattice 
	# target_mps_path = joinpath(@__DIR__, "..", "data", 
	# 	"kitaev_honeycomb_kappa-0.4_Lx4_Ly3.h5")

	# Load the target ground-state MPS for the Kitaev model on a honeycomb lattice 
	target_mps_path = joinpath(@__DIR__, "..", "data", 
		"interferometer_input_Nx9_Ny3_kappa-0.2.h5")

	ψ_T, sites = h5open(target_mps_path, "r") do file
		ψ = read(file, "psi", MPS)
		return ψ, siteinds(ψ)
	end
	@info "Loaded target MPS" path=target_mps_path N=length(sites) maxlinkdim=maxlinkdim(ψ_T)
	println("")


	# ── Initialize the trial MPS as a product state. ─────────────────────────
	# A random MPS is also supported (see below) but the all-Up product state is
	# the cleanest reference for a Kitaev variational compilation: it has zero
	# entanglement, so any entanglement in ψ_opt comes from the optimized circuit.
	random_seed = 123
	Random.seed!(random_seed)
	state = fill("Up", N)
	ψ₀    = MPS(sites, state)
	# ψ₀	= random_mps(sites, state; linkdims = 8)  # bond-dimension-8 random MPS


	# ------- Construct the Hamiltonian as an MPO to measure the energy ----------------------------------
	# Set up the Hamiltonian MPO for the Kitaev model on the interferometer geometry
	# geom = honeycomb_geometry(sites; model...)
	
	
	# Set up the Hamiltonian MPO for the Kitaev model on the interferometer geometry
	geom = interferometer_geometry(sites; model...)
	

	
	# -----------------------------------------------------------------------------------------
	# Build the variational brickwall ansatz used to compile the target MPS.
	#
	# Each repeating block has four sublayers:
	#   1. two-qubit gates on odd bonds  (i, i+1), i = 1, 3, 5, …
	#   2. two-qubit gates on even bonds (i, i+1), i = 2, 4, 6, …
	# -----------------------------------------------------------------------------------------
	# Configure the brickwall gate pattern of Rzz(θ) gates
	n_initial = 5
	n_total   = 5
	brickwall_block = [
		[[i, i + 1] for i in 1 : 2 : N - 1],
		[[i, i + 1] for i in 2 : 2 : N - 1],
	]
	rzz_input_layers = repeat(brickwall_block, n_total)
	initial_layers = length(brickwall_block) * n_initial
	total_layers = length(rzz_input_layers)

	
	# Helper to expand one Rzz input layer into 3 circuit sublayers
	function build_dressed_block(rzz_pairs, sites; single_qubit_init = :random, rzz_init = :random)
		front, rzz, back = dressed_rzz_layer(rzz_pairs, sites; single_qubit_init=single_qubit_init, rzz_init=rzz_init)
		sq_pairs = [[k] for pair in rzz_pairs for k in pair] 
		return [
			(sq_pairs, front),
			(rzz_pairs, rzz),
			(sq_pairs, back),
		]
	end


	# Build the initial circuit
	input_pairs   = Vector{Vector{Int}}[]
	circuit_gates = Vector{ITensor}[]
	for k in 1 : initial_layers
		for (pairs, gates) in build_dressed_block(rzz_input_layers[k], sites; single_qubit_init = :random, rzz_init = :random)
			push!(input_pairs,   pairs)
			push!(circuit_gates, gates)
		end
	end

	

	# -----------------------------------------------------------------------------------------
	# Optimize the parameters of single-qubit & Rzz(θ) gates in the circuit layer by layer
	# -----------------------------------------------------------------------------------------
	cost_function = Float64[]
	energy_trace = Float64[]
	plaquette_trace = Vector{Float64}[]
	optimization_trace = Float64[]
	fidelity_trace = Float64[]
	stage_starts = Int[]

	
	for n_active in initial_layers : total_layers
		if n_active > initial_layers
			for (pairs, gates) in build_dressed_block(rzz_input_layers[n_active], sites; 
				single_qubit_init = :random, rzz_init = :identity)
				push!(input_pairs,   pairs)
				push!(circuit_gates, gates)
			end
		end

		# Record the sweep at which this stage starts 
		push!(stage_starts, length(cost_function) + 1)


		# Tracking per-stage early-stop to avoid over-optimizing each layer and getting stuck in local minima
		fidelity_prev_sweep = -Inf
		for iteration in 1 : nsweeps 
			println(repeat("-", 100))
			@printf "Sweep %d/%d\n" iteration nsweeps
			println("\n")

			# Optimize each layer of the two-qubit gate in a forward sweeping order 
			for layer_idx in 1 : length(circuit_gates)
				optimization_gates = circuit_gates[layer_idx]
				idx_pairs = input_pairs[layer_idx]
				M = length(idx_pairs)

				
				# Compress the optimization circuit from the initial MPS side
				ψ_left = deepcopy(ψ₀)
				if layer_idx > 1
					for idx in 1 : layer_idx - 1
						ψ_left = apply(circuit_gates[idx], ψ_left; cutoff=cutoff)
					end
					normalize!(ψ_left)
				end
				# ψ_left = ψ_ket_collection[layer_idx]
				
				
				# Compress the optimization circuit from the target MPS side 
				ψ_right = deepcopy(ψ_T)
				if layer_idx < length(circuit_gates)
					for tmp_idx in length(circuit_gates):-1:layer_idx + 1
						temporary_gates = deepcopy(circuit_gates[tmp_idx])
						for gate_idx in 1 : length(temporary_gates)
							temporary_gates[gate_idx] = dag(temporary_gates[gate_idx])
							swapprime!(temporary_gates[gate_idx], 0 => 1)
						end
						ψ_right = apply(temporary_gates, ψ_right; cutoff=cutoff)
					end
					normalize!(ψ_right)  
				end
				# ψ_right = ψ_bra_collection[layer_idx]
				

				# Precompute the left and right environments for each gate.
				# ψ_intermediate is no longer needed: ups/dns are built directly
				# from ψ_left/ψ_right with gates applied on the fly, so bond
				# Index IDs are inherited from ψ_left and never drift.
				ups = Vector{ITensor}(undef, M)
				dns = Vector{ITensor}(undef, M)

				init_ups_left!(ups, ψ_left, ψ_right, idx_pairs)
				precompute_dns!(dns, ψ_left, ψ_right, optimization_gates, idx_pairs, N)

				
				# Optimize all gates in the current layer by sweeping
				fidelity₁ = fidelity₂ = 0
				for iter_idx in 1:default_iters
					# Forward sweep from left to right
					for k in 1 : M
						E_T = build_env(ups, dns, ψ_left, ψ_right, k, idx_pairs)

						new_gate = if length(idx_pairs[k]) == 1
							update_single_qubit_from_env(E_T, sites[idx_pairs[k][1]])
						else
							update_Rzz_from_env(E_T, sites,
												idx_pairs[k][1], idx_pairs[k][2])
						end
						optimization_gates[k] = new_gate

						if k < M
							extend_ups!(ups, ψ_left, ψ_right, optimization_gates, idx_pairs, k)
						end
					end


					# Backward sweep from right to left
					for k in M : -1 : 1
						E_T = build_env(ups, dns, ψ_left, ψ_right, k, idx_pairs)

						new_gate = if length(idx_pairs[k]) == 1
							update_single_qubit_from_env(E_T, sites[idx_pairs[k][1]])
						else
							update_Rzz_from_env(E_T, sites,
												idx_pairs[k][1], idx_pairs[k][2])
						end
						optimization_gates[k] = new_gate

						if k > 1
							contract_dns_from_right!(dns, ψ_left, ψ_right, optimization_gates, idx_pairs, k)
						end
					end


					fidelity₂ = compute_cost_function_multi_layers(ψ₀, ψ_T, circuit_gates, cutoff)
					if iter_idx > 1 && abs(fidelity₂ - fidelity₁) < stop_criteria
						println("The change of the cost function is smaller than the stopping criteria. Stop the optimization of gates at layer $(layer_idx).")
						println([fidelity₁, fidelity₂, abs(fidelity₂ - fidelity₁)])
						break
					end
					fidelity₁ = fidelity₂
				end
			end


			# Compute the cost function after each sweep — bind once, use for both push! and printf.
			fidelity_sweep = compute_cost_function_multi_layers(ψ₀, ψ_T, circuit_gates, 1e-6)
			en             = validate_circuit(circuit_gates, ψ₀, geom; cutoff = 1e-6)

			push!(cost_function,   fidelity_sweep)
			push!(energy_trace,    en.E_opt)
			push!(plaquette_trace, en.wp_opt)

			println()
			@printf "──── sweep %d/%d done  Fidelity = %+.6f  Energy = %+.6f\n" iteration nsweeps fidelity_sweep en.E_opt
			println(repeat("-", 100), "\n")


			# # Per-stage early stop 
			# if iteration > 1 && abs(fidelity_sweep - fidelity_prev_sweep) < per_stage_stop_criteria
			# 	@info "Per-stage convergence reached; advancing to next stage" sweep = iteration ΔF = abs(fidelity_sweep - fidelity_prev_sweep) threshold = per_stage_stop_criteria
           	# 	break
			# end
			# fidelity_prev_sweep = fidelity_sweep
		end
	end



	# Verifying the optimization results: report the cost function and energy trace during the optimization process.
	# @show cost_function
	# @show energy_trace


	# -----------------------------------------------------------------------------------------
	# Save the optimization results in an HDF5 file for future analysis and visualization
	# -----------------------------------------------------------------------------------------
	# output_filename = "data/kitaev/kitaev_compilation_kappa-0.4_L$(n_total)_Rzz_test.h5"
	# h5open(output_filename, "w") do file
	# 	write(file, "cost_function", cost_function)
	# 	write(file, "energy_trace", energy_trace)
	# 	write(file, "plaquette_trace", Matrix(reduce(hcat, plaquette_trace)'))
	# 	write(file, "optimization_trace", optimization_trace)
	# 	write(file, "fidelity_trace", fidelity_trace)
	# 	write(file, "stage_starts", stage_starts)
	# end




	# ------- Validate the compiled circuit against the target MPS ---------------------------------------------
	# Measure energy + ⟨Wp⟩ on both states. Variance is opt-in (expensive), so we
	# compute it once here rather than inside the per-sweep validation.
	compiled = validate_circuit(circuit_gates, ψ₀, geom; cutoff = cutoff)
	target   = validate_reference(ψ_T, geom)
	ΔE           = compiled.E_opt - target.E_target

	
	
	# Cross-check the target energy against the stored DMRG ground-state energy.
	E0_stored = h5open(target_mps_path, "r") do f; haskey(f, "E0") ? read(f, "E0") : nothing; end
	if E0_stored !== nothing
		ΔE0 = abs(target.E_target - E0_stored)
		ΔE0 < 1e-6 || @warn "Computed target energy disagrees with stored E0" E0_stored target.E_target ΔE0
	end




	# -------- Report ------------------------------------------------------------------------------------------
	section(title) = (println("\n", repeat("-", 100)); println(title); println(repeat("-", 100)))

	section("Energy, variance, fidelity")
	@printf "  %-10s E  = %+.8f   \n"  "target"   target.E_target
	@printf "  %-10s E  = %+.8f   \n"  "compiled" compiled.E_opt
	@printf "  %-10s ΔE = %+.8f   \n"  "gap"      ΔE

	section("⟨Wp⟩ on each plaquette")
	for (label, wp) in (("optimized", compiled.wp_opt), ("target", target.wp_target))
		@printf "  %-12s %s\n" label join((@sprintf("%+.8f", x) for x in wp), "  ")
	end



	# Save the expectation values of the plaquette operators in an HDF5 file
	# h5open(output_filename, "r+") do file
	# 	write(file, "Wp_opt", compiled.wp_opt)
	# 	write(file, "Wp_target", target.wp_target)
	# end


  return
end