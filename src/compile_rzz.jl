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



# ------- Choose the geometry ------------------------------------------------
const GEOMETRY = :cluster        # :cluster or :interferometer


# ------- Shared hyperparameters (both geometries) --------------------------
const cutoff            = 1e-4
const validation_cutoff = 1e-8
const default_iters     = 25


# ------- Geometry-specific parameters --------------------------------------
if GEOMETRY === :cluster
    const model = (; Nx = 8, Ny = 3, Jx = 1.0, Jy = 1.0, Jz = 1.0, κ = -0.4, yperiodic = true)
    const N = model.Nx * model.Ny
	const target_mps_path = joinpath(@__DIR__, "..", "data",
        "kitaev_honeycomb_kappa-0.4_Lx4_Ly3.h5")
    const nsweeps = 20
    const stop_criteria = 1e-4
    const per_stage_stop_criteria = 1e-6
elseif GEOMETRY === :interferometer
    const Nx_unit = 9
    const Ny_unit = 3
    const model = (; Nx = 2*Nx_unit, Ny = Ny_unit + 1, Nx_unit = Nx_unit,
                     Jx = 1.0, Jy = 1.0, Jz = 1.0, κ = -0.2, α = 4.0,
                     width_profile = [3,4,4,3,4,4,3,4,4,4,4,3,4,4,3,4,4,3],
                     constrictions = ([17,20], [47,50]))
    const N = model.Nx * model.Ny - 6
	const target_mps_path = joinpath(@__DIR__, "..", "data",
        "interferometer_input_Nx9_Ny3_kappa-0.2.h5")
    const nsweeps = 2
    const stop_criteria = 1e-9
    const per_stage_stop_criteria = 1e-10
else
    error("GEOMETRY must be :cluster or :interferometer, got $GEOMETRY")
end




let
	# -----------------------------------------------------------------------------------------------------------------------------------------------------------
	# Set up single-qubit and two-qubit gates and optimize the circuit variationally to compile the target MPS
	# -----------------------------------------------------------------------------------------------------------------------------------------------------------
	println(repeat("-", 150))
	println("Variational circuit compilation: ground state preparation for the interferometer based on the Kitaev honeycomb model")
	println(repeat("-", 150), "\n")



	# Load the target ground-state MPS (file selected by GEOMETRY above).
    isfile(target_mps_path) ||
        error("target MPS not found: $target_mps_path  (check GEOMETRY / data dir)")

	ψ_T, sites = h5open(target_mps_path, "r") do file
		ψ = read(file, "psi", MPS)
		return ψ, siteinds(ψ)
	end
	@assert length(sites) == N "loaded $(length(sites)) sites but model implies N=$N — wrong file or params?"
	@info "Loaded target MPS" path=target_mps_path N=length(sites) maxlinkdim=maxlinkdim(ψ_T)
	println("\n")



	# ------- Initialize the trial MPS as a product state -------------------------------------------------------------------------------------------------------
	# A random MPS is also supported (see below) but the all-Up product state is
	# the cleanest reference for a variational compilation of the FM Kitaev model
	random_seed = 123
	Random.seed!(random_seed)
	state = fill("Up", N)
	ψ₀    = MPS(sites, state)
	# ψ₀	= random_mps(sites, state; linkdims = 8)  # bond-dimension-8 random MPS



	# ------- Construct the Hamiltonian as an MPO to measure the energy -----------------------------------------------------------------------------------------
	geom = GEOMETRY === :cluster ? 
		honeycomb_geometry(sites; model...) :
		interferometer_geometry(sites; model...)
	

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
		
		# Precompute the intermediate MPS states that are used in optimizing each layer
		nlayers = length(circuit_gates)
		ψ_left_collection = Vector{MPS}(undef, nlayers)
		ψ_left_collection[1] = ψ₀          # [2..end] filled just-in-time during the sweep	
		
		

		for iteration in 1 : nsweeps 
			println(repeat("-", 150))
			@printf "Sweep %d/%d\n" iteration nsweeps
			println("\n")

			ψ_right_collection = build_psi_right(circuit_gates, ψ_T; cutoff)   # ← one line, O(n)

			# Optimize each layer of the two-qubit gate in a forward sweeping order 
			for layer_idx in 1 : length(circuit_gates)
				optimization_gates = circuit_gates[layer_idx]
				idx_pairs = input_pairs[layer_idx]
				M = length(idx_pairs)


				# fresh = deepcopy(ψ₀); for i in 1:layer_idx-1; fresh = apply(circuit_gates[i], fresh; cutoff); end; normalize!(fresh)
				# @info 1 - abs(inner(fresh, ψ_left_collection[layer_idx])) < 1e-10

				
				# Read in the left and right intermediate MPS states for optimizing the current layer
				ψ_left = ψ_left_collection[layer_idx]
				ψ_right = ψ_right_collection[layer_idx]

				
				# Precompute the left and right environments for each gate.
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
							update_Rzz_from_env(E_T, sites, idx_pairs[k][1], idx_pairs[k][2])
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
							update_Rzz_from_env(E_T, sites, idx_pairs[k][1], idx_pairs[k][2])
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


				# Update the left intermediate MPS states for the next sweep
				if layer_idx < nlayers
					ψ_left_collection[layer_idx + 1] = normalize!(apply(optimization_gates, ψ_left_collection[layer_idx]; cutoff=cutoff))
				end
			end


			# Update the right intermediate MPS states for the next sweep
			ψ_right_collection = build_psi_right(circuit_gates, ψ_T; cutoff)
			

			
			# Compute the cost function after each sweep — bind once, use for both push! and printf.
			fidelity_sweep = compute_cost_function_multi_layers(ψ₀, ψ_T, circuit_gates, cutoff)
			en             = validate_circuit(circuit_gates, ψ₀, geom; cutoff=cutoff)

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


	# ------- Save the optimization results in an HDF5 file ----------------------------------------------
	# output_filename = "data/kitaev/kitaev_compilation_kappa-0.4_L$(n_total)_Rzz_test.h5"
	# h5open(output_filename, "w") do file
	# 	write(file, "cost_function", cost_function)
	# 	write(file, "energy_trace", energy_trace)
	# 	write(file, "plaquette_trace", Matrix(reduce(hcat, plaquette_trace)'))
	# 	write(file, "stage_starts", stage_starts)
	# end




	# ------- Validate the compiled circuit against the target MPS ---------------------------------------------
	# Measure energy + ⟨Wp⟩ on both states. Variance is opt-in (expensive), so we
	# compute it once here rather than inside the per-sweep validation.
	compiled = validate_circuit(circuit_gates, ψ₀, geom; cutoff = validation_cutoff)
	target   = validate_reference(ψ_T, geom)
	ΔE           = compiled.E_opt - target.E_target

	
	
	# Cross-check the target energy against the stored DMRG ground-state energy.
	E0_stored = h5open(target_mps_path, "r") do f; haskey(f, "E0") ? read(f, "E0") : nothing; end
	if E0_stored !== nothing
		ΔE0 = abs(target.E_target - E0_stored)
		ΔE0 < 1e-6 || @warn "Computed target energy disagrees with stored E0" E0_stored target.E_target ΔE0
	end



	# -------- Report ------------------------------------------------------------------------------------------
	section(title) = (println("\n", repeat("-", 100)); println(title); println(repeat("-", 150)))

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