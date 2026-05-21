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
include("plaquette.jl")
include("validation.jl")
include("cached_environment.jl")
include("gates_update.jl")
include("update_gates.jl")



# ─── Set up parameters for multithreading and parallelization ────────────
BLAS.set_num_threads(8)
@info "BLAS configuration" vendor=BLAS.vendor() config=BLAS.get_config() threads=BLAS.get_num_threads()
# println()



# ─── Compilation parameters ──────────────────────────────────────────────
const model = (; Nx = 8, Ny = 3, Jx = 1.0, Jy = 1.0, Jz = 1.0, κ = -0.4, yperiodic = true)
const N = model.Nx * model.Ny            # Total number of qubits
const cutoff = 1e-4
const nsweeps = 1
const default_iters = 25                 # Number of iterations for optimizing each layer of two-qubit gates in the sweeping procedure
const stop_criteria = 1e-4               # Stopping criteria for the optimization of two-qubit gates; if the change of the cost function is smaller than this value, stop the optimization
const per_stage_stop_criteria = 1e-6     



let
	# -----------------------------------------------------------------------------------------
	# Set up and optimize single-qubit & two-qubit gates to variationally
	# compile the wave function of the Kitaev model on a cylinder.
	# -----------------------------------------------------------------------------------------
	println(repeat("-", 100))
	println("Variational circuit compilation: ground state preparation for the interferometer based on the Kitaev honeycomb model")
	println(repeat("-", 100), "\n")


	# Load the target ground-state MPS (e.g. Kitaev honeycomb).
	target_mps_path = joinpath(@__DIR__, "..", "data", 
		"kitaev_honeycomb_kappa-0.4_Lx4_Ly3.h5")

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


	#  ── Construct the Hamiltonian MPO for energy measurement. ─────────────── 
	H = energy_mpo(sites; model...)
	
	
	
	# -----------------------------------------------------------------------------------------
	# Project the initial MPS into the same topological sector as the target MPS
	# by applying ∏ₚ (1 + Wₚ)/√2, then align the global phase.
	# -----------------------------------------------------------------------------------------
	# println(repeat("-", 100))
	# println("Flux-sector projection of the initial MPS")
	# println(repeat("-", 100))
	
	
	# Build one (1 + Wp)/√2 tensor per plaquette.
	# indices = hexagonal_plaquettes(N, 4)
	# projection = ITensor[]
	# for p_sites in indices
	# 	s = sites[p_sites]
		
	# 	id_tensor = prod(op("Id", tmp_site) for tmp_site in s)
	# 	pauli_tensor = op("Y", s[1]) * op("Z", s[2]) * op("X", s[3]) * op("X", s[4]) * op("Z", s[5]) * op("Y", s[6])

	# 	hj = (id_tensor + pauli_tensor) / sqrt(2)
	# 	push!(projection, hj)
	# end

	
	# Apply the projector and align the global phase to maximize Re⟨ψ_T | ψ₀⟩.
	# ψ₀ = apply(projection, ψ₀; cutoff=cutoff)
	# fidelity₀ = inner(ψ_T, ψ₀)
	# ϕ_phase = angle(fidelity₀)
	# ψ₀[1] = ψ₀[1] * exp(-im * ϕ_phase)
	# fidelity₀_rotated = inner(ψ_T, ψ₀)	
	

	# Diagnostics: bond dimensions, overlaps, and per-plaquette ⟨Wp⟩.
	# result_proj = measure_plaquettes(ψ₀, sites; Ny = 4)
	# evals₀      = result_proj.wp
	# @printf "  bond dimensions     : %s\n" linkdims(ψ₀)
	# @printf "  ⟨ψ_T | ψ₀⟩          : %+.6f %+.6fi\n" real(fidelity₀)   imag(fidelity₀)
	# @printf "  ⟨ψ_T | ψ₀⟩ rotated  : %+.6f %+.6fi\n" real(fidelity₀_rotated) imag(fidelity₀_rotated)
	# println("⟨Wp⟩ on each hexagon:           ", evals₀)


	
	# -----------------------------------------------------------------------------------------
	# Build the variational brickwall ansatz used to compile the target MPS.
	#
	# Each repeating block has four sublayers:
	#   1. two-qubit gates on odd bonds  (i, i+1), i = 1, 3, 5, …
	#   2. two-qubit gates on even bonds (i, i+1), i = 2, 4, 6, …
	# -----------------------------------------------------------------------------------------
	# Configure the brickwall gate pattern of Rzz(θ) gates
	n_initial = 1
	n_total   = 1
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
			en             = validate_circuit(circuit_gates, ψ₀; Ny = model.Ny, Hamiltonian = H, cutoff = 1e-6)

			push!(cost_function,   fidelity_sweep)
			push!(energy_trace,    en.E_opt)
			push!(plaquette_trace, en.wp_opt)

			println()
			@printf "──── sweep %d/%d done  Fidelity = %+.6f  Energy = %+.6f\n" iteration nsweeps fidelity_sweep en.E_opt
			println(repeat("-", 100), "\n")


			# Per-stage early stop 
			if iteration > 1 && abs(fidelity_sweep - fidelity_prev_sweep) < per_stage_stop_criteria
				@info "Per-stage convergence reached; advancing to next stage" sweep = iteration ΔF = abs(fidelity_sweep - fidelity_prev_sweep) threshold = per_stage_stop_criteria
           		break
			end
			fidelity_prev_sweep = fidelity_sweep
		end
	end



	# Verifying the optimization results: report the cost function and energy trace during the optimization process.
	# @show cost_function
	# @show energy_trace
	# @show (optimization_trace - fidelity_trace)[1 : 20]


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


	# -----------------------------------------------------------------------------------------
	# Validate the optimized circuit by measuring the total energy of the system and
	# the hexagonal plaquette operators on both the compiled MPS and the target MPS, 
	# and report the optimization history.
	# -----------------------------------------------------------------------------------------
	compiled = validate_circuit(circuit_gates, ψ₀; Ny = model.Ny, Hamiltonian = H, cutoff = cutoff)
	target   = validate_reference(ψ_T;                       Ny = model.Ny, Hamiltonian = H)


	# Cross-check vs the DMRG ground-state energy stored alongside ψ_T.
	# A large gap here points at energy_mpo (bond/wedge dispatch or κ sign).
	E0_stored = h5open(target_mps_path, "r") do f; read(f, "E0"); end
	ΔE0       = abs(target.E_target - E0_stored)
	ΔE0 < 1e-6 || @warn "Computed target energy disagrees with stored E0" E0_stored target.E_target ΔE0
	ΔE       = compiled.E_opt - target.E_target


	println("\n", repeat("-", 100))
	println("Energy, variance, fidelity")
	println(repeat("-", 100))
	@printf "  %-10s E = %+.8f    variance = %.3e\n"		"target"   target.E_target target.var_target
	@printf "  %-10s E = %+.8f    variance = %.3e\n"		"compiled" compiled.E_opt  compiled.var_opt
	@printf "  %-10s ΔE = %+.8f   \n"						"gap"      ΔE 


	println("\n", repeat("-", 100))
	println("⟨Wp⟩ on each plaquette")
	println(repeat("-", 100))
	@printf "  %-12s %s\n" "optimized" join((@sprintf("%+.8f", x) for x in compiled.wp_opt), "  ")
	@printf "  %-12s %s\n" "target"    join((@sprintf("%+.8f", x) for x in target.wp_target), "  ")



	# Save the expectation values of the plaquette operators in an HDF5 file
	# h5open(output_filename, "r+") do file
	# 	write(file, "Wp_opt", compiled.wp_opt)
	# 	write(file, "Wp_target", target.wp_target)
	# end


  return
end