# 4/20/2026
# Compiling the Kitaev model on the interferometer geometry
# Optimize parameters of single-qubit gates and two-qubit Rzz gates


using ITensors, ITensorMPS
using HDF5, MAT
using Random
using TimerOutputs
using LinearAlgebra, MKL



include("compute_cost_function.jl")
include("update_gates.jl")
include("compilation_initialization.jl")
include("plaquette.jl")
include("validation.jl")




# Set up parameters for multithreading and parallelization
BLAS.set_num_threads(8)
@info "BLAS configuration" vendor=BLAS.vendor() config=BLAS.get_config() threads=BLAS.get_num_threads()
println()

# MKL_NUM_THREADS = 8
# OPENBLAS_NUM_THREADS = 8
# OMP_NUM_THREADS = 8


# ─── Compilation parameters ──────────────────────────────────────────────
const N = 48                             # Total number of qubits
const J₁ = 1.0
const τ = 1.0
const cutoff = 1e-4
const nsweeps = 1
const default_iters = 25                 # Number of iterations for optimizing each layer of two-qubit gates in the sweeping procedure
const stop_criteria = 1e-4               # Stopping criteria for the optimization of two-qubit gates; if the change of the cost function is smaller than this value, stop the optimization


let
	# ---------------------------------------------------------------------------
	# Set up and optimize single-qubit & two-qubit gates to variationally
	# compile the wave function of the Kitaev model on a cylinder.
	#  ---------------------------------------------------------------------------
	println("─"^70)
	println("Variational circuit compilation: ground state preparation for the interferometer based on the Kitaev honeycomb model")
	println("─"^70, "\n")

	
	# Load the target ground-state MPS (e.g. Kitaev honeycomb, κ = 0.4, 6×4).
	target_mps_path = joinpath(@__DIR__, "..", "data", 
		"kitaev_honeycomb_kappa-0.4_Lx6_Ly4.h5")

	ψ_T, sites = h5open(target_mps_path, "r") do file
		ψ = read(file, "psi", MPS)
		return ψ, siteinds(ψ)
	end
	@info "Loaded target MPS" path=target_mps_path N=length(sites) maxlinkdim=maxlinkdim(ψ_T)


	# ── Initialize the trial MPS as a product state. ─────────────────────────
	# A random MPS is also supported (see below) but the all-Up product state is
	# the cleanest reference for a Kitaev variational compilation: it has zero
	# entanglement, so any entanglement in ψ_opt comes from the optimized circuit.
	random_seed = 100
	Random.seed!(random_seed)
	state = fill("Up", N)
	# state = ["Up" for n in 1:N]
	ψ₀    = MPS(sites, state)
	
	# Alternative initializations — uncomment as needed:
	# state = [isodd(n) ? "Up" : "Dn" for n in 1:N]   # Néel product state
	# ψ₀	= random_mps(sites, state; linkdims = 8)  # bond-dimension-8 random MPS

	
	
	
	# ---------------------------------------------------------------------------
	# Project the initial MPS into the same topological sector as the target MPS
	# by applying ∏ₚ (1 + Wₚ)/√2, then align the global phase.
	#---------------------------------------------------------------------------
	println("─"^70)
	println("Flux-sector projection of the initial MPS")
	println("─"^70)
	
	
	# Build one (1 + Wp)/√2 tensor per plaquette.
	indices = hexagonal_plaquettes(N, 4)
	projection = ITensor[]
	for p_sites in indices
		s = sites[p_sites]
		
		id_tensor = prod(op("Id", tmp_site) for tmp_site in s)
		pauli_tensor = op("Y", s[1]) * op("Z", s[2]) * op("X", s[3]) * op("X", s[4]) * op("Z", s[5]) * op("Y", s[6])

		hj = (id_tensor + pauli_tensor) / sqrt(2)
		push!(projection, hj)
	end

	
	# Apply the projector and align the global phase to maximize Re⟨ψ_T | ψ₀⟩.
	ψ₀ = apply(projection, ψ₀; cutoff=cutoff)
	fidelity₀ = inner(ψ_T, ψ₀)
	ϕ_phase = angle(fidelity₀)
	ψ₀[1] = ψ₀[1] * exp(-im * ϕ_phase)
	fidelity₀_rotated = inner(ψ_T, ψ₀)	
	

	# Diagnostics: bond dimensions, overlaps, and per-plaquette ⟨Wp⟩.
	result_proj = measure_plaquettes(ψ₀, sites; width = 4)
	evals₀      = result_proj.wp
	@printf "  bond dimensions     : %s\n" linkdims(ψ₀)
	@printf "  ⟨ψ_T | ψ₀⟩          : %+.6f %+.6fi\n" real(fidelity₀)   imag(fidelity₀)
	@printf "  ⟨ψ_T | ψ₀⟩ rotated  : %+.6f %+.6fi\n" real(fidelity₀_rotated) imag(fidelity₀_rotated)
	println("⟨Wp⟩ on each hexagon:           ", evals₀)



	
	# ---------------------------------------------------------------------------
	# Build the variational brickwall ansatz used to compile the target MPS.
	#
	# Each repeating block has four sublayers:
	#   1. single-qubit gates on all sites
	#   2. two-qubit gates on odd bonds  (i, i+1), i = 1, 3, 5, …
	#   3. single-qubit gates on interior sites 2:N-1
	#   4. two-qubit gates on even bonds (i, i+1), i = 2, 4, 6, …
	# The block is repeated `n_layers` times and capped with a final
	# single-qubit layer on every site.
	# ---------------------------------------------------------------------------

	# Configure the brickwall gate pattern by defining qubit indices
	n_layers = 6
	brickwall_block = [
		[[i] for i in 1 : N],
		[[i, i + 1] for i in 1 : 2 : N - 1],
		[[i] for i in 2 : N - 1],
		[[i, i + 1] for i in 2 : 2 : N - 1],
	]
	input_pairs = repeat(brickwall_block, n_layers)	
	push!(input_pairs, [[i] for i in 1 : N])

	
	# Randomly initialize the mixed single- and two-qubit gates in each layer.
	circuit_gates = multi_layers_mixed_Rzz(input_pairs, sites)
	
	# Check the consistency between the number of layers of gates and the number of layers of input pairs
	@assert length(circuit_gates) == length(input_pairs) """
		Layer-count mismatch: got $(length(circuit_gates)) layers of gates for $(length(input_pairs)) layer specs. 
	""" 
	


	# """
	# 	Optimize the parameters of all SU(4) gates in the circuit layer by layer
	# """
	# cost_function = zeros(Float64, nsweeps)
	# reference = zeros(Float64, nsweeps)
	# optimization_trace = Float64[]
	# fidelity_trace = Float64[]


	# for iteration in 1 : nsweeps 
	# 	# Optimize each layer of the two-qubit gate in a forward sweeping order 
	# 	for layer_idx in 1 : length(circuit_gates)
	# 		optimization_gates = circuit_gates[layer_idx]
	# 		idx_pairs = input_pairs[layer_idx]

			
	# 		# Compress the optimization circuit from the initial MPS side
	# 		ψ_left = deepcopy(ψ₀)
	# 		if layer_idx > 1
	# 			for idx in 1 : layer_idx - 1
	# 				ψ_left = apply(circuit_gates[idx], ψ_left; cutoff=cutoff)
	# 			end
	# 			normalize!(ψ_left)
	# 		end
	# 		# ψ_left = ψ_ket_collection[layer_idx]
			
			
	# 		# Compress the optimization circuit from the target MPS side 
	# 		ψ_right = deepcopy(ψ_T)
	# 		if layer_idx < length(circuit_gates)
	# 			for tmp_idx in length(circuit_gates):-1:layer_idx + 1
	# 				temporary_gates = deepcopy(circuit_gates[tmp_idx])
	# 				for gate_idx in 1 : length(temporary_gates)
	# 					temporary_gates[gate_idx] = dag(temporary_gates[gate_idx])
	# 					swapprime!(temporary_gates[gate_idx], 0 => 1)
	# 				end
	# 				ψ_right = apply(temporary_gates, ψ_right; cutoff=cutoff)
	# 			end
	# 			normalize!(ψ_right)  
	# 		end
	# 		# ψ_right = ψ_bra_collection[layer_idx]
			
 

	# 		println("\n", repeat("#", 200))
	# 		fidelity₁ = 0
	# 		fidelity₂ = 0

	# 		for iter_idx in 1:default_iters
	# 			# Update all gates from top to bottom
	# 			println("Forward Propagation: @iteration = $iteration, layer = $layer_idx: top-down sweeping")
	# 			for idx in 1:length(idx_pairs)
	# 				updated_gate, tmp_trace, tmp_cost = if length(idx_pairs[idx]) == 1
	# 					update_single_qubit_gate(ψ_left, ψ_right, optimization_gates, idx, idx_pairs[idx][1], cutoff)
	# 				else
	# 					update_Rzz(ψ_left, ψ_right, optimization_gates, idx, idx_pairs[idx][1], idx_pairs[idx][2], sites, cutoff)
	# 				end
	# 				optimization_gates[idx] = updated_gate
	# 				append!(optimization_trace, tmp_trace)
	# 				append!(fidelity_trace, tmp_cost)
	# 			end
	# 			println("\n")

	# 			# Update all gates from bottom to top
	# 			println("Forward Propagation: @iteration = $iteration, layer = $layer_idx: bottom-up sweeping")
	# 			for idx in length(idx_pairs):-1:1
	# 				updated_gate, tmp_trace, tmp_cost = if length(idx_pairs[idx]) == 1
	# 					update_single_qubit_gate(ψ_left, ψ_right, optimization_gates, idx, idx_pairs[idx][1], cutoff)
	# 				else
	# 					idx₁, idx₂ = idx_pairs[idx][1], idx_pairs[idx][2]
	# 					update_Rzz(ψ_left, ψ_right, optimization_gates, idx, idx₁, idx₂, sites, cutoff)
	# 				end
	# 				optimization_gates[idx] = updated_gate
	# 				append!(optimization_trace, tmp_trace)
	# 				append!(fidelity_trace, tmp_cost)
	# 			end
	# 			println("\n")

	# 			fidelity₂ = compute_cost_function_multi_layers(ψ₀, ψ_T, circuit_gates, cutoff)
	# 			if iter_idx > 1 && abs(fidelity₂ - fidelity₁) < stop_criteria
	# 				println("\nThe change of the cost function is smaller than the stopping criteria. Stop the optimization of gates in this layer.")
	# 				println([fidelity₁, fidelity₂, abs(fidelity₂ - fidelity₁)])
	# 				break
	# 			end
	# 			fidelity₁ = fidelity₂
	# 		end
	# 		println(repeat("#", 200), "\n")
	# 	end

	# 	# Compute the cost function after each sweep
	# 	cost_function[iteration] = compute_cost_function_multi_layers(ψ₀, ψ_T, circuit_gates, cutoff)
	# 	# reference[iteration] = compute_cost_function_multi_layers(ψ₀, ψ_T, gates, cutoff)
	# end

	
	# """Print the history of the cost function and the difference between the optimization trace and the fidelity trace in the terminal"""
	# println("\nThe history of the cost function during the optimization: ")
	# @show cost_function
	# println("\nComputing the fidelity using two different approaches & the difference should fluctuate around zero with machine precision: ")
	# @show (optimization_trace - fidelity_trace)[1 : 20]

	
	# """Save the optimization results in an HDF5 file for future analysis and visualization"""
	# output_filename = "data/kitaev/kitaev_compilation_kappa-0.4_L$(n_layers)_Rzz.h5"
	# h5open(output_filename, "w") do file
	# 	write(file, "cost function", cost_function)
	# 	write(file, "fidelity0", fidelity₀)
	# 	write(file, "Wp_0", evals₀)
	# 	write(file, "Wp_1", evals₁)
	# 	write(file, "Wp_2", evals₂)
	# 	write(file, "optimization trace", optimization_trace)
	# 	write(file, "fidelity trace", fidelity_trace)
	# end


	
	
	
	# ---------------------------------------------------------------------------
	# Validate the optimized circuit by measuring the hexagonal plaquette
	# operators on both the compiled MPS and the target MPS, and report the
	# optimization history.
	# ---------------------------------------------------------------------------
	result  = validate_plaquettes(circuit_gates, sites, state, ψ_T; width = 4)
	
	println("\n", "─"^70)
	println("⟨Wp⟩ per plaquette")
	println("─"^70)
	@printf "  %-12s %s\n" "optimized" join((@sprintf("%+.8f", x) for x in result.wp_opt), "  ")
	@printf "  %-12s %s\n" "target"    join((@sprintf("%+.8f", x) for x in result.wp_target), "  ")
	# @printf "  %-12s max |⟨Wp⟩ - 1| = %.2e\n" "deviation" maximum(abs, result.wp_opt .- 1)
  

	# Save the expectation values of the plaquette operators in an HDF5 file
	# h5open(output_filename, "r+") do file
	# 	write(file, "Wp_opt", result.wp_opt)
	# 	write(file, "Wp_target", result.wp_target)
	# end


  return
end