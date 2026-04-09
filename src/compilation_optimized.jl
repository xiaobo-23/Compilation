## 10/7/2025
## Compiling initial state of the Kitaev model on a cylinder geometry
## Optimize parameters of two-qubit gates

using ITensors
using ITensorMPS
using HDF5
using MKL
using LinearAlgebra
using TimerOutputs
using MAT
using Random


include("compute_cost_function.jl")
include("update_gates.jl")
include("compilation_initialization.jl")


# Set up parameters for multithreading and parallelization
MKL_NUM_THREADS = 8
OPENBLAS_NUM_THREADS = 8
OMP_NUM_THREADS = 8

# Monitor the number of threads used by BLAS and LAPACK
@info "BLAS configuration" BLAS.get_config(), BLAS.get_num_threads()


# Set up the parameters for the optimization of two-qubit gates
const N = 24                            # Total number of qubits
const J₁ = 1.0
const τ = 1.0
const cutoff = 1e-4
const nsweeps = 100
const default_iters = 25                 # Number of iterations for optimizing each layer of two-qubit gates in the sweeping procedure
const stop_criteria = 1e-4               # Stopping criteria for the optimization of two-qubit gates; if the change of the cost function is smaller than this value, stop the optimization
# const time_machine = TimerOutput()     # Timing and profiling


let
	"""
	Compile the wave function of many-body Hamiltonian by optimizing the parameters of two-qubit gates 
	to approximate the target MPS
	"""
	
	println(repeat("#", 200))
	println(repeat("#", 200))
	println("\nSET UP AND OPTIMIZE TWO-QUBIT GATES TO VARIATIONALLY COMPILE THE WAVE FUNCTION OF THE KITAEV MODEL...")
  
	
	# Read in the target MPS which is the ground-state wave function of many-body Hamiltonian
	# e.g. the Kitaev model on a cylinderical geometry; 
	file = h5open("../data/kitaev_honeycomb_kappa-0.4_Lx4_Ly3.h5", "r")
	ψ_T = read(file, "psi", MPS)
	# @show typeof(ψ_T)
	sites = siteinds(ψ_T)
	close(file)


	
	"""Initialize the original MPS in a product state """
	Random.seed!(897)
	# sites = siteinds("S=1/2", N; conserve_qns=false)
	# state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
	state = ["Up" for n in 1:N]
	ψ₀ = MPS(sites, state)                          # Initialize the originial MPS as a product state
	# ψ₀ = random_mps(sites, state; linkdims=8)     # Initialize the original random MPS
	# @show linkdims(ψ₀)


	#*******************************************************************************************************************************
	#*******************************************************************************************************************************
	"""Applying projectors ∏ₚ(1 + Wₚ) to the Neel state MPS"""
	indices = [1  2  7  6  11 12; 3  4  9  2  7  8; 5  6  11 4  9  10; 7  8  13 12 17 18; 9  10 15 8  13 14;
	    11 12 17 10 15 16; 13 14 19 18 23 24; 15 16 21 14 19 20; 17 18 23 16 21 22]

	projection = ITensor[]
	for idx in 1 : size(indices, 1)
	  tmp = indices[idx, :]
	  s₁, s₂, s₃, s₄, s₅, s₆ = sites[tmp[1]], sites[tmp[2]], sites[tmp[3]], sites[tmp[4]], sites[tmp[5]], sites[tmp[6]]

	  hj = 1/sqrt(2) * op("Id", s₁) * op("Id", s₂) * op("Id", s₃) * op("Id", s₄) * op("Id", s₅) * op("Id", s₆) +
	    1/sqrt(2) * op("Y", s₁) * op("Z", s₂) * op("X", s₃) * op("X", s₄) * op("Z", s₅) * op("Y", s₆)
	  push!(projection, hj)
	end

	ψ₀ = apply(projection, ψ₀; cutoff=cutoff)
	fidelity₀ = inner(ψ₀, ψ_T)


	println("\nInitial state bond dimensions after applying projectors: $(linkdims(ψ₀))")
	println("\nThe overlap between the projected initial state and the target state is: $fidelity₀")
	

	# Compute the expectation value of the plaquette operator defined on each hexagonal plaquette
	evals₀ = zeros(Float64, size(indices, 1))
	plaquette_operator = Vector{String}(["iY", "Z", "X", "X", "Z", "iY"])

	
	for idx in 1 : size(indices, 1)
	  os_wp = OpSum()
	  os_wp += plaquette_operator[1], indices[idx, 1], 
	    plaquette_operator[2], indices[idx, 2], 
	    plaquette_operator[3], indices[idx, 3], 
	    plaquette_operator[4], indices[idx, 4], 
	    plaquette_operator[5], indices[idx, 5], 
	    plaquette_operator[6], indices[idx, 6]

	    WP = MPO(os_wp, sites)
	  evals₀[idx] = -real(inner(ψ₀', WP, ψ₀))
	end

	
	@show evals₀

	# output_filename = "data/kitaev_compilation_kappa-0.4_N$(N)_projection.h5"
	# h5open(output_filename, "w") do file
	#   write(file, "fidelity", inner(ψ₀, ψ_T))
	#   write(file, "chi", linkdims(ψ₀))
	#   write(file, "chi_T", linkdims(ψ_T))
	#   write(file, "plaquette", plaquette_evals)
	# end

	# Measure local observables (one-point functions)
	# Sx₀, Sy₀, Sz₀ = zeros(Float64, N), zeros(ComplexF64, N), zeros(Float64, N)
	# Sx₀ = expect(ψ₀, "Sx", sites = 1 : N)
	# Sy₀ = -im*expect(ψ₀, "iSy", sites = 1 : N)
	# Sz₀ = expect(ψ₀, "Sz", sites = 1 : N)

	# @show Sx₀
	# @show Sy₀
	# @show Sz₀
	#*******************************************************************************************************************************
	#*******************************************************************************************************************************
  
  
  
	"""
		Construct the sequence of target gates and randomly intiialized gates to be optimized
		Define pairs of qubit indices for two-qubit gates 
	"""

	# Define the pairs of qubit indices for two-qubit gates based on the nearest-neighbor connectivity & bond in the target Hamiltonian
	# input_pairs = [
	# 				[[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18], [19, 20], [21, 22], [23, 24]],
	# 				[[2, 3], [4, 5], [8, 9], [10, 11], [14, 15], [16, 17], [20, 21], [22, 23]], 
	# 				[[1, 6], [7, 12], [13, 18], [19, 24]], 
	# 				[[2, 7], [8, 13], [14, 19]], 
	# 				[[4, 9], [10, 15], [16, 21]], 
	# 				[[6, 11], [12, 17], [18, 23]],
	# 			]
	# Define the additional pairs of qubit indices to cover nearest-neighbor sites in the MPS representation
	# auxiliary_layers=0
	# brickwall = [
	# 	[[i, i + 1] for i in 1 : 2 : N - 1],
	# 	[[i, i + 1] for i in 2 : 2 : N - 1],
	# ]
	# input_pairs = vcat(input_pairs, repeat(brickwall, auxiliary_layers))

	

	# Define the pairs of qubit indices for two-qubit gates in a brickwall pattern
	input_layers=3
	brickwall = [
		[[i, i + 1] for i in 1 : 2 : N - 1],
		[[i, i + 1] for i in 2 : 2 : N - 1],
	]
	input_pairs = repeat(brickwall, input_layers)	
	
	# Print the pairs of qubit indices for two-qubit gates in each layer 
	# & verify the consistency between the gate pattern design and the actual implementation of the optimization procedure
	println(repeat("#", 200), "\n")
	println("The pairs of qubit indices for two-qubit gates in each layer: ")
	for ( _, layer) in enumerate(input_pairs)
		@show layer
	end
	println(repeat("#", 200), "\n")


	# Initialize the two-qubit gates randomly for each layer of the optimization circuit;  
	circuit_gates = random_gates_multi_layers(input_pairs, sites)
	# @show circuit_gates


	# Check the consistency between the number of layers of two-qubit gates and the number of layers of input pairs
	if length(circuit_gates) != length(input_pairs)
		error("The number of layers of two-qubit gates does not match the number of layers of input pairs.")
	end



	"""Optimize the parameters of all SU(4) gates in the circuit layer by layer"""
	cost_function = zeros(Float64, nsweeps)
	reference = zeros(Float64, nsweeps)
	optimization_trace = []
	fidelity_trace = []


	for iteration in 1 : nsweeps 
		# Optimize each layer of the two-qubit gate in a forward sweeping order 
		for layer_idx in 1 : length(circuit_gates)
			optimization_gates = circuit_gates[layer_idx]
			idx_pairs = input_pairs[layer_idx]

			
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
			

			println("\n", repeat("#", 200))
			fidelity₁ = 0
			fidelity₂ = 0

			for iter_idx in 1 : default_iters
				# Update all two-qubit gates in the forward order
				println("Forward Propagation: @iteration = $iteration, layer = $layer_idx: up-down sweeping")
				for idx in 1 : length(idx_pairs)
					idx₁, idx₂ = idx_pairs[idx][1], idx_pairs[idx][2]
					@show idx₁, idx₂
					updated_gate, tmp_trace, tmp_cost = update_single_gate(
					ψ_left, ψ_right, optimization_gates, idx, idx₁, idx₂, cutoff
					)
					optimization_gates[idx] = updated_gate
					append!(optimization_trace, tmp_trace)
					append!(fidelity_trace, tmp_cost)
				end
				println("\n")
				

				# Update all two-qubit gates in the backward order
				println("Forward Propagation: @iteration = $iteration, layer = $layer_idx: bottom-up sweeping")
				for idx in length(idx_pairs):-1:1
					idx₁, idx₂ = idx_pairs[idx][1], idx_pairs[idx][2]
					@show idx₁, idx₂
					updated_gate, tmp_trace, tmp_cost = update_single_gate(
					ψ_left, ψ_right, optimization_gates, idx, idx₁, idx₂, cutoff
					)
					optimization_gates[idx] = updated_gate
					append!(optimization_trace, tmp_trace)
					append!(fidelity_trace, tmp_cost)
				end
				println("\n")


				if iter_idx == 1
					fidelity₂ = compute_cost_function_multi_layers(ψ₀, ψ_T, circuit_gates, cutoff)
				else
					fidelity₁ = fidelity₂
					fidelity₂ = compute_cost_function_multi_layers(ψ₀, ψ_T, circuit_gates, cutoff)
					
					if abs(fidelity₂ - fidelity₁) < stop_criteria
						println("\nThe change of the cost function is smaller than the stopping criteria. 
						Stop the optimization of two-qubit gates in this layer.")
						@show [fidelity₁, fidelity₂, abs(fidelity₂ - fidelity₁)]
						break
					end
				end
			end
			println(repeat("#", 200), "\n")
		end


		# Compute the cost function after each sweep
		cost_function[iteration] = compute_cost_function_multi_layers(ψ₀, ψ_T, circuit_gates, cutoff)
		# reference[iteration] = compute_cost_function_multi_layers(ψ₀, ψ_T, gates, cutoff)
	end



	
	
	
	"""Validate the expectation values of the plaquette operators defined on each hexagonal plaquette using the optimized MPS"""
	psi_test = MPS(sites, state)
	for idx in 1 : length(circuit_gates)
		psi_test = apply(circuit_gates[idx], psi_test; cutoff=cutoff)	
	end
	normalize!(psi_test)
	
	
	# Compute the expectation value of the plaquette operator defined on each hexagonal plaquette
	plaquette_operator = Vector{String}(["iY", "Z", "X", "X", "Z", "iY"])
	indices = [1  2  7  6  11 12; 3  4  9  2  7  8; 5  6  11 4  9  10; 7  8  13 12 17 18; 9  10 15 8  13 14;
	    11 12 17 10 15 16; 13 14 19 18 23 24; 15 16 21 14 19 20; 17 18 23 16 21 22]
	evals₁ = zeros(Float64, axes(indices, 1))
	evals₂ = similar(evals₁)

	
	for (idx, tmp) in enumerate(eachrow(indices))
		os_wp = OpSum()
		os_wp += plaquette_operator[1], tmp[1], 
			plaquette_operator[2], tmp[2], 
			plaquette_operator[3], tmp[3], 
			plaquette_operator[4], tmp[4], 
			plaquette_operator[5], tmp[5], 
			plaquette_operator[6], tmp[6]
		WP = MPO(os_wp, sites)


		# Validate the expectation values of the plaquette operators using the optimized MPS
		z₁ = inner(psi_test', WP, psi_test)
		abs(imag(z₁)) < 1e-8 || @warn "Non-negligible imaginary part: $z₁"
		
		
		# Validate the expectation values of the plaquette operators using the target MPS
		z₂ = inner(ψ_T', WP, ψ_T)
		abs(imag(z₂)) < 1e-8 || @warn "Non-negligible imaginary part: $z₂"

		evals₁[idx] = -real(z₁)
		evals₂[idx] = -real(z₂)
	end
	println("\nExpectation values of the plaquette operators defined on each hexagonal plaquette using the optimized MPS: ")
	@show evals₁
	println("\nExpectation values of the plaquette operators defined on each hexagonal plaquette using the target MPS: ")
	@show evals₂



	"""Print the history of the cost function and the difference between the optimization trace and the fidelity trace in the terminal"""
	println("\nThe history of the cost function during the optimization: ")
	@show cost_function
	println("\nComputing the fidelity using two different approaches & the difference should fluctuate around zero with machine precision: ")
	@show (optimization_trace - fidelity_trace)[1 : 20]

  

	"""Save the optimization results in an HDF5 file for future analysis and visualization"""
	output_filename = "data/kitaev/kitaev_compilation_kappa-0.4_L$(input_layers).h5"
	h5open(output_filename, "w") do file
	  write(file, "cost function", cost_function)
	#   write(file, "fidelity0", fidelity₀)
	#   write(file, "Wp_0", evals₀)
	  write(file, "Wp_1", evals₁)
	  write(file, "Wp_2", evals₂)
	  write(file, "optimization trace", optimization_trace)
	  write(file, "fidelity trace", fidelity_trace)
	end


  return
end