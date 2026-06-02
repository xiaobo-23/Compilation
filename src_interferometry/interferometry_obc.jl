# 12/10/2025
# Simuulate the interferometry setup of the 2D Kitaev model on a honeycomb lattice using DMRG
# The code constructs the Hamiltonian and obtain the ground-state wave function; used as the input for state compilation


using ITensors
using ITensorMPS
using HDF5
# using MKL
using AppleAccelerate
using LinearAlgebra
using TimerOutputs


include("interferometry_lattice.jl")
include("interferometry_plaquettes.jl")
include("hamiltonian.jl")
include("CustomObserver.jl")


# ------------- Threading setup --------------------------------------------------------------------------------------
# BLAS threads (matrix-multiply parallelism inside DMRG kernels).
# Works for MKL / OpenBLAS / AppleAccelerate alike via libblastrampoline.
const N_BLAS_THREADS = 8
BLAS.set_num_threads(N_BLAS_THREADS)


# ITensor block-sparse threading (parallelism across blocks of a sparse
# tensor). Uses Julia tasks — only meaningful if Julia was started with
# multiple threads.
if Threads.nthreads() > 1
    ITensors.Strided.disable_threads()            # avoid oversubscription with BLAS threads
    ITensors.enable_threaded_blocksparse()
end


@info "Threading setup" begin
    blas_vendor   = BLAS.vendor()
    blas_threads  = BLAS.get_num_threads()
    julia_threads = Threads.nthreads()
    blocksparse_threaded = ITensors.using_threaded_blocksparse()
end


# ------------- Interferometer geometry ------------------------------------------------------------------------------
# A Kitaev honeycomb lattice (Nx × Ny columns × rows) with two narrow
# constrictions: each constriction is a width-3 column flanked by width-4
# bulk regions.
const Nx_unit::Int   = 9                       # honeycomb unit cells along x
const Ny_unit::Int   = 3                       # honeycomb unit cells along y
const Nx::Int        = 2 * Nx_unit             # = 18 lattice columns (2 cols / unit cell)
const Ny::Int        = Ny_unit + 1             # = 4  lattice rows
const N_CONSTRICTION = 6                       # sites removed at the two constrictions
const N::Int         = Nx * Ny - N_CONSTRICTION   # = 66 total sites


# ------------- Kitaev Hamiltonian couplings -------------------------------------------------------------------------
# H = -Σ Jᵅ Sᵅᵢ Sᵅⱼ on bonds  +  κ Σ Sᵃᵢ Sᵇⱼ Sᶜₖ on wedges (TRSB term)
const Jx::Float64 = 1.0
const Jy::Float64 = 1.0
const Jz::Float64 = 1.0
const κ::Float64  = -0.2
const α::Float64  = 4.0                              # constriction-bond coupling enhancement


# ------------- Instrumentation --------------------------------------------------------------------------------------
const time_machine = TimerOutput()



let
    # ------- Obtain the ground state wavefunction of the interferometer using DMRG --------------------------------------
    header = repeat('#', 200)
    println(header)
    println(header)
    println("Obtain the ground state of the interferometry setup of the 2D Kitaev model using DMRG")
    println("")
    println("")


    # -------- Set up geometry of the interferometer --------------------------------------------------------------------
    # Set up the width and gauge profiles so that we can define the x and y coordinates for each lattice point
    # Allocate constrictions on the interferometer lattice 

    # # Example 1: Narrow constrictions on Lx = 8, Ly = 4 lattice
    # constriction₁ = [17, 20]
    # constriction₂ = [39, 42]

    # width_profile = Int[]
    # for i in 1:5
    # 	append!(width_profile, [3, 4, 4])
    # end
    # push!(width_profile, 3)


    # Example 2: Design 10 plaquettes between constrictions on Lx = 9, Ly = 4 lattice
    constriction₁ = [17, 20]
    constriction₂ = [47, 50]
    width_profile = [3, 4, 4, 3, 4, 4, 3, 4, 4, 4, 4, 3, 4, 4, 3, 4, 4, 3]

    # Collect all bonds that should get the enhanced coupling. Stored as
    # unordered (smaller_site, larger_site) pairs so we don't have to think
    # about bond orientation when looking up.
    unordered_pair(s1, s2) = s1 < s2 ? (s1, s2) : (s2, s1)
    constrictions          = (constriction₁, constriction₂)        # extensible: just push more in
    constriction_bond_set  = Set(unordered_pair(c[1], c[2]) for c in constrictions)



    # ------- Set up the bonds on the interferometer ---------------------------------------------------------------------
    println(repeat("*", 100))
    println("Setting up all bonds on the interferometer")

    lattice = interferometry_lattice_obc(Nx, Ny, N, width_profile)
    number_of_bonds = length(lattice)
    println("\n")
  
  
  
    # -------- Set up the three-site objects on the interferometer -------------------------------------------------------
    println(repeat("*", 100))
    println("Setting up all three-site objects on the interferometer")

    wedge = interferometry_wedge(Nx, Ny, N, width_profile)
    number_of_wedges = length(wedge)
    println("\n")



    # ------- Set up the two-body interactions in the Hamiltonian --------------------------------------------------------
    println(repeat("*", 100))
    println("Setting up two-body interactions in the Hamiltonian")

    x_gauge = Int[]
    for idx in 0:length(width_profile)
        append!(x_gauge, sum(width_profile[1:idx]))
    end
    # println("\nThe gauge for x coordinates of each lattice site is:")
    # @show x_gauge

    # Set up counters for different types of bonds 
    xbond::Int = 0
    ybond::Int = 0
    zbond::Int = 0


    # Loop through all the bonds in the lattice and set up the two-body interaction terms correspondingly.
    os = OpSum()
    if abs(Jx) > 1e-8 || abs(Jy) > 1e-8 || abs(Jz) > 1e-8
        for b in lattice
            ops = bond_operator(b, x_gauge, Ny)
            ops == nothing && continue

            scale = unordered_pair(b.s1, b.s2) ∈ constriction_bond_set ? α : 1.0
            J = ops[1] == "X" ? Jx : 
                ops[1] == "Y" ? Jy : Jz

            os .+= -scale * J, ops[1], b.s1, ops[2], b.s2
            @info "Two-body term" sites=(b.s1, b.s2) op=ops[1] coupling=-scale * J

            ops[1] == "X" && (xbond += 1)
            ops[1] == "Y" && (ybond += 1)  
            ops[1] == "Z" && (zbond += 1)
        end


        # ------- Validate the construction of the two-body interaction terms ------------------------------------------------
        total_bonds_assigned = xbond + ybond + zbond
        @info "Two-body bond counts" xbond ybond zbond total=total_bonds_assigned expected=number_of_bonds

        total_bonds_assigned == number_of_bonds || error("""
            Two-body bond count mismatch:
            x-bonds (b.s2 - b.s1 == Ny):     $xbond
            y-bonds (b.s2 - b.s1 == Ny - 1): $ybond
            z-bonds (even-x columns):        $zbond
            total assigned to OpSum:         $total_bonds_assigned
            expected (length of lattice):    $number_of_bonds
            missing:                         $(number_of_bonds - total_bonds_assigned)
            """)
        println("\n")
    end
 
  

    # ------- Set up the three-body interactions in the Hamiltonian ------------------------------------------------------
    if abs(κ) > 1e-8
        println(repeat("*", 100))
        println("Setting up three-body interactions in the Hamiltonian")
        
        wedge_count::Int = 0
        for w in wedge
            x_odd = isodd(column_index(w.s2, x_gauge))
            op1, op2, op3 = wedge_operators(w, x_odd, Ny)
            os .+= κ, op1, w.s1, op2, w.s2, op3, w.s3
            @info "Three-spin term" sites=(w.s1, w.s2, w.s3) ops=(op1, op2, op3)
            @debug "Three-spin term" sites=(w.s1, w.s2, w.s3) ops=(op1, op2, op3)
            @assert Set((op1, op2, op3)) == Set(("X", "Y", "Z")) "wedge_operators returned non-permutation $((op1,op2,op3)) for wedge $(w)"
            
            # Count the number of three-spin interaction terms
            wedge_count += 1
        end

        # Check the three-spin interaction terms count matches the number of three-site objects
        @info "Three-spin term count check" assigned=wedge_count expected=number_of_wedges
        wedge_count == number_of_wedges || error("""
            Three-spin term count mismatch:
            added to OpSum:   $wedge_count
            expected:         $number_of_wedges
            missing:          $(number_of_wedges - wedge_count)
            """)
        println("\n")
    end


  
    # ------- Run DMRG to find the ground-state wavefunction ---------------------------------------------------------------
    println(repeat("*", 100))
    println("Running DMRG simulations to find the ground-state wavefunction")


    # Initialize the wavefunction as a random MPS and set up the Hamiltonian as an MPO
    sites = siteinds("S=1/2", N)
    state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
    ψ₀ = randomMPS(sites, state, 8)
    H = MPO(os, sites)


    # Set up hyperparameters used in the DMRG simulations, including bond dimensions, cutoff etc.
    nsweeps = 10
    maxdim  = [20, 60, 100, 500, 800, 1000]
    cutoff  = [1E-10]
    eigsolve_krylovdim = 50
    # noise = [1E-6, 1E-7, 1E-8, 0.0]   # Add a noise term to prevent DMRG from getting stuck in local minima


    # Measure one-point functions of the initial state
    # Sx₀ = expect(ψ₀, "Sx", sites = 1 : N)
    # Sy₀ = -im * expect(ψ₀, "iSy", sites = 1 : N)
    # Sz₀ = expect(ψ₀, "Sz", sites = 1 : N)


    # Construct a custom observer and stop the DMRG calculation early if criteria are met
    custom_observer = CustomObserver()
    # @show custom_observer.etolerance
    # @show custom_observer.minsweeps
    
    @timeit time_machine "dmrg simulation" begin
        energy, ψ = dmrg(H, ψ₀; nsweeps, maxdim, cutoff, eigsolve_krylovdim, observer = custom_observer)
    end
    println("Final ground-state energy = $energy")
    println("\n")


  
    # ------- Measure various observables on the ground-state wavefunction ------------------------------------------------- 
    # Measure local observables (one-point functions)
    @timeit time_machine "one-point functions" begin
        Sx = expect(ψ, "Sx", sites = 1 : N)
        Sy = -im * expect(ψ, "iSy", sites = 1 : N)
        Sz = expect(ψ, "Sz", sites = 1 : N)
    end

    
    # Measure spin correlation functions (two-point functions)
    @timeit time_machine "two-point functions" begin
        xxcorr = correlation_matrix(ψ, "Sx", "Sx", sites = 1 : N)
        zzcorr = correlation_matrix(ψ, "Sz", "Sz", sites = 1 : N)
        yycorr = -1.0 * correlation_matrix(ψ, "iSy", "iSy", sites = 1 : N)
    end


   
    # ------- Measure the expectation values of the plaquette operators on each hexagon ------------------------------------
    println(repeat("*", 100))
    println("Measure expectation values of the plaquette operators on each hexagon")

    plaquette_ops     = ["iY", "Z", "X", "iY", "Z", "X"]
    plaquette_refs    = interferometry_plaquette_reference_obc(N, Nx_unit, width_profile, x_gauge)
    plaquette_indices = interferometry_plaquette_obc(width_profile, x_gauge, plaquette_refs)
    @timeit time_machine "plaquette operators" begin
        plaquette_vals    = measure_plaquettes(ψ, sites, plaquette_indices, plaquette_ops)
    end
    println("\n")


    # # ------- Compute the variance of the energy to check how good the ground-state wavefunction is ------------------------
    # println(repeat("*", 100))
    # println("Compute the variance of the energy to check how good the ground-state wavefunction is")

    # @timeit time_machine "compute the variance" begin
    #   H2 = inner(H, ψ, H, ψ)
    #   E₀ = inner(ψ', H, ψ)
    #   variance = real(H2) - real(E₀)^2
    # end
    # println("Variance of the energy is $variance")
    # println("\n")



    # ------- Print out a summary of results -------------------------------------------------------------------------------
    # println(repeat("*", 100))
    # println("Summary of results:")
    # println("\n")


    # println("Expectation values of the plaquette oprators on each hexagon are: ")
    # @info "Plaquette ⟨Wₚ⟩" 
    # for (p, w) in enumerate(plaquette_vals)
    #     @printf "  plaquette %3d : %+.8f\n" p w
    # end
    # println("\n")


    # println("Bond dimensions of the ground-state wavefunction are: ")
    # @show linkdims(ψ) 
    # println("\n")


    # println(header)
    # println(header)
    # println("\n")
 

    # ------- Save the ground-state wavefunction and various observables to an HDF5 file -----------------------------------
    h5open("data/interferometer_input_Nx$(Nx_unit)_Ny$(Ny_unit)_kappa$(κ).h5", "cw") do file
        write(file, "psi", ψ)
        write(file, "E0", energy)
        # write(file, "E0_bond", E_bond)
        # write(file, "E0_wedge", E_wedge)
        # write(file, "E0variance", variance)
        # write(file, "Ehist", custom_observer.ehistory)
        # write(file, "Bond", custom_observer.chi)
        write(file, "chi", linkdims(ψ))
        write(file, "plaquette", plaquette_vals)
    end

  return
end