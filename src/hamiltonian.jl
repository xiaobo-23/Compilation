# 6/1/2026
# Helper functions that set up the Hamiltonian for the interferometer. 
# Construct interactions including two-body Kitaev interactions and three-body TRSB terms based on the geometry of the interferometer lattice.

using ITensors
using ITensorMPS

include("interferometer_lattice.jl")



# -------- Geometry helpers ------------------------------------------------------------------------
"""
    column_index(site::Int, x_gauge::AbstractVector{Int}) -> Int

Column number x of a site, defined by  x_gauge[x] < site <= x_gauge[x+1].
Throws if the site falls outside the gauge range.
"""
function column_index(site::Int, x_gauge::AbstractVector{Int})
    for idx in 1 : length(x_gauge) - 1
        if site > x_gauge[idx] && site <= x_gauge[idx + 1]
            return idx
        end
    end
    error("site=$site is outside x_gauge range $(extrema(x_gauge))")
end



# -------- Two-body Kitaev term dispatch -----------------------------------------------------------
"""
    bond_operator(b, x_gauge, Ny) -> Union{Nothing, NTuple{2,String}}

Return two-body interaction for the Kitaev flavor of bond `b`, or `nothing` if
the bond doesn't match any expected pattern. Dispatch is:

    even-x column                       → z-bond  (Y, Y)
    odd-x column, |s2-s1| == Ny         → x-bond  (X, X)
    odd-x column, |s2-s1| == Ny - 1     → y-bond  (Z, Z)
"""
function bond_operator(b::LatticeBond, x_gauge, Ny::Int)
    x = column_index(b.s1, x_gauge)
    if iseven(x)
        return ("Y", "Y")
    elseif abs(b.s1 - b.s2) == Ny
        return ("X", "X")
    elseif abs(b.s1 - b.s2) == Ny - 1
        return ("Z", "Z")
    else
        return nothing
    end
end



# -------- Three-body TRSB term dispatch -----------------------------------------------------------
"""
    wedge_operators(w::WedgeBond, x_odd::Bool, Ny::Int) -> NTuple{3,String}

The (op_at_s1, op_at_s2, op_at_s3) triple for the three-spin term contributed
by wedge `w`. Encodes the dispatch table in the comment above.
"""
function wedge_operators(w::WedgeBond, x_odd::Bool, Ny::Int)
    if abs(w.s1 - w.s3) == 1
        return x_odd ? ("Z", "Y", "X") : ("X", "Y", "Z")
    elseif x_odd
        return abs(w.s3 - w.s2) == Ny ? ("Y", "Z", "X") : ("Y", "X", "Z")
    else
        return abs(w.s2 - w.s1) == Ny ? ("X", "Z", "Y") : ("Z", "X", "Y")
    end
end



# -------- Geometry helpers ------------------------------------------------------------------------
"""
    cumulative_gauge(width_profile) -> Vector{Int}

Column gauge: returns g with g[x] < site ≤ g[x+1] ⇔ site is in column x.
g[1] = 0, g[end] = sum(width_profile).
"""
function cumulative_gauge(width_profile)
    g = Int[]
    for idx in 0:length(width_profile)
        push!(g, sum(width_profile[1:idx]))
    end
    return g
end




# -------- Set up the Hamiltonian on the interferometer as an MPO ----------------------------------
"""
    interferometer_energy_mpo(sites; Nx, Ny, width_profile, constrictions,
                              Jx=1.0, Jy=1.0, Jz=1.0, κ=0.0, α=1.0) -> MPO

Kitaev interferometer Hamiltonian (Pauli convention) on the OBC lattice:
    H = -Σ_bonds scale·Jᵅ σᵅσᵅ  +  κ Σ_wedges σᵃσᵇσᶜ
scale = α on the two constriction bonds, else 1. MUST match the DMRG run
that produced the target MPS.
"""
function interferometer_energy_mpo(sites; Nx::Integer, Ny::Integer,
        width_profile::AbstractVector{Int}, constrictions,
        Jx::Real=1.0, Jy::Real=1.0, Jz::Real=1.0, κ::Real=0.0, α::Real=4.0)

    Nsites        = length(sites)
    x_gauge  = cumulative_gauge(width_profile)
    pair(a,b) = a < b ? (a,b) : (b,a)
    constr   = Set(pair(c[1], c[2]) for c in constrictions)
    Jof      = Dict("X"=>Jx, "Y"=>Jy, "Z"=>Jz)
    os       = OpSum()

    
    # Build the two-body Kitaev terms by looping through the lattice bonds and dispatching on their geometry
    lattice = interferometry_lattice_obc(Nx, Ny, Nsites, width_profile)
    xbond, ybond, zbond = 0, 0, 0
    
    for b in lattice
        ops = bond_operator(b, x_gauge, Ny)
        ops === nothing && error("Unclassified bond ($(b.s1),$(b.s2))")
        scale = pair(b.s1, b.s2) ∈ constr ? α : 1.0
        os .+= -scale * Jof[ops[1]], ops[1], b.s1, ops[2], b.s2

        ops[1] == "X" && (xbond += 1)
        ops[1] == "Y" && (ybond += 1)  
        ops[1] == "Z" && (zbond += 1)
    end
    xbond + ybond + zbond == length(lattice) || error("two-body dispatch covered $xbond/$ybond/$zbond/$(length(lattice))")



    # Build the three-spin interaction terms by looping through the lattice wedges and dispatching on their geometry
    if abs(κ) > 1e-12
        wedge = interferometry_wedge(Nx, Ny, Nsites, width_profile)
        nw = 0
        for w in wedge
            o1,o2,o3 = wedge_operators(w, isodd(column_index(w.s2, x_gauge)), Ny)
            os .+= κ, o1, w.s1, o2, w.s2, o3, w.s3
            nw += 1
        end
        nw == length(wedge) || error("wedge dispatch covered $nw/$(length(wedge))")
    end

    return MPO(os, sites)
end



# -------- Set up the Hamiltonian on a cluster as an MPO -------------------------------------------
"""
    cluster_energy_mpo(sites; Nx, Ny, Jx = 1.0, Jy = 1.0, Jz = 1.0, κ = 0.0,
               yperiodic::Bool = true) -> MPO

Build the Kitaev honeycomb Hamiltonian as an MPO on the C-style site labelling:

    H = -Jx Σ Sxᵢ Sxⱼ  - Jy Σ Syᵢ Syⱼ  - Jz Σ Szᵢ Szⱼ
        + κ  Σ Sᵅᵢ Sᵝⱼ Sᵞₖ      (three-spin "wedge" terms)

Bond and wedge dispatch matches `src_evolution/kitaev_honeycomb.jl`, so the MPO
is consistent with the ground-state MPS stored in `data/kitaev_honeycomb_*.h5`.
Set `κ = 0.0` to skip the wedge construction.
"""
function cluster_energy_mpo(sites; Nx::Integer, Ny::Integer, 
                    Jx::Real = 1.0, Jy::Real = 1.0, Jz::Real = 1.0, 
                    κ::Real  = 0.0, yperiodic::Bool = true)

    @assert length(sites) == Nx * Ny "sites length $(length(sites)) ≠ Nx*Ny = $(Nx*Ny)"
    
    
    # Set up the Hamiltonian with two-body and three-spin terms as an MPO
    os = OpSum()
    

    # ── two-body Kitaev bonds ─────────────────────────────────────────────────
    bonds = honeycomb_lattice_Cstyle(Nx, Ny; yperiodic)
    xbond, ybond, zbond = 0, 0, 0


    for b in bonds
        xcoord = 2 * div(b.s1 - 1, 2 * Ny) + (iseven(b.s1) ? 2 : 1)

        if iseven(xcoord)
            os .+= -Jy, "Sy", b.s1, "Sy", b.s2
            # @show b.s1, b.s2, "Sy"
            ybond += 1
        elseif b.s2 - b.s1 == 1
            os .+= -Jx, "Sx", b.s1, "Sx", b.s2
            # @show b.s1, b.s2, "Sx"
            xbond += 1
        else
            os .+= -Jz, "Sz", b.s1, "Sz", b.s2
            # @show b.s1, b.s2, "Sz"
            zbond += 1
        end
    end
    xbond + ybond + zbond == length(bonds) || error("Setting up $(xbond + ybond + zbond) bonds instead of $(length(bonds)) inconsistent.")


    # ── three-spin (wedge) terms ─────────────────────────────────────────
    if abs(κ) > 1e-12
        wedge = honeycomb_Cstyle_wedge(Nx, Ny; yperiodic)
        count = 0
        
        for w in wedge 
            tmp = div(w.s2 - 1, 2 * Ny)
            x = 2 * tmp + mod(w.s2 - 1, 2) + 1

            if isodd(x)
                if w.s1 - w.s2 == 1 && w.s3 - w.s2 == 2 * Ny - 1
                    os .+= κ, "Sx", w.s1, "Sy", w.s2, "Sz", w.s3
                    # @show w.s1, w.s2, w.s3, "Sx", "Sy", "Sz"
                    count += 1
                end

                if w.s3 - w.s2 == 1 && w.s2 - w.s1 == 1
                    os .+= κ, "Sz", w.s1, "Sy", w.s2, "Sx", w.s3
                    # @show w.s1, w.s2, w.s3, "Sz", "Sy", "Sx"
                    count += 1
                end 

                if x != 1 && w.s2 - w.s1 == 2 * Ny - 1
                    if w.s3 - w.s2 == 1
                        os .+= κ, "Sy", w.s1, "Sz", w.s2, "Sx", w.s3
                        # @show w.s1, w.s2, w.s3, "Sy", "Sz", "Sx"
                        count += 1  
                    else
                        os .+= κ, "Sy", w.s1, "Sx", w.s2, "Sz", w.s3
                        # @show w.s1, w.s2, w.s3, "Sy", "Sx", "Sz"
                        count += 1  
                    end
                end
            else
                if w.s3 - w.s2 == w.s2 - w.s1 == 1
                    os .+= κ, "Sx", w.s1, "Sy", w.s2, "Sz", w.s3
                    # @show w.s1, w.s2, w.s3, "Sx", "Sy", "Sz"
                    count += 1
                end

                if w.s2 - w.s3 == 1 && w.s2 - w.s1 == 2 * Ny - 1
                    os .+= κ, "Sz", w.s1, "Sy", w.s2, "Sx", w.s3
                    # @show w.s1, w.s2, w.s3, "Sz", "Sy", "Sx"
                    count += 1
                end

                if x != Nx && w.s3 - w.s2 == 2 * Ny - 1
                    if w.s2 - w.s1 == 1
                        os .+= κ, "Sx", w.s1, "Sz", w.s2, "Sy", w.s3
                        # @show w.s1, w.s2, w.s3, "Sx", "Sz", "Sy"
                        count += 1
                    else
                        os .+= κ, "Sz", w.s1, "Sx", w.s2, "Sy", w.s3
                        # @show w.s1, w.s2, w.s3, "Sz", "Sx", "Sy"
                        count += 1
                    end
                end
            end
        end
        # @show count

        count == length(wedge) || error("Wedge dispatch covered $count of $(length(wedge)) wedges — bond classification is inconsistent.")
    end

    return MPO(os, sites)
end