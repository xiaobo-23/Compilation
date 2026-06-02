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



"""
    interferometer_energy_mpo(sites; Nx, Ny, width_profile, constrictions,
                              Jx=1.0, Jy=1.0, Jz=1.0, κ=0.0, α=1.0) -> MPO

Kitaev interferometer Hamiltonian (Pauli convention) on the OBC lattice:
    H = -Σ_bonds scale·Jᵅ σᵅσᵅ  +  κ Σ_wedges σᵃσᵇσᶜ
scale = α on the two constriction bonds, else 1. MUST match the DMRG run
that produced the target MPS.
"""
function interferometry_energy_mpo(sites; Nx::Integer, Ny::Integer,
        width_profile::AbstractVector{Int}, constrictions,
        Jx::Real=1.0, Jy::Real=1.0, Jz::Real=1.0, κ::Real=0.0, α::Real=1.0)

    N        = length(sites)
    x_gauge  = cumulative_gauge(width_profile)
    pair(a,b) = a < b ? (a,b) : (b,a)
    constr   = Set(pair(c[1], c[2]) for c in constrictions)
    Jof      = Dict("X"=>Jx, "Y"=>Jy, "Z"=>Jz)
    os       = OpSum()

    lattice = interferometer_lattice_obc(Nx, Ny, N, width_profile)
    nb = 0
    for b in lattice
        ops = bond_operator(b, x_gauge, Ny)
        ops === nothing && error("Unclassified bond ($(b.s1),$(b.s2))")
        scale = pair(b.s1, b.s2) ∈ constr ? α : 1.0
        os .+= -scale * Jof[ops[1]], ops[1], b.s1, ops[2], b.s2
        nb += 1
    end
    nb == length(lattice) || error("two-body dispatch covered $nb/$(length(lattice))")

    if abs(κ) > 1e-12
        wedge = interferometry_wedge(Nx, Ny, N, width_profile)
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