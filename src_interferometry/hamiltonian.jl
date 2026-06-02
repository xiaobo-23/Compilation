# 6/1/2026
# Helper functions that set up the Hamiltonian for the interferometer. 
# Construct interactions including two-body Kitaev interactions and three-body TRSB terms based on the geometry of the interferometer lattice.

using ITensors
using ITensorMPS

include("interferometry_lattice.jl")



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