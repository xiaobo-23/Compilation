# 6/1/2026
# Helper functions that set up the Hamiltonian for the interferometer. 
# Construct interactions including two-body Kitaev interactions and three-body TRSB terms based on the geometry of the interferometer lattice.


using ITensors
using ITensorMPS
using HDF5
using AppleAccelerate
using LinearAlgebra
using TimerOutputs



include("interferometry_lattice.jl")
include("interferometry_plaquettes.jl")
include("CustomObserver.jl")


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


"""
    wedge_operators(w::WedgeBond, x_odd::Bool, Ny::Int) -> NTuple{3,String}

The (op_at_s1, op_at_s2, op_at_s3) triple for the three-spin term contributed
by wedge `w`. Encodes the dispatch table in the comment above.
"""
function wedge_operators(w::WedgeBond, x_odd::Bool, Ny::Int)
    if abs(w.s1 - w.s3) == 1
        return x_odd ? ("Sy", "Sz", "Sx") : ("Sx", "Sz", "Sy")
    elseif x_odd
        return abs(w.s3 - w.s2) == Ny ? ("Sz", "Sy", "Sx") : ("Sz", "Sx", "Sy")
    else
        return abs(w.s2 - w.s1) == Ny ? ("Sx", "Sy", "Sz") : ("Sy", "Sx", "Sz")
    end
end