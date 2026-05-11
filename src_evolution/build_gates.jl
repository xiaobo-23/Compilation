# 5/10/2026
# Build the two-site and three-site gates for time evolution using TEBD


# ---- Imports ----
using ITensors
using ITensorMPS
using HDF5
using MKL
using LinearAlgebra
using TimerOutputs



function build_two_body_gates(sites, groups, couplings, dt)
    gates = ITensor[]
    for g in groups
        J = couplings[g.ops]
        for (i, j) in g.bonds
            hij = -J * op(g.ops[1], sites[i]) * op(g.ops[2], sites[j])
            push!(gates, exp(-im * dt/2 * hij))
        end
    end
    return gates
end