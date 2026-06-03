# 6/3/2026
# Build the two-site and three-site gates used in TEBD
# Use second-order Trotter decompistion as the default

using ITensors
using ITensorMPS


function build_two_body_gates(sites, groups, couplings, dt)
    gates = ITensor[]
    for g in groups
        J = couplings[g.ops]
        for (i, j) in g.bonds
            hij = -J * op(g.ops[1], sites[i]) * op(g.ops[2], sites[j])
            push!(gates, exp(-im * dt/2 * hij))
        end
    end
    append!(gates, reverse(gates))
    
    return gates
end