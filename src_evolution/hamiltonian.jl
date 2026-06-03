# 6/3/2026
# Set up the Kitaev Hamiltonnian as an MPO using the OpSum interface.
# The MPO can be used to compute the energy, variance in TEBD or set up the Hamiltonian used in TDVP time evolution


using ITensors
using ITensorMPS
include("honeycomb_lattice.jl")


"""
    hamiltonian_cluster(sites; Nx, Ny, Jx=1.0, Jy=1.0, Jz=1.0, yperiodic=true) -> MPO

Kitaev honeycomb cluster Hamiltonian (two-body terms only), C-style ordering:

    H = -Jx Σ SˣᵢSˣⱼ  - Jy Σ SʸᵢSʸⱼ  - Jz Σ SᶻᵢSᶻⱼ

Bond-flavor dispatch. This must match both the TEBD gates and the Hamiltonian that
produced the loaded ground state:
    even-x column               → y-bond
    odd-x column, s₂ - s₁ == 1  → x-bond
    odd-x column, otherwise     → z-bond
"""
function hamiltonian_cluster(sites; Nx::Int, Ny::Int,
        Jx::Real=1.0, Jy::Real=1.0, Jz::Real=1.0, yperiodic::Bool=true)

    @assert length(sites) == Nx * Ny "length(sites)=$(length(sites)) ≠ Nx*Ny=$(Nx*Ny)"

    lattice = honeycomb_lattice_Cstyle(Nx, Ny; yperiodic)
    os = OpSum()
    xbond, ybond, zbond = 0, 0, 0

    for b in lattice
        xcoord = 2 * div(b.s1 - 1, 2 * Ny) + (iseven(b.s1) ? 2 : 1)
        # ycoord = div(mod(b.s1 - 1, 2 * Ny), 2) + 1   # unused; kept for reference

        if iseven(xcoord)
            os .+= -Jy, "Y", b.s1, "Y", b.s2
            ybond += 1
            # @show b.s1, b.s2, "Y"
        elseif b.s2 - b.s1 == 1
            os .+= -Jx, "X", b.s1, "X", b.s2
            xbond += 1
            # @show b.s1, b.s2, "X"
        else
            os .+= -Jz, "Z", b.s1, "Z", b.s2
            zbond += 1
            # @show b.s1, b.s2, "Z"
        end
    end

    xbond + ybond + zbond == length(lattice) ||
        error("dispatch covered $(xbond+ybond+zbond) of $(length(lattice)) bonds")

    return MPO(os, sites)
end


measure_energy(ψ::MPS, H::MPO) = real(inner(ψ', H, ψ))



  # # Set up the three-spin interaction terms in the Hamiltonian
  # count = 0
  # for w in wedge 
  #   # Calculate the (x, y) coordinates of the site n based on C-style ordering
  #   tmp = div(w.s2 - 1, 2 * Ny)
  #   x = 2 * tmp + mod(w.s2 - 1, 2) + 1
  #   y = mod(div(w.s2 - 1, 2), Ny) + 1

  #   if mod(x, 2) == 1
  #     if w.s1 - w.s2 == 1 && w.s3 - w.s2 == 2 * Ny - 1
  #       os .+= κ, "Sx", w.s1, "Sy", w.s2, "Sz", w.s3
  #       @show w.s1, w.s2, w.s3, "Sx", "Sy", "Sz"
  #       count += 1
  #     end

  #     if w.s3 - w.s2 == 1 && w.s2 - w.s1 == 1
  #       os .+= κ, "Sz", w.s1, "Sy", w.s2, "Sx", w.s3
  #       @show w.s1, w.s2, w.s3, "Sz", "Sy", "Sx"
  #       count += 1
  #     end 

  #     if x != 1 && w.s2 - w.s1 == 2 * Ny - 1
  #       if w.s3 - w.s2 == 1
  #         os .+= κ, "Sy", w.s1, "Sz", w.s2, "Sx", w.s3
  #         @show w.s1, w.s2, w.s3, "Sy", "Sz", "Sx"
  #         count += 1  
  #       else
  #         os .+= κ, "Sy", w.s1, "Sx", w.s2, "Sz", w.s3
  #         @show w.s1, w.s2, w.s3, "Sy", "Sx", "Sz"
  #         count += 1  
  #       end
  #     end
  #   else
  #     if w.s3 - w.s2 == w.s2 - w.s1 == 1
  #       os .+= κ, "Sx", w.s1, "Sy", w.s2, "Sz", w.s3
  #       @show w.s1, w.s2, w.s3, "Sx", "Sy", "Sz"
  #       count += 1
  #     end

  #     if w.s2 - w.s3 == 1 && w.s2 - w.s1 == 2 * Ny - 1
  #       os .+= κ, "Sz", w.s1, "Sy", w.s2, "Sx", w.s3
  #       @show w.s1, w.s2, w.s3, "Sz", "Sy", "Sx"
  #       count += 1
  #     end

  #     if x != Nx && w.s3 - w.s2 == 2 * Ny - 1
  #       if w.s2 - w.s1 == 1
  #         os .+= κ, "Sx", w.s1, "Sz", w.s2, "Sy", w.s3
  #         @show w.s1, w.s2, w.s3, "Sx", "Sz", "Sy"
  #         count += 1
  #       else
  #         os .+= κ, "Sz", w.s1, "Sx", w.s2, "Sy", w.s3
  #         @show w.s1, w.s2, w.s3, "Sz", "Sx", "Sy"
  #         count += 1
  #       end
  #     end
  #   end
  # end
  # @show count 

  # if count != length(wedge)
  #   error("The number of three-spin interaction terms generated does not match the expected number.")
  # end