# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# du .= zero(eltype(du)) doesn't scale when using multiple threads.
# See https://github.com/trixi-framework/Trixi.jl/pull/924 for a performance comparison.
function reset_du!(du, dg, cache, element_indices = eachelement(dg, cache))
    @threaded for element in element_indices
        du[.., element] .= zero(eltype(du))
    end

    return du
end

function volume_jacobian(element, mesh::TreeMesh, cache)
    return inv(cache.elements.inverse_jacobian[element])^ndims(mesh)
end

@inline function get_inverse_jacobian(inverse_jacobian, mesh::TreeMesh,
                                      indices...)
    element = last(indices)
    return inverse_jacobian[element]
end

# Indicators used for shock-capturing and AMR
include("indicators.jl")
include("indicators_1d.jl")
include("indicators_2d.jl")
include("indicators_3d.jl")

# Container data structures
include("containers.jl")

# Dimension-agnostic parallel setup
include("dg_parallel.jl")

# Helper structs for parabolic AMR
include("containers_viscous.jl")

# 1D DG implementation
include("dg_1d.jl")
include("dg_1d_parabolic.jl")

# 2D DG implementation
include("dg_2d.jl")
include("dg_2d_parallel.jl")
include("dg_2d_parabolic.jl")

# 3D DG implementation
include("dg_3d.jl")
include("dg_3d_parabolic.jl")

# Auxiliary functions that are specialized on this solver
# as well as specialized implementations used to improve performance
include("dg_2d_compressible_euler.jl")
include("dg_3d_compressible_euler.jl")

# Subcell limiters
include("subcell_limiters.jl")
include("subcell_limiters_2d.jl")
include("dg_2d_subcell_limiters.jl")
end # @muladd
