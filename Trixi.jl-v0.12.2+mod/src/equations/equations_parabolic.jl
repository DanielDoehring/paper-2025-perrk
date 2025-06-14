# specify transformation of conservative variables prior to taking gradients.
# specialize this function to compute gradients e.g., of primitive variables instead of conservative
gradient_variable_transformation(::AbstractEquationsParabolic) = cons2cons

# By default, the gradients are taken with respect to the conservative variables.
# this is reflected by the type parameter `GradientVariablesConservative` in the abstract
# type `AbstractEquationsParabolic{NDIMS, NVARS, GradientVariablesConservative}`.
struct GradientVariablesConservative end

# Linear scalar diffusion for use in linear scalar advection-diffusion problems
abstract type AbstractLaplaceDiffusion{NDIMS, NVARS} <:
              AbstractEquationsParabolic{NDIMS, NVARS, GradientVariablesConservative} end
include("laplace_diffusion_1d.jl")
include("laplace_diffusion_2d.jl")
include("laplace_diffusion_3d.jl")

include("laplace_diffusion_entropy_variables.jl")
include("laplace_diffusion_entropy_variables_1d.jl")
include("laplace_diffusion_entropy_variables_2d.jl")
include("laplace_diffusion_entropy_variables_3d.jl")

# Compressible Navier-Stokes equations
abstract type AbstractCompressibleNavierStokesDiffusion{NDIMS, NVARS, GradientVariables} <:
              AbstractEquationsParabolic{NDIMS, NVARS, GradientVariables} end
include("compressible_navier_stokes.jl")
include("compressible_navier_stokes_1d.jl")
include("compressible_navier_stokes_2d.jl")
include("compressible_navier_stokes_3d.jl")

# Resistive and visuous MHD
abstract type AbstractViscoResistiveMhdDiffusion{NDIMS, NVARS, GradientVariables} <:
              AbstractEquationsParabolic{NDIMS, NVARS, GradientVariables} end
include("visco_resistive_mhd.jl")
include("visco_resistive_mhd_2d.jl")
