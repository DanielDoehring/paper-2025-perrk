using OrdinaryDiffEqSSPRK, OrdinaryDiffEqLowStorageRK
using Trixi

# This is the classic 1D viscous shock wave problem with analytical solution
# for a special value of the Prandtl number.
# The original references are:
#
# - R. Becker (1922)
#   Stoßwelle und Detonation.
#   [DOI: 10.1007/BF01329605](https://doi.org/10.1007/BF01329605)
#
#   English translations:
#   Impact waves and detonation. Part I.
#   https://ntrs.nasa.gov/api/citations/19930090862/downloads/19930090862.pdf
#   Impact waves and detonation. Part II.
#   https://ntrs.nasa.gov/api/citations/19930090863/downloads/19930090863.pdf
#
# - M. Morduchow, P. A. Libby (1949)
#   On a Complete Solution of the One-Dimensional Flow Equations
#   of a Viscous, Head-Conducting, Compressible Gas
#   [DOI: 10.2514/8.11882](https://doi.org/10.2514/8.11882)
#
#
# The particular problem considered here is described in
# - L. G. Margolin, J. M. Reisner, P. M. Jordan (2017)
#   Entropy in self-similar shock profiles
#   [DOI: 10.1016/j.ijnonlinmec.2017.07.003](https://doi.org/10.1016/j.ijnonlinmec.2017.07.003)

### Fixed parameters ###

# Special value for which nonlinear solver can be omitted
# Corresponds essentially to fixing the Mach number
alpha = 0.5
# We want kappa = cp * mu = mu_bar to ensure constant enthalpy
prandtl_number() = 1

### Free choices: ###
gamma() = 5 / 3

mu() = 0.15
mu_bar() = mu() / (gamma() - 1) # Re-scaled viscosity

rho_0() = 1
v() = 1 # Shock speed

### Derived quantities ###

Ma() = 2 / sqrt(3 - gamma()) # Mach number for alpha = 0.5
c_0() = v() / Ma() # Speed of sound ahead of the shock

# From constant enthalpy condition
p_0() = c_0()^2 * rho_0() / gamma()

l() = mu_bar() / (rho_0() * v()) * 2 * gamma() / (gamma() + 1) # Appropriate length scale

"""
    initial_condition_viscous_shock(x, t, equations)

Classic 1D viscous shock wave problem with analytical solution
for a special value of the Prandtl number.
The version implemented here is described in
- L. G. Margolin, J. M. Reisner, P. M. Jordan (2017)
  Entropy in self-similar shock profiles
  [DOI: 10.1016/j.ijnonlinmec.2017.07.003](https://doi.org/10.1016/j.ijnonlinmec.2017.07.003)
"""
function initial_condition_viscous_shock(x, t, equations)
    y = x[1] - v() * t # Translated coordinate

    # Coordinate transformation. See eq. (33) in Margolin et al. (2017)
    chi = 2 * exp(y / (2 * l()))

    w = 1 + 1 / (2 * chi^2) * (1 - sqrt(1 + 2 * chi^2))

    rho = rho_0() / w
    u = v() * (1 - w)
    p = p_0() * 1 / w * (1 + (gamma() - 1) / 2 * Ma()^2 * (1 - w^2))

    return prim2cons(SVector(rho, u, p), equations)
end
initial_condition = initial_condition_viscous_shock

###############################################################################
# semidiscretization of the ideal compressible Navier-Stokes equations

equations = CompressibleEulerEquations1D(gamma())
equations_parabolic = CompressibleNavierStokesDiffusion1D(equations, mu = mu_bar(),
                                                          Prandtl = prandtl_number(),
                                                          gradient_variables = GradientVariablesPrimitive())

PolyDeg = 1
solver = DGSEM(polydeg = PolyDeg, surface_flux = flux_hlle)

domain_length = 4.0
coordinates_min = -domain_length / 2
coordinates_max = domain_length / 2

refinement_patches = ((type = "box", coordinates_min = (-1.0,),
                       coordinates_max = (1.0,)),)

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4, # 2 to 7
                periodicity = false,
                refinement_patches = refinement_patches,
                n_cells_max = 30_000)

### Inviscid boundary conditions ###

# Prescribe pure influx based on initial conditions
function boundary_condition_inflow(u_inner, orientation::Integer, normal_direction, x, t,
                                   surface_flux_function,
                                   equations::CompressibleEulerEquations1D)
    u_cons = initial_condition_viscous_shock(x, t, equations)
    flux = Trixi.flux(u_cons, orientation, equations)

    return flux
end

# Completely free outflow
function boundary_condition_outflow(u_inner, orientation::Integer, normal_direction, x, t,
                                    surface_flux_function,
                                    equations::CompressibleEulerEquations1D)
    # Calculate the boundary flux entirely from the internal solution state
    flux = Trixi.flux(u_inner, orientation, equations)

    return flux
end

boundary_conditions = (; x_neg = boundary_condition_inflow,
                       x_pos = boundary_condition_outflow)

### Viscous boundary conditions ###

# For the viscous BCs, we use the known analytical solution
velocity_bc = NoSlip() do x, t, equations_parabolic
    Trixi.velocity(initial_condition_viscous_shock(x,
                                                   t,
                                                   equations_parabolic),
                   equations_parabolic)
end

heat_bc = Isothermal() do x, t, equations_parabolic
    Trixi.temperature(initial_condition_viscous_shock(x,
                                                      t,
                                                      equations_parabolic),
                      equations_parabolic)
end

boundary_condition_parabolic = BoundaryConditionNavierStokesWall(velocity_bc, heat_bc)

boundary_conditions_parabolic = (; x_neg = boundary_condition_parabolic,
                                 x_pos = boundary_condition_parabolic)

# We use by default the Bassi-Rebay 1 scheme.
# Since this is a diffusion-dominated problem, using the LDG scheme should achieve optimal rates of convergence. 
# In contrast, BR-1 may achieve suboptimal rates of convergence in diffusion-dominated regimes. 
# The LDG scheme can be used by specifying the keyword
# solver_parabolic = ViscousFormulationLocalDG()
# in the semidiscretization call below.
semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver;
                                             boundary_conditions = (boundary_conditions,
                                                                    boundary_conditions_parabolic))

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span `tspan`
tspan = (0.0, 0.5)
#tspan = (0.0, 0.0) # For plotting of IC

ode = semidiscretize(semi, tspan; split_problem = false)

summary_callback = SummaryCallback()

alive_callback = AliveCallback(alive_interval = 10000)

analysis_interval = 1_000_000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     analysis_errors = [:l2_error, :l1_error, :linf_error],
                                         #:l2_error_primitive, :l1_error_primitive, :linf_error_primitive],
                                     analysis_integrals = (;))

callbacks = CallbackSet(summary_callback, alive_callback, analysis_callback)

###############################################################################
# run the simulation

# For diffusion-dominated case we need four times the timestep between the methods
dtRatios = [1, 0.25]
cd(@__DIR__)
basepath = "./"

relaxation_solver = Trixi.RelaxationSolverBisection(gamma_min = 0.8,
                                                    root_tol = 1e-15,
                                                    gamma_tol = 1e-15)


path = basepath * "p2/"
Stages = [8, 4]

#ode_algorithm = Trixi.PairedExplicitRK2Multi(Stages, path, dtRatios)
ode_algorithm = Trixi.PairedExplicitRelaxationRK2Multi(Stages, path, dtRatios, relaxation_solver = relaxation_solver)


#=
path = basepath * "p3/"
Stages = [9, 5]

#ode_algorithm = Trixi.PairedExplicitRK3Multi(Stages, path, dtRatios)
ode_algorithm = Trixi.PairedExplicitRelaxationRK3Multi(Stages, path, dtRatios, relaxation_solver = relaxation_solver)
=#

#=
path = basepath * "p4/"
Stages = [10, 6]

#ode_algorithm = Trixi.PairedExplicitRK4Multi(Stages, path, dtRatios)
ode_algorithm = Trixi.PairedExplicitRelaxationRK4Multi(Stages, path, dtRatios, relaxation_solver = relaxation_solver)
=#

dtRef = 8.75e-3
max_level = Trixi.maximum_level(mesh.tree)
dt = dtRef / 4.0^(max_level - 2)

sol = Trixi.solve(ode, ode_algorithm,
                  dt = dt,
                  save_everystep = false, callback = callbacks,
                  maxiters = typemax(Int));

###############################################################################

# Plot IC & grid
#=
using Plots

pd = PlotData1D(sol)

plot(getmesh(pd), label = "")

plot!(pd["rho"], xlabel = "\$x\$",
      label = "\$\\rho\$",
      linewidth = 3, color = RGB(0, 84 / 256, 159 / 256),
      guidefont = font("Computer Modern", 16), tickfont = font("Computer Modern", 14),
      legend = true)

plot!(pd["v1"],
      label = "\$u\$",
      linewidth = 3, color = RGB(246 / 256, 169 / 256, 0), legend = true)

plot!(pd["p"], title = "Viscous Shock: Initial Solution",
      label = "\$p\$",
      linewidth = 3, color = RGB(70 / 256, 171 / 256, 39 / 256),
      titlefont = font("Computer Modern", 18),
      legendfont = font("Computer Modern", 16),
      legend = :topright)
=#