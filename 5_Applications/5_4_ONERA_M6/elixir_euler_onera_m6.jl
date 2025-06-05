using Trixi
using OrdinaryDiffEqSSPRK, OrdinaryDiffEqLowStorageRK
using LinearAlgebra: norm

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations3D(1.4)

# NOTE: True Mach = 0.8395

@inline function initial_condition(x, t, equations::CompressibleEulerEquations3D)
    # set the freestream flow parameters
    rho_freestream = 1.4

    # v_total = 0.84 = Mach

    # AoA = 3.06
    v1 = 0.8388023121403883
    v2 = 0.0448406193973588
    v3 = 0.0

    p_freestream = 1.0

    prim = SVector(rho_freestream, v1, v2, v3, p_freestream)
    return prim2cons(prim, equations)
end

bc_farfield = BoundaryConditionDirichlet(initial_condition)

# Ensure that rho and p are the same across symmetry line and allow only 
# tangential velocity
@inline function bc_symmetry(u_inner, normal_direction::AbstractVector, x, t,
                              surface_flux_function,
                              equations::CompressibleEulerEquations3D)

    norm_ = norm(normal_direction)
    normal = normal_direction / norm_

    # compute the primitive variables
    rho, v1, v2, v3, p = cons2prim(u_inner, equations)

    v_normal = normal[1] * v1 + normal[2] * v2 + normal[3] * v3

    u_mirror = prim2cons(SVector(rho,
                                v1 - 2 * v_normal * normal[1],
                                v2 - 2 * v_normal * normal[2],
                                v3 - 2 * v_normal * normal[3],
                                p), equations)

    flux = surface_flux_function(u_inner, u_mirror, normal, equations) * norm_

    return flux
end

polydeg = 2
basis = LobattoLegendreBasis(polydeg)

shock_indicator = IndicatorHennemannGassner(equations, basis,
                                            alpha_max = 0.5,
                                            alpha_min = 0.001,
                                            alpha_smooth = true, # true
                                            variable = density_pressure)

surface_flux = flux_lax_friedrichs
volume_flux = flux_ranocha

volume_integral = VolumeIntegralShockCapturingHG(shock_indicator;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)

# NOTE: Flux Differencing is required, shock capturing not (at least not for simply running the code)
volume_integral = VolumeIntegralFluxDifferencing(volume_flux)

solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
               volume_integral = volume_integral)

cd(@__DIR__)
mesh_path = "./"
mesh_file = mesh_path * "m6wing_sanitized.inp"

boundary_symbols = [:Symmetry,
                    :FarField,
                    :BottomWing,
                    :TopWing]

mesh = P4estMesh{3}(mesh_file, polydeg = polydeg, boundary_symbols = boundary_symbols)

boundary_conditions = Dict(:Symmetry => bc_symmetry, # Symmetry: bc_symmetry
                           :FarField => bc_farfield, # Farfield: bc_farfield
                           :BottomWing => boundary_condition_slip_wall, # Wing: bc_slip_wall
                           :TopWing => boundary_condition_slip_wall, # Wing: bc_slip_wall
                          )

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

#tspan = (0.0, 6.049)
#ode = semidiscretize(semi, tspan)

# TODO: Host restart file somewhere!
restart_file = "restart_t605_undamped.h5"

restart_filename = joinpath("./", restart_file)

tspan = (load_time(restart_filename), 6.05) # 6.05

ode = semidiscretize(semi, tspan, restart_filename)


# Callbacks
###############################################################################

summary_callback = SummaryCallback()

force_boundary_names = (:BottomWing, :TopWing)

aoa() = deg2rad(3.06)

rho_inf() = 1.4
u_inf(equations) = 0.84
# Area calculated from information given at https://www.grc.nasa.gov/www/wind/valid/m6wing/m6wing.html

#height = 1.1963
height = 1.0 # Mesh we use normalizes wing height to one

g_I = tan(deg2rad(30)) * height

#base = 0.8059
base = 0.8059 / 1.1963 # Mesh we use normalizes wing height to one

g_II = base - g_I
g_III = tan(deg2rad(15.8)) * height
A = height * (0.5 * (g_I + g_III) + g_II)

lift_coefficient = AnalysisSurfaceIntegral(force_boundary_names,
                                           LiftCoefficientPressure3D(aoa(), rho_inf(),
                                                                     u_inf(equations), A))

p_inf() = 1.0
pressure_coefficient = AnalysisSurfacePointwise(force_boundary_names,
                                                SurfacePressureCoefficient(p_inf(), rho_inf(),
                                                                        u_inf(equations), A))

analysis_interval = 100_000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     analysis_errors = Symbol[],
                                     analysis_integrals = (lift_coefficient,),
                                     #analysis_pointwise = (pressure_coefficient,)
                                     )

alive_callback = AliveCallback(alive_interval = 200)

save_sol_interval = analysis_interval

save_solution = SaveSolutionCallback(interval = save_sol_interval,
                                     save_initial_solution = false,
                                     save_final_solution = true,
                                     solution_variables = cons2prim,
                                     output_directory="./")

save_restart = SaveRestartCallback(interval = save_sol_interval,
                                   save_final_restart = true,
                                   output_directory="./")

## k = 2 ##

cd(@__DIR__)
base_path = "./"

stepsize_callback = StepsizeCallback(cfl = 10.0, interval = 2) # PERRK p3 15 standalone

dtRatios_complete_p3 = [ 
    0.309904923439026,
    0.277295976877213,
    0.250083755254746,
    0.228134118318558,
    0.20889208316803,
    0.185411275029182,
    0.160719511508942,
    0.138943578004837,
    0.111497408151627,
    0.0973129367828369,
    0.0799268364906311,
    0.0501513481140137,
    0.0280734300613403
                      ] ./ 0.309904923439026
Stages_complete_p3 = reverse(collect(range(3, 15)))

## 6.049 -> 6.05 ##

# Only Flux-Differencing #
cfl_interval = 2

stepsize_callback = StepsizeCallback(cfl = 10.0, interval = cfl_interval) # PER(R)K p3 3-15
#stepsize_callback = StepsizeCallback(cfl = 10.7, interval = cfl_interval) # PER(R)K p3 15
#stepsize_callback = StepsizeCallback(cfl = 2.7, interval = cfl_interval) # (R-)CKL43
#stepsize_callback = StepsizeCallback(cfl = 2.8, interval = cfl_interval) # (R-)RK33

callbacks = CallbackSet(summary_callback,
                        alive_callback,
                        analysis_callback,
                        #save_solution,
                        #save_restart,
                        stepsize_callback
                        )

# Run the simulation
###############################################################################

newton = Trixi.RelaxationSolverNewton(max_iterations = 5, root_tol = 1e-12)

## k = 2, p = 3 ##

#ode_alg = Trixi.PairedExplicitRK3Multi(Stages_complete_p3, base_path, dtRatios_complete_p3)
#ode_alg = Trixi.PairedExplicitRK3(15, base_path)

ode_alg = Trixi.PairedExplicitRelaxationRK3Multi(Stages_complete_p3, base_path, dtRatios_complete_p3;
                                                 relaxation_solver = newton)

#ode_alg = Trixi.PairedExplicitRelaxationRK3(15, base_path; relaxation_solver = newton)                                                 

#ode_alg = Trixi.RelaxationCKL43(; relaxation_solver = newton)
#ode_alg = Trixi.CKL43()

#ode_alg = Trixi.RelaxationRK33(; relaxation_solver = newton)
#ode_alg = Trixi.RK33()

sol = Trixi.solve(ode, ode_alg, dt = 42.0, 
                  save_everystep = false, callback = callbacks);

