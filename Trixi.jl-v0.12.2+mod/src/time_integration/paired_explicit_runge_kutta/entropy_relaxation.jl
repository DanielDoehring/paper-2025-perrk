# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# NOTE: This could actually live in a more general location,
# as it is not PERK-specific.

@inline function int_w_dot_stage(stage, u_stage,
                                 mesh::Union{TreeMesh{1}, StructuredMesh{1}},
                                 equations, dg::DG, cache)
    @trixi_timeit timer() "Integrate w ⋅ k" begin
        # Calculate ∫(∂S/∂u ⋅ k)dΩ = ∫(w ⋅ k)dΩ
        integrate_via_indices(u_stage, mesh, equations, dg, cache,
                              stage) do u_stage, i, element, equations, dg, stage
            u_local = get_node_vars(u_stage, equations, dg, i, element)
            w_node = cons2entropy(u_local, equations)
            stage_node = get_node_vars(stage, equations, dg, i, element)
            dot(w_node, stage_node)
        end
    end
end

@inline function int_w_dot_stage(stage, u_stage,
                                 mesh::Union{TreeMesh{2}, StructuredMesh{2},
                                             UnstructuredMesh2D, P4estMesh{2},
                                             T8codeMesh{2}},
                                 equations, dg::DG, cache)
    @trixi_timeit timer() "Integrate w ⋅ k" begin
        # Calculate ∫(∂S/∂u ⋅ k)dΩ = ∫(w ⋅ k)dΩ
        integrate_via_indices(u_stage, mesh, equations, dg, cache,
                              stage) do u_stage, i, j, element, equations, dg, stage
            u_local = get_node_vars(u_stage, equations, dg, i, j, element)
            w_node = cons2entropy(u_local, equations)
            stage_node = get_node_vars(stage, equations, dg, i, j, element)
            dot(w_node, stage_node)
        end
    end
end

@inline function int_w_dot_stage(stage, u_stage,
                                 mesh::Union{TreeMesh{3}, StructuredMesh{3},
                                             P4estMesh{3}, T8codeMesh{3}},
                                 equations, dg::DG, cache)
    @trixi_timeit timer() "Integrate w ⋅ k" begin
        # Calculate ∫(∂S/∂u ⋅ k)dΩ = ∫(w ⋅ k)dΩ
        integrate_via_indices(u_stage, mesh, equations, dg, cache,
                              stage) do u_stage, i, j, k, element, equations, dg, stage
            u_local = get_node_vars(u_stage, equations, dg, i, j, k, element)
            w_node = cons2entropy(u_local, equations)
            stage_node = get_node_vars(stage, equations, dg, i, j, k, element)
            dot(w_node, stage_node)
        end
    end
end

@inline function entropy_difference(gamma, S_old, dS, u_gamma_dir, mesh,
                                    equations, dg::DG, cache)
    return integrate(entropy_math, u_gamma_dir, mesh, equations, dg, cache) -
           S_old - gamma * dS
end

abstract type RelaxationSolver end

struct RelaxationSolverBisection{RealT <: Real} <: RelaxationSolver
    # General parameters
    max_iterations::Int # Maximum number of bisection iterations
    root_tol::RealT     # Function-tolerance for the relaxation equation
    gamma_tol::RealT    # Absolute tolerance for the bracketing interval length
    # Method-specific parameters
    gamma_min::RealT    # Lower bound of the initial bracketing interval
    gamma_max::RealT    # Upper bound of the initial bracketing interval
end

function RelaxationSolverBisection(; max_iterations = 25,
                                   root_tol = 1e-15, gamma_tol = 1e-13,
                                   gamma_min = 0.1, gamma_max = 1.2)
    return RelaxationSolverBisection(max_iterations, root_tol, gamma_tol,
                                     gamma_min, gamma_max)
end

struct RelaxationSolverSecant{RealT <: Real} <: RelaxationSolver
    # General parameters
    max_iterations::Int # Maximum number of bisection iterations
    root_tol::RealT     # Function-tolerance for the relaxation equation
    gamma_tol::RealT    # Absolute tolerance for the bracketing interval length
    # Method-specific parameters
    gamma_min::RealT    # Lower bound of the initial bracketing interval
    gamma_max::RealT    # Upper bound of the initial bracketing interval 
end

function RelaxationSolverSecant(; max_iterations = 15,
                                root_tol = 1e-15, gamma_tol = 1e-13,
                                gamma_min = 0.1, gamma_max = 1.2)
    return RelaxationSolverSecant(max_iterations, root_tol, gamma_tol,
                                  gamma_min, gamma_max)
end
function Base.show(io::IO,
                   relaxation_solver::Union{RelaxationSolverBisection,
                                            RelaxationSolverSecant})
    if typeof(relaxation_solver) <: RelaxationSolverBisection
        solver_type = "RelaxationSolverBisection"
    elseif typeof(relaxation_solver) <: RelaxationSolverSecant
        solver_type = "RelaxationSolverSecant"
    end
    print(io, "$solver_type(max_iterations=", relaxation_solver.max_iterations,
          ", root_tol=", relaxation_solver.root_tol,
          ", gamma_tol=", relaxation_solver.gamma_tol,
          ", gamma_min=", relaxation_solver.gamma_min,
          ", gamma_max=", relaxation_solver.gamma_max, ")")
end
function Base.show(io::IO, ::MIME"text/plain",
                   relaxation_solver::Union{RelaxationSolverBisection,
                                            RelaxationSolverSecant})
    if get(io, :compact, false)
        show(io, relaxation_solver)
    else
        setup = [
            "max_iterations" => relaxation_solver.max_iterations,
            "root_tol" => relaxation_solver.root_tol,
            "gamma_tol" => relaxation_solver.gamma_tol,
            "gamma_min" => relaxation_solver.gamma_min,
            "gamma_max" => relaxation_solver.gamma_max
        ]
        if typeof(relaxation_solver) <: RelaxationSolverBisection
            solver_type = "RelaxationSolverBisection"
        elseif typeof(relaxation_solver) <: RelaxationSolverSecant
            solver_type = "RelaxationSolverSecant"
        end
        summary_box(io, solver_type, setup)
    end
end

struct RelaxationSolverNewton{RealT <: Real} <: RelaxationSolver
    # General parameters
    max_iterations::Int # Maximum number of Newton iterations
    root_tol::RealT     # Function-tolerance for the relaxation equation
    gamma_tol::RealT    # Absolute tolerance for the Newton update step size
    # Method-specific parameters
    # Minimum relaxation parameter. If the Newton iteration computes a value smaller than this, 
    # the relaxation parameter is set to 1.
    gamma_min::RealT
    step_scaling::RealT # Scaling factor for the Newton step
end
function RelaxationSolverNewton(; max_iterations = 5,
                                root_tol = 1e-15, gamma_tol = 1e-13,
                                gamma_min = 1e-13, step_scaling = 1.0)
    return RelaxationSolverNewton(max_iterations, root_tol, gamma_tol,
                                  gamma_min, step_scaling)
end

function Base.show(io::IO,
                   relaxation_solver::RelaxationSolverNewton)
    print(io, "RelaxationSolverNewton(max_iterations=",
          relaxation_solver.max_iterations,
          ", root_tol=", relaxation_solver.root_tol,
          ", gamma_tol=", relaxation_solver.gamma_tol,
          ", gamma_min=", relaxation_solver.gamma_min,
          ", step_scaling=", relaxation_solver.step_scaling, ")")
end

function Base.show(io::IO, ::MIME"text/plain",
                   relaxation_solver::RelaxationSolverNewton)
    if get(io, :compact, false)
        show(io, relaxation_solver)
    else
        setup = [
            "max_iterations" => relaxation_solver.max_iterations,
            "root_tol" => relaxation_solver.root_tol,
            "gamma_tol" => relaxation_solver.gamma_tol,
            "gamma_min" => relaxation_solver.gamma_min,
            "step_scaling" => relaxation_solver.step_scaling
        ]
        summary_box(io, "RelaxationSolverNewton", setup)
    end
end

function relaxation_solver!(integrator,
                            u_tmp_wrap, u_wrap, dir_wrap,
                            S_old, dS,
                            mesh, equations, dg::DG, cache,
                            relaxation_solver::RelaxationSolverBisection)
    @unpack root_tol = relaxation_solver

    @threaded for element in eachelement(dg, cache)
        @views @. u_tmp_wrap[.., element] = u_wrap[.., element] + # gamma = 1
                                            dir_wrap[.., element]
    end
    @trixi_timeit timer() "Δη" r_1=entropy_difference(1, S_old, dS,
                                                      u_tmp_wrap, mesh,
                                                      equations, dg, cache)
    if abs(r_1) <= root_tol
        integrator.gamma = 1
        return nothing
    end

    @unpack max_iterations, gamma_tol, gamma_min, gamma_max = relaxation_solver

    @threaded for element in eachelement(dg, cache)
        @views @. u_tmp_wrap[.., element] = u_wrap[.., element] +
                                            gamma_max * dir_wrap[.., element]
    end
    @trixi_timeit timer() "Δη" r_max=entropy_difference(gamma_max, S_old, dS,
                                                        u_tmp_wrap, mesh,
                                                        equations, dg, cache)

    @threaded for element in eachelement(dg, cache)
        @views @. u_tmp_wrap[.., element] = u_wrap[.., element] +
                                            gamma_min * dir_wrap[.., element]
    end
    @trixi_timeit timer() "Δη" r_min=entropy_difference(gamma_min, S_old, dS,
                                                        u_tmp_wrap, mesh,
                                                        equations, dg, cache)

    # Check if there exists a root for `r` in the interval [gamma_min, gamma_max]
    if r_max > 0 && r_min < 0
        iterations = 0
        while gamma_max - gamma_min > gamma_tol && iterations < max_iterations
            integrator.gamma = (gamma_max + gamma_min) / 2

            @threaded for element in eachelement(dg, cache)
                @views @. u_tmp_wrap[.., element] = u_wrap[.., element] +
                                                    integrator.gamma *
                                                    dir_wrap[.., element]
            end
            @trixi_timeit timer() "Δη" r_gamma=entropy_difference(integrator.gamma,
                                                                  S_old, dS,
                                                                  u_tmp_wrap, mesh,
                                                                  equations, dg, cache)

            if r_gamma < 0
                gamma_min = integrator.gamma
            else
                gamma_max = integrator.gamma
            end
            iterations += 1
        end
    else # No proper bracketing interval found
        integrator.gamma = 1
        # CARE: This is an experimental strategy: 
        # Set gamma to smallest value s.t. convergence is still assured
        #integrator.gamma = 1 - integrator.dt^(ORDER - 1)
    end

    # TODO: Can we store `S_old` by using `r_gamma` here for the next timestep?

    return nothing
end

function relaxation_solver!(integrator,
                            u_tmp_wrap, u_wrap, dir_wrap,
                            S_old, dS,
                            mesh, equations, dg::DG, cache,
                            relaxation_solver::RelaxationSolverSecant)
    @unpack root_tol = relaxation_solver

    @threaded for element in eachelement(dg, cache)
        @views @. u_tmp_wrap[.., element] = u_wrap[.., element] + # gamma = 1
                                            dir_wrap[.., element]
    end
    @trixi_timeit timer() "Δη" r_1=entropy_difference(1, S_old, dS,
                                                      u_tmp_wrap, mesh,
                                                      equations, dg, cache)
    if abs(r_1) <= root_tol
        integrator.gamma = 1
        return nothing
    end

    @unpack max_iterations, gamma_tol, gamma_min, gamma_max = relaxation_solver

    # Naming aliases to avoid confusion
    gamma_0, gamma_1 = gamma_min, gamma_max

    @threaded for element in eachelement(dg, cache)
        @views @. u_tmp_wrap[.., element] = u_wrap[.., element] +
                                            gamma_1 * dir_wrap[.., element]
    end
    @trixi_timeit timer() "Δη" r_1=entropy_difference(gamma_1, S_old, dS,
                                                      u_tmp_wrap, mesh,
                                                      equations, dg, cache)

    @threaded for element in eachelement(dg, cache)
        @views @. u_tmp_wrap[.., element] = u_wrap[.., element] +
                                            gamma_0 * dir_wrap[.., element]
    end
    @trixi_timeit timer() "Δη" r_0=entropy_difference(gamma_0, S_old, dS,
                                                      u_tmp_wrap, mesh,
                                                      equations, dg, cache)

    # Check if there exists a root for `r` in the interval [gamma_0, gamma_1] = [gamma_min, gamma_max]
    if r_1 > 0 && r_0 < 0
        # Perform first step which does not require extra evaluation of the `entropy_difference` function
        # We consider `gamma_1 = gamma_max` as the better initial guess, as this is for the default values most likely closer to the root
        gamma_0 = gamma_1 - r_1 * (gamma_1 - gamma_0) / (r_1 - r_0)
        # Switch order of 0, 1:
        gamma_0, gamma_1 = gamma_1, gamma_0

        iterations = 1
        integrator.gamma = gamma_1 # Write result back to integrator
        while abs(gamma_1 - gamma_0) > gamma_tol && iterations < max_iterations
            @threaded for element in eachelement(dg, cache)
                @views @. u_tmp_wrap[.., element] = u_wrap[.., element] +
                                                    gamma_1 * dir_wrap[.., element]
            end
            r_0 = r_1
            @trixi_timeit timer() "Δη" r_1=entropy_difference(gamma_1, S_old, dS,
                                                              u_tmp_wrap, mesh,
                                                              equations, dg, cache)

            gamma_0 = gamma_1 - r_1 * (gamma_1 - gamma_0) / (r_1 - r_0)
            # Switch order of 0, 1:
            gamma_0, gamma_1 = gamma_1, gamma_0

            iterations += 1
            integrator.gamma = gamma_1 # Write result back to integrator

            # Catch failure
            if integrator.gamma < gamma_min
                break
                integrator.gamma = 1
            end
        end
    else # No proper bracketing interval found
        integrator.gamma = 1
        # CARE: This is an experimental strategy: 
        # Set gamma to smallest value s.t. convergence is still assured
        #integrator.gamma = 1 - integrator.dt^(ORDER - 1)
    end

    # TODO: Can we store `S_old` by using `r_1` here for the next timestep?

    return nothing
end

#=
function relaxation_solver!(integrator::Union{AbstractPairedExplicitRelaxationRKIntegrator{ORDER},
                                              AbstractPairedExplicitRelaxationRKMultiParabolicIntegrator{ORDER}},
                            u_tmp_wrap, u_wrap, dir_wrap,
                            S_old, dS,
                            mesh, equations, dg::DG, cache,
                            relaxation_solver::RelaxationSolverNewton) where {ORDER}
=#
function relaxation_solver!(integrator,
                            u_tmp_wrap, u_wrap, dir_wrap,
                            S_old, dS,
                            mesh, equations, dg::DG, cache,
                            relaxation_solver::RelaxationSolverNewton)
    @unpack max_iterations, root_tol, gamma_tol, gamma_min, step_scaling = relaxation_solver

    iterations = 0
    while iterations < max_iterations
        @threaded for element in eachelement(dg, cache)
            @views @. u_tmp_wrap[.., element] = u_wrap[.., element] +
                                                integrator.gamma *
                                                dir_wrap[.., element]
        end
        @trixi_timeit timer() "Δη" r_gamma=entropy_difference(integrator.gamma,
                                                              S_old, dS,
                                                              u_tmp_wrap, mesh,
                                                              equations, dg, cache)

        if abs(r_gamma) <= root_tol
            # TODO: Can we store `S_old` by using `r_gamma` here for the next timestep?
            break
        end

        dr = int_w_dot_stage(dir_wrap, u_tmp_wrap, mesh, equations, dg, cache) - dS

        step = step_scaling * r_gamma / dr
        if abs(step) <= gamma_tol
            break
        end

        integrator.gamma -= step
        iterations += 1
    end

    # Catch Newton failures
    if integrator.gamma < gamma_min || isnan(integrator.gamma) ||
       isinf(integrator.gamma)
        integrator.gamma = 1
        # CARE: This is an experimental strategy: 
        # Set gamma to smallest value s.t. convergence is still assured
        #integrator.gamma = 1 - integrator.dt^(ORDER - 1)
    end

    return nothing
end

@inline function update_t_relaxation!(integrator)
    # Check if due to entropy relaxation the final step is not reached
    if integrator.finalstep == true && integrator.gamma != 1
        # If we would go beyond the final time, clip gamma at 1.0
        #if integrator.gamma > 1.0
        integrator.gamma = 1.0
        #else # If we are below the final time, reset finalstep flag
        #    integrator.finalstep = false
        #end
    end
    integrator.t += integrator.gamma * integrator.dt

    # Write t and gamma to file for plotting
    #=
    open("relaxation_log.txt", "a") do file
        write(file, "$(integrator.t) $(integrator.gamma)\n")
    end
    =#
    return nothing
end
end # @muladd
