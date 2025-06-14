# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

abstract type vanderHouwenAlgorithm end
abstract type vanderHouwenRelaxationAlgorithm end

"""
    CKL43()

Carpenter-Kennedy-Lewis 4-stage, 3rd-order Runge-Kutta method.
"""
struct CKL43 <: vanderHouwenAlgorithm
    a::SVector{4, Float64}
    b::SVector{4, Float64}
    c::SVector{4, Float64}
end
function CKL43()
    a = SVector(0.0,
                11847461282814 / 36547543011857,
                3943225443063 / 7078155732230,
                -346793006927 / 4029903576067)

    b = SVector(1017324711453 / 9774461848756,
                8237718856693 / 13685301971492,
                57731312506979 / 19404895981398,
                -101169746363290 / 37734290219643)
    c = SVector(0.0,
                a[2],
                b[1] + a[3],
                b[1] + b[2] + a[4])

    return CKL43(a, b, c)
end

struct RelaxationCKL43{RelaxationSolver} <: vanderHouwenRelaxationAlgorithm
    van_der_houwen_alg::CKL43
    relaxation_solver::RelaxationSolver
end
function RelaxationCKL43(; relaxation_solver = RelaxationSolverNewton())
    return RelaxationCKL43{typeof(relaxation_solver)}(CKL43(), relaxation_solver)
end

"""
    CKL54()

Carpenter-Kennedy-Lewis 5-stage, 4th-order Runge-Kutta method.
"""
struct CKL54 <: vanderHouwenAlgorithm
    a::SVector{5, Float64}
    b::SVector{5, Float64}
    c::SVector{5, Float64}
end
function CKL54()
    a = SVector(0.0,
                970286171893 / 4311952581923,
                6584761158862 / 12103376702013,
                2251764453980 / 15575788980749,
                26877169314380 / 34165994151039)

    b = SVector(1153189308089 / 22510343858157,
                1772645290293 / 4653164025191,
                -1672844663538 / 4480602732383,
                2114624349019 / 3568978502595,
                5198255086312 / 14908931495163)
    c = SVector(0.0,
                a[2],
                b[1] + a[3],
                b[1] + b[2] + a[4],
                b[1] + b[2] + b[3] + a[5])

    return CKL54(a, b, c)
end

struct RelaxationCKL54{RelaxationSolver} <: vanderHouwenRelaxationAlgorithm
    van_der_houwen_alg::CKL54
    relaxation_solver::RelaxationSolver
end
function RelaxationCKL54(; relaxation_solver = RelaxationSolverNewton())
    return RelaxationCKL54{typeof(relaxation_solver)}(CKL54(), relaxation_solver)
end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.jl.
mutable struct vanderHouwenIntegrator{RealT <: Real, uType, Params, Sol, F, Alg,
                                      SimpleIntegrator2NOptions} <:
               AbstractTimeIntegrator
    u::uType
    du::uType
    u_tmp::uType
    t::RealT
    dt::RealT # current time step
    dtcache::RealT # ignored
    iter::Int # current number of time steps (iteration)
    p::Params # will be the semidiscretization from Trixi.jl
    sol::Sol # faked
    f::F # `rhs` of the semidiscretization
    alg::Alg
    opts::SimpleIntegrator2NOptions
    finalstep::Bool # added for convenience
    # Addition for efficient implementation
    k_prev::uType
end

mutable struct vanderHouwenRelaxationIntegrator{RealT <: Real, uType, Params, Sol, F,
                                                Alg,
                                                SimpleIntegrator2NOptions,
                                                RelaxationSolver} <:
               AbstractTimeIntegrator
    u::uType
    du::uType
    u_tmp::uType
    t::RealT
    dt::RealT # current time step
    dtcache::RealT # ignored
    iter::Int # current number of time steps (iteration)
    p::Params # will be the semidiscretization from Trixi.jl
    sol::Sol # faked
    f::F # `rhs` of the semidiscretization
    alg::Alg
    opts::SimpleIntegrator2NOptions
    finalstep::Bool # added for convenience
    # Addition for efficient implementation
    k_prev::uType
    # For entropy relaxation
    direction::uType
    gamma::RealT
    relaxation_solver::RelaxationSolver
end

# Forward integrator.stats.naccept to integrator.iter (see GitHub PR#771)
function Base.getproperty(integrator::Union{vanderHouwenIntegrator,
                                            vanderHouwenRelaxationIntegrator},
                          field::Symbol)
    if field === :stats
        return (naccept = getfield(integrator, :iter),)
    end
    # general fallback
    return getfield(integrator, field)
end

function init(ode::ODEProblem, alg::vanderHouwenAlgorithm;
              dt, callback::Union{CallbackSet, Nothing} = nothing, kwargs...)
    u = copy(ode.u0)
    du = similar(u)
    u_tmp = copy(u)
    k_prev = similar(u)

    t = first(ode.tspan)
    iter = 0

    integrator = vanderHouwenIntegrator(u, du, u_tmp, t, dt, zero(dt), iter, ode.p,
                                        (prob = ode,), ode.f, alg,
                                        SimpleIntegrator2NOptions(callback, ode.tspan;
                                                                  kwargs...), false,
                                        k_prev)

    # initialize callbacks
    if callback isa CallbackSet
        foreach(callback.continuous_callbacks) do cb
            throw(ArgumentError("Continuous callbacks are unsupported with van-der-Houwen time integration methods."))
        end
        foreach(callback.discrete_callbacks) do cb
            cb.initialize(cb, integrator.u, integrator.t, integrator)
        end
    end

    return integrator
end

function init(ode::ODEProblem, alg::vanderHouwenRelaxationAlgorithm;
              dt, callback::Union{CallbackSet, Nothing} = nothing, kwargs...)
    u = copy(ode.u0)
    du = similar(u)
    u_tmp = copy(u)
    k_prev = similar(u)

    t = first(ode.tspan)
    iter = 0

    # For entropy relaxation
    direction = similar(u)
    gamma = one(eltype(u))

    integrator = vanderHouwenRelaxationIntegrator(u, du, u_tmp, t, dt, zero(dt), iter,
                                                  ode.p,
                                                  (prob = ode,), ode.f,
                                                  alg.van_der_houwen_alg,
                                                  SimpleIntegrator2NOptions(callback,
                                                                            ode.tspan;
                                                                            kwargs...),
                                                  false,
                                                  k_prev, direction, gamma,
                                                  alg.relaxation_solver)

    # initialize callbacks
    if callback isa CallbackSet
        foreach(callback.continuous_callbacks) do cb
            throw(ArgumentError("Continuous callbacks are unsupported with van-der-Houwen time integration methods."))
        end
        foreach(callback.discrete_callbacks) do cb
            cb.initialize(cb, integrator.u, integrator.t, integrator)
        end
    end

    return integrator
end

# Fakes `solve`: https://diffeq.sciml.ai/v6.8/basics/overview/#Solving-the-Problems-1
function solve(ode::ODEProblem,
               alg::Union{vanderHouwenAlgorithm, vanderHouwenRelaxationAlgorithm};
               dt, callback = nothing, kwargs...)
    integrator = init(ode, alg, dt = dt, callback = callback; kwargs...)

    # Start actual solve
    solve!(integrator)
end

function step!(integrator::vanderHouwenIntegrator)
    @unpack prob = integrator.sol
    @unpack alg = integrator
    t_end = last(prob.tspan)
    callbacks = integrator.opts.callback

    @assert !integrator.finalstep
    if isnan(integrator.dt)
        error("time step size `dt` is NaN")
    end

    # if the next iteration would push the simulation beyond the end time, set dt accordingly
    if integrator.t + integrator.dt > t_end ||
       isapprox(integrator.t + integrator.dt, t_end)
        integrator.dt = t_end - integrator.t
        terminate!(integrator)
    end

    @trixi_timeit timer() "VdH RK integration step" begin
        num_stages = length(alg.c)

        # First stage
        integrator.f(integrator.du, integrator.u, prob.p, integrator.t)
        @threaded for i in eachindex(integrator.du)
            integrator.k_prev[i] = integrator.du[i] # Faster than broadcasted version (with .=)
        end

        # Second to last stage
        for stage in 2:num_stages
            @threaded for i in eachindex(integrator.u)
                integrator.u_tmp[i] = integrator.u[i] +
                                      alg.a[stage] * integrator.dt * integrator.du[i]
            end

            integrator.f(integrator.du, integrator.u_tmp, prob.p,
                         integrator.t + alg.c[stage] * integrator.dt)

            @threaded for i in eachindex(integrator.u)
                integrator.u[i] += alg.b[stage - 1] * integrator.dt *
                                   integrator.k_prev[i]
                integrator.k_prev[i] = integrator.du[i] # Faster than broadcasted version (with .=)
            end
        end

        # Update solution
        @threaded for i in eachindex(integrator.u)
            integrator.u[i] += integrator.dt * alg.b[num_stages] * integrator.du[i]
        end
    end

    integrator.iter += 1
    integrator.t += integrator.dt

    @trixi_timeit timer() "Step-Callbacks" begin
        # handle callbacks
        if callbacks isa CallbackSet
            foreach(callbacks.discrete_callbacks) do cb
                if cb.condition(integrator.u, integrator.t, integrator)
                    cb.affect!(integrator)
                end
                return nothing
            end
        end
    end

    # respect maximum number of iterations
    if integrator.iter >= integrator.opts.maxiters && !integrator.finalstep
        @warn "Interrupted. Larger maxiters is needed."
        terminate!(integrator)
    end
end

function step!(integrator::vanderHouwenRelaxationIntegrator)
    @unpack prob = integrator.sol
    @unpack alg = integrator
    t_end = last(prob.tspan)
    callbacks = integrator.opts.callback

    @assert !integrator.finalstep
    if isnan(integrator.dt)
        error("time step size `dt` is NaN")
    end

    # if the next iteration would push the simulation beyond the end time, set dt accordingly
    if integrator.t + integrator.dt > t_end ||
       isapprox(integrator.t + integrator.dt, t_end)
        integrator.dt = t_end - integrator.t
        terminate!(integrator)
    end

    @trixi_timeit timer() "Relaxation vdH RK integration step" begin
        num_stages = length(alg.c)

        mesh, equations, dg, cache = mesh_equations_solver_cache(prob.p)

        u_wrap = wrap_array(integrator.u, prob.p)
        S_old = integrate(entropy_math, u_wrap, mesh, equations, dg, cache)

        u_tmp_wrap = wrap_array(integrator.u_tmp, prob.p)

        # First stage
        integrator.f(integrator.du, integrator.u, prob.p, integrator.t)
        @threaded for i in eachindex(integrator.u)
            integrator.direction[i] = alg.b[1] * integrator.du[i] * integrator.dt

            integrator.k_prev[i] = integrator.du[i] # Faster than broadcasted version (with .=)
        end

        du_wrap = wrap_array(integrator.du, prob.p)
        # Entropy change due to first stage
        dS = alg.b[1] * integrator.dt *
             int_w_dot_stage(du_wrap, u_wrap, mesh, equations, dg, cache)

        @threaded for i in eachindex(integrator.u)
            integrator.u_tmp[i] = integrator.u[i] +
                                  alg.a[2] * integrator.dt * integrator.du[i]
        end

        # Second to last stage
        for stage in 2:(num_stages - 1)
            integrator.f(integrator.du, integrator.u_tmp, prob.p,
                         integrator.t + alg.c[stage] * integrator.dt)

            dS += alg.b[stage] * integrator.dt *
                  int_w_dot_stage(du_wrap, u_tmp_wrap, mesh, equations, dg, cache)

            @threaded for i in eachindex(integrator.u)
                integrator.direction[i] += alg.b[stage] * integrator.du[i] *
                                           integrator.dt

                integrator.u_tmp[i] += integrator.dt *
                                       ((alg.b[stage - 1] - alg.a[stage]) *
                                        integrator.k_prev[i] +
                                        alg.a[stage + 1] * integrator.du[i])

                integrator.k_prev[i] = integrator.du[i] # Faster than broadcasted version (with .=)
            end
        end

        # Last stage
        integrator.f(integrator.du, integrator.u_tmp, prob.p,
                     integrator.t + alg.c[num_stages] * integrator.dt)

        dS += alg.b[num_stages] * integrator.dt *
              int_w_dot_stage(du_wrap, u_tmp_wrap, mesh, equations, dg, cache)

        @threaded for i in eachindex(integrator.u)
            integrator.direction[i] += alg.b[num_stages] * integrator.du[i] *
                                       integrator.dt
        end

        direction_wrap = wrap_array(integrator.direction, prob.p)

        @trixi_timeit timer() "Relaxation solver" relaxation_solver!(integrator,
                                                                     u_tmp_wrap, u_wrap,
                                                                     direction_wrap,
                                                                     S_old, dS,
                                                                     mesh, equations,
                                                                     dg, cache,
                                                                     integrator.relaxation_solver)

        integrator.iter += 1
        update_t_relaxation!(integrator)

        # Do relaxed update
        @threaded for i in eachindex(integrator.u)
            integrator.u[i] += integrator.gamma * integrator.direction[i]
        end
    end

    @trixi_timeit timer() "Step-Callbacks" begin
        # handle callbacks
        if callbacks isa CallbackSet
            foreach(callbacks.discrete_callbacks) do cb
                if cb.condition(integrator.u, integrator.t, integrator)
                    cb.affect!(integrator)
                end
                return nothing
            end
        end
    end

    # respect maximum number of iterations
    if integrator.iter >= integrator.opts.maxiters && !integrator.finalstep
        @warn "Interrupted. Larger maxiters is needed."
        terminate!(integrator)
    end
end

# used for AMR
function Base.resize!(integrator::vanderHouwenIntegrator, new_size)
    resize!(integrator.u, new_size)
    resize!(integrator.du, new_size)
    resize!(integrator.u_tmp, new_size)
    resize!(integrator.k_prev, new_size)
end

function Base.resize!(integrator::vanderHouwenRelaxationIntegrator, new_size)
    resize!(integrator.u, new_size)
    resize!(integrator.du, new_size)
    resize!(integrator.u_tmp, new_size)
    resize!(integrator.k_prev, new_size)

    resize!(integrator.direction, new_size)
end
end # @muladd
