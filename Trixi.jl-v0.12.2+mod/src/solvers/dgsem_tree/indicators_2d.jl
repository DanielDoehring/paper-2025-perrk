# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# this method is used when the indicator is constructed as for shock-capturing volume integrals
function create_cache(::Type{IndicatorHennemannGassner},
                      equations::AbstractEquations{2}, basis::LobattoLegendreBasis)
    alpha = Vector{real(basis)}()
    alpha_tmp = similar(alpha)

    A = Array{real(basis), ndims(equations)}
    indicator_threaded = [A(undef, nnodes(basis), nnodes(basis))
                          for _ in 1:Threads.nthreads()]
    modal_threaded = [A(undef, nnodes(basis), nnodes(basis))
                      for _ in 1:Threads.nthreads()]
    modal_tmp1_threaded = [A(undef, nnodes(basis), nnodes(basis))
                           for _ in 1:Threads.nthreads()]

    return (; alpha, alpha_tmp, indicator_threaded, modal_threaded, modal_tmp1_threaded)
end

# this method is used when the indicator is constructed as for AMR
function create_cache(typ::Type{IndicatorHennemannGassner}, mesh,
                      equations::AbstractEquations{2}, dg::DGSEM, cache)
    create_cache(typ, equations, dg.basis)
end

# Use this function barrier and unpack inside to avoid passing closures to Polyester.jl
# with @batch (@threaded).
# Otherwise, @threaded does not work here with Julia ARM on macOS.
# See https://github.com/JuliaSIMD/Polyester.jl/issues/88.
@inline function calc_indicator_hennemann_gassner!(indicator_hg, threshold, parameter_s,
                                                   u,
                                                   element, mesh::AbstractMesh{2},
                                                   equations, dg, cache)
    @unpack alpha_max, alpha_min, alpha_smooth, variable = indicator_hg
    @unpack alpha, alpha_tmp, indicator_threaded, modal_threaded,
    modal_tmp1_threaded = indicator_hg.cache

    indicator = indicator_threaded[Threads.threadid()]
    modal = modal_threaded[Threads.threadid()]
    modal_tmp1 = modal_tmp1_threaded[Threads.threadid()]

    # Calculate indicator variables at Gauss-Lobatto nodes
    for j in eachnode(dg), i in eachnode(dg)
        u_local = get_node_vars(u, equations, dg, i, j, element)
        indicator[i, j] = indicator_hg.variable(u_local, equations)
        #indicator[i, j] = indicator_hg.variable(u_local, equations).value # For AD
    end

    # Convert to modal representation
    multiply_scalar_dimensionwise!(modal, dg.basis.inverse_vandermonde_legendre,
                                   indicator, modal_tmp1)

    # Calculate total energies for all modes, without highest, without two highest
    total_energy = zero(eltype(modal))
    for j in 1:nnodes(dg), i in 1:nnodes(dg)
        total_energy += modal[i, j]^2
    end
    total_energy_clip1 = zero(eltype(modal))
    for j in 1:(nnodes(dg) - 1), i in 1:(nnodes(dg) - 1)
        total_energy_clip1 += modal[i, j]^2
    end
    total_energy_clip2 = zero(eltype(modal))
    for j in 1:(nnodes(dg) - 2), i in 1:(nnodes(dg) - 2)
        total_energy_clip2 += modal[i, j]^2
    end

    # Calculate energy in higher modes
    if !(iszero(total_energy))
        energy_frac_1 = (total_energy - total_energy_clip1) / total_energy
    else
        energy_frac_1 = zero(total_energy)
    end
    if !(iszero(total_energy_clip1))
        energy_frac_2 = (total_energy_clip1 - total_energy_clip2) / total_energy_clip1
    else
        energy_frac_2 = zero(total_energy_clip1)
    end
    energy = max(energy_frac_1, energy_frac_2)

    alpha_element = 1 / (1 + exp(-parameter_s / threshold * (energy - threshold)))

    # Take care of the case close to pure DG
    if alpha_element < alpha_min
        alpha_element = zero(alpha_element)
    end

    # Take care of the case close to pure FV
    if alpha_element > 1 - alpha_min
        alpha_element = one(alpha_element)
    end

    # Clip the maximum amount of FV allowed
    alpha[element] = min(alpha_max, alpha_element)
end

# Diffuse alpha values by setting each alpha to at least 50% of neighboring elements' alpha
function apply_smoothing!(mesh::Union{TreeMesh{2}, P4estMesh{2}, T8codeMesh{2}}, alpha,
                          alpha_tmp, dg, cache,
                          interface_indices = eachinterface(dg, cache),
                          mortar_indices = eachmortar(dg, cache))
    # Copy alpha values such that smoothing is indpedenent of the element access order
    alpha_tmp .= alpha

    # Loop over interfaces
    for interface in interface_indices
        # Get neighboring element ids
        left = cache.interfaces.neighbor_ids[1, interface]
        right = cache.interfaces.neighbor_ids[2, interface]

        # Apply smoothing
        alpha[left] = max(alpha_tmp[left], 0.5f0 * alpha_tmp[right], alpha[left])
        alpha[right] = max(alpha_tmp[right], 0.5f0 * alpha_tmp[left], alpha[right])
    end

    # Loop over L2 mortars
    for mortar in mortar_indices
        # Get neighboring element ids
        lower = cache.mortars.neighbor_ids[1, mortar]
        upper = cache.mortars.neighbor_ids[2, mortar]
        large = cache.mortars.neighbor_ids[3, mortar]

        # Apply smoothing
        alpha[lower] = max(alpha_tmp[lower], 0.5f0 * alpha_tmp[large], alpha[lower])
        alpha[upper] = max(alpha_tmp[upper], 0.5f0 * alpha_tmp[large], alpha[upper])
        alpha[large] = max(alpha_tmp[large], 0.5f0 * alpha_tmp[lower], alpha[large])
        alpha[large] = max(alpha_tmp[large], 0.5f0 * alpha_tmp[upper], alpha[large])
    end

    return alpha
end

# this method is used when the indicator is constructed as for shock-capturing volume integrals
function create_cache(::Type{IndicatorLöhner}, equations::AbstractEquations{2},
                      basis::LobattoLegendreBasis)
    alpha = Vector{real(basis)}()

    A = Array{real(basis), ndims(equations)}
    indicator_threaded = [A(undef, nnodes(basis), nnodes(basis))
                          for _ in 1:Threads.nthreads()]

    return (; alpha, indicator_threaded)
end

# this method is used when the indicator is constructed as for AMR
function create_cache(typ::Type{IndicatorLöhner}, mesh, equations::AbstractEquations{2},
                      dg::DGSEM, cache)
    create_cache(typ, equations, dg.basis)
end

function (löhner::IndicatorLöhner)(u::AbstractArray{<:Any, 4},
                                   mesh, equations, dg::DGSEM, cache;
                                   kwargs...)
    @assert nnodes(dg)>=3 "IndicatorLöhner only works for nnodes >= 3 (polydeg > 1)"
    @unpack alpha, indicator_threaded = löhner.cache
    @unpack variable = löhner
    resize!(alpha, nelements(dg, cache))

    @threaded for element in eachelement(dg, cache)
        indicator = indicator_threaded[Threads.threadid()]

        # Calculate indicator variables at Gauss-Lobatto nodes
        for j in eachnode(dg), i in eachnode(dg)
            u_local = get_node_vars(u, equations, dg, i, j, element)
            indicator[i, j] = variable(u_local, equations)
        end

        estimate = zero(real(dg))
        for j in eachnode(dg), i in 2:(nnodes(dg) - 1)
            # x direction
            u0 = indicator[i, j]
            up = indicator[i + 1, j]
            um = indicator[i - 1, j]
            estimate = max(estimate, local_löhner_estimate(um, u0, up, löhner))
        end

        for j in 2:(nnodes(dg) - 1), i in eachnode(dg)
            # y direction
            u0 = indicator[i, j]
            up = indicator[i, j + 1]
            um = indicator[i, j - 1]
            estimate = max(estimate, local_löhner_estimate(um, u0, up, löhner))
        end

        # use the maximum as DG element indicator
        alpha[element] = estimate
    end

    return alpha
end

# this method is used when the indicator is constructed as for shock-capturing volume integrals
function create_cache(::Type{IndicatorMax}, equations::AbstractEquations{2},
                      basis::LobattoLegendreBasis)
    alpha = Vector{real(basis)}()

    A = Array{real(basis), ndims(equations)}
    indicator_threaded = [A(undef, nnodes(basis), nnodes(basis))
                          for _ in 1:Threads.nthreads()]

    return (; alpha, indicator_threaded)
end

# this method is used when the indicator is constructed as for AMR
function create_cache(typ::Type{IndicatorMax}, mesh, equations::AbstractEquations{2},
                      dg::DGSEM, cache)
    cache = create_cache(typ, equations, dg.basis)
end

function (indicator_max::IndicatorMax)(u::AbstractArray{<:Any, 4},
                                       mesh, equations, dg::DGSEM, cache;
                                       kwargs...)
    @unpack alpha, indicator_threaded = indicator_max.cache
    resize!(alpha, nelements(dg, cache))
    indicator_variable = indicator_max.variable

    @threaded for element in eachelement(dg, cache)
        indicator = indicator_threaded[Threads.threadid()]

        # Calculate indicator variables at Gauss-Lobatto nodes
        for j in eachnode(dg), i in eachnode(dg)
            u_local = get_node_vars(u, equations, dg, i, j, element)
            indicator[i, j] = indicator_variable(u_local, equations)
        end

        alpha[element] = maximum(indicator)
    end

    return alpha
end
end # @muladd
