# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function apply_smoothing!(mesh::StructuredMesh{2}, alpha, alpha_tmp, dg, cache,
                          element_indices = eachelement(dg, cache))
    # Diffuse alpha values by setting each alpha to at least 50% of neighboring elements' alpha
    # Copy alpha values such that smoothing is indpedenent of the element access order
    alpha_tmp .= alpha

    # So far, alpha smoothing doesn't work for non-periodic initial conditions for structured meshes.
    @assert isperiodic(mesh) "alpha smoothing for structured meshes works only with periodic initial conditions so far"

    # Loop over elements, because there is no interface container
    for element in element_indices
        # Get neighboring element ids
        left = cache.elements.left_neighbors[1, element]
        lower = cache.elements.left_neighbors[2, element]

        # Apply smoothing
        alpha[left] = max(alpha_tmp[left], 0.5f0 * alpha_tmp[element], alpha[left])
        alpha[element] = max(alpha_tmp[element], 0.5f0 * alpha_tmp[left],
                             alpha[element])

        alpha[lower] = max(alpha_tmp[lower], 0.5f0 * alpha_tmp[element], alpha[lower])
        alpha[element] = max(alpha_tmp[element], 0.5f0 * alpha_tmp[lower],
                             alpha[element])
    end
end
end # @muladd
