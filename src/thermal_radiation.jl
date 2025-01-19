"""
    blackbody_radiance(λ, T) -> L_λ

Accroding to Planck's law, calculate the spectral intensity of blackbody radiation
at wavelength `λ` and temperature `T`.

# Arguments
- `λ` : Wavelength [m]
- `T` : Temperature [K]

# Return
- `L_λ` : Spectral radiance [W/m²/m/steradian]

cf. https://github.com/JuliaAstro/Planck.jl/blob/main/src/Planck.jl
"""
blackbody_radiance(λ, T) = 2 * h * c₀^2 / λ^5 / expm1(h * c₀ / (λ * k_B * T))

"""
    blackbody_radiance(T) -> L

According to Stefan-Boltzmann law, calculate the total radiance of blackbody radiation
at temperature `T`, integrated over all wavelength.

# Arguments
- `T` : Temperature [K]

# Return
- `L` : Radiance [W/m²/steradian]
"""
blackbody_radiance(T) = σ_SB * T^4


"""
    thermal_radiance(shape, emissivities, temperatures, obs) -> L

Calculate the radiance from the temperature distribution based on a shape model.

# Arguments
- `shape`        : Shape model of an asteroid
- `emissivities` : Emissivity of each facet of the shape model [-]
- `temperatures` : Temperature of each facet of the shape model [K]
- `obs`          : Position vector of the observer in the same coordinate system as `shape` [m]

# Return
- `L` : Radiance [W/m²]
"""
function thermal_radiance(shape::ShapeModel, emissivities::AbstractVector{<:Real}, temperatures::AbstractVector{<:Real}, obs::StaticVector{3, <:Real})
    if length(temperatures) != length(shape.faces)
        throw(ArgumentError("Length of `temperatures` must be equal to the number of faces of the shape model."))
    end

    L = 0.0  # Radiance [W/m²]
    for i in eachindex(shape.faces)
        c = shape.face_centers[i]
        n̂ = shape.face_normals[i]
        a = shape.face_areas[i]

        ε = emissivities[i]
        T = temperatures[i]
        
        ## Direction and distance from the facet center to the observer
        d̂ = normalize(obs - c)
        d = norm(obs - c)

        cosθ = n̂ ⋅ d̂
        cosθ < 0 && continue  # Ignore the facet if the observer is below the horizon.

        ## TODO: Ray-trace for each facet

        ## TODO: Consider the solid angle of the facet.
        ## The coefficient may change from π to 2π or 4π.
        L += ε * σ_SB * T^4 * a * cosθ / (π * d^2)
    end
    return L
end
