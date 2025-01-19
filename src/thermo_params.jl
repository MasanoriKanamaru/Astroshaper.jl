

# ****************************************************************
#              Thermal skin depth & Thermal inertia
# ****************************************************************

"""
    thermal_skin_depth(P, k, ρ, Cp) -> l_2π

# Arguments
- `P`  : Cycle of thermal cycle [sec]
- `k`  : Thermal conductivity [W/m/K]
- `ρ`  : Material density [kg/m³]
- `Cₚ` : Heat capacity [J/kg/K]

# Return
- `l_2π` : Thermal skin depth [m], as defined in Rozitis & Green (2011).
"""
thermal_skin_depth(P, k, ρ, Cₚ) = @. √(4π * P * k / (ρ * Cₚ))


"""
    thermal_inertia(k, ρ, Cp) -> Γ

# Arguments
- `k`  : Thermal conductivity [W/m/K]
- `ρ`  : Material density [kg/m³]
- `Cₚ` : Heat capacity [J/kg/K]

# Return
- `Γ` : Thermal inertia [tiu (thermal intertia unit)]
"""
thermal_inertia(k, ρ, Cₚ) = @. √(k * ρ * Cₚ)


# ****************************************************************
#               Struct for thermophysical parameters
# ****************************************************************

abstract type AbstractThermoParams end

"""
    struct ThermoParams

# Fields
- `period`          : Period of thermal cycle (rotation period) [sec]
- `skindepth`       : Vector of thermal skin depth for each facet [m]
- `inertia`         : Vector of thermal inertia for each facet [thermal inertia unit (tiu)]
- `reflectance_vis` : Vector of reflectance in visible light for each facet [-]
- `reflectance_ir`  : Vector of reflectance in thermal infrared for each facet [-]
- `emissivity`      : Vector of emissivity for each facet [-]

- `z_max`   : Depth of the lower boundary of a heat conduction equation [m]
- `Δz`      : Depth step width [m]
- `n_depth` : Number of depth steps
"""
struct ThermoParams <: AbstractThermoParams
    period          ::Float64
    skindepth       ::Vector{Float64}
    inertia         ::Vector{Float64}
    reflectance_vis ::Vector{Float64}
    reflectance_ir  ::Vector{Float64}
    emissivity      ::Vector{Float64}

    z_max   ::Float64
    Δz      ::Float64
    n_depth ::Int
end


"""
    ThermoParams(
        period          ::Float64,
        skindepth       ::Float64,
        inertia         ::Float64,
        reflectance_vis ::Float64,
        reflectance_ir  ::Float64,
        emissivity      ::Float64,
        z_max           ::Float64,
        Δz              ::Float64,
        n_depth         ::Int
    )

Outer constructor for `ThermoParams`.
You can give the same parameters to all facets by `Float64`.

# Arguments
- `period`          : Period of thermal cycle (rotation period) [sec]
- `skindepth`       : Thermal skin depth [m]
- `inertia`         : Thermal inertia [tiu]
- `reflectance_vis` : Reflectance in visible light [-]
- `reflectance_ir`  : Reflectance in thermal infrared [-]
- `emissivity`      : Emissivity [-]
- `z_max`           : Depth of the lower boundary of a heat conduction equation [m]
- `Δz`              : Depth step width [m]
- `n_depth`         : Number of depth steps
"""
function ThermoParams(
    period          ::Float64,
    skindepth       ::Float64,
    inertia         ::Float64,
    reflectance_vis ::Float64,
    reflectance_ir  ::Float64,
    emissivity      ::Float64,
    z_max           ::Float64,
    Δz              ::Float64,
    n_depth         ::Int
)

    return ThermoParams(period, [skindepth], [inertia], [reflectance_vis], [reflectance_ir], [emissivity], z_max, Δz, n_depth)
end
