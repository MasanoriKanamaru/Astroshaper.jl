
# ****************************************************************
#         Types of solvers for heat conduction equations
# ****************************************************************

"""
Abstract type of a solver for a heat conduction equation 
"""
abstract type HeatConductionSolver end


"""
Type of the forward Euler method:
- Explicit in time
- First order in time

The `ForwardEulerSolver` type includes a vector for the temperature at the next time step.
"""
struct ForwardEulerSolver <: HeatConductionSolver
    T::Vector{Float64}
end

ForwardEulerSolver(thermo_params::AbstractThermoParams) = ForwardEulerSolver(thermo_params.n_depth)
ForwardEulerSolver(N::Integer) = ForwardEulerSolver(zeros(N))


"""
Type of the backward Euler method:
- Implicit in time (Unconditionally stable in the heat conduction equation)
- First order in time

The `BackwardEulerSolver` type has vectors for the tridiagonal matrix algorithm.
"""
struct BackwardEulerSolver <: HeatConductionSolver
    a::Vector{Float64}
    b::Vector{Float64}
    c::Vector{Float64}
    d::Vector{Float64}
    x::Vector{Float64}
end

BackwardEulerSolver(thermo_params::AbstractThermoParams) = BackwardEulerSolver(thermo_params.n_depth)
BackwardEulerSolver(N::Integer) = BackwardEulerSolver(zeros(N), zeros(N), zeros(N), zeros(N), zeros(N))


"""
Type of the Crank-Nicolson method:
- Implicit in time (Unconditionally stable in the heat conduction equation)
- Second order in time

The `CrankNicolsonSolver` type has vectors for the tridiagonal matrix algorithm.

# References
- https://en.wikipedia.org/wiki/Crank–Nicolson_method
"""
struct CrankNicolsonSolver <: HeatConductionSolver
    a::Vector{Float64}
    b::Vector{Float64}
    c::Vector{Float64}
    d::Vector{Float64}
    x::Vector{Float64}
end

CrankNicolsonSolver(thermo_params::AbstractThermoParams) = CrankNicolsonSolver(thermo_params.n_depth)
CrankNicolsonSolver(N::Integer) = CrankNicolsonSolver(zeros(N), zeros(N), zeros(N), zeros(N), zeros(N))


# ****************************************************************
#    Types of boundary conditions for heat conduction equations
# ****************************************************************

"""
Abstract type of a boundary condition for a heat conduction equation
"""
abstract type BoundaryCondition end


"""
Singleton type of radiation boundary condition
"""
struct RadiationBoundaryCondition <: BoundaryCondition end


"""
Singleton type of insulation boundary condition
"""
struct InsulationBoundaryCondition <: BoundaryCondition end


"""
Type of isothermal boundary condition
"""
struct IsothermalBoundaryCondition <: BoundaryCondition
    T_iso::Float64
end


# ****************************************************************
#                Types of thermophsycal models
# ****************************************************************


"""
Abstract type of a thermophysical model
"""
abstract type ThermoPhysicalModel end


"""
    broadcast_thermo_params!(thermo_params::ThermoParams, n_face::Int)

Broadcast the thermophysical parameters to all faces if the values are uniform globally.

# Arguments
- `thermo_params` : Thermophysical parameters
- `n_face`        : Number of faces of the shape model
"""
function broadcast_thermo_params!(thermo_params::ThermoParams, n_face::Int)
    if length(thermo_params.inertia) == 1
        resize!(thermo_params.skindepth,       n_face)
        resize!(thermo_params.inertia,         n_face)
        resize!(thermo_params.reflectance_vis, n_face)
        resize!(thermo_params.reflectance_ir,  n_face)
        resize!(thermo_params.emissivity,      n_face)

        thermo_params.skindepth[2:end]       .= thermo_params.skindepth[begin]
        thermo_params.inertia[2:end]         .= thermo_params.inertia[begin]
        thermo_params.reflectance_vis[2:end] .= thermo_params.reflectance_vis[begin]
        thermo_params.reflectance_ir[2:end]  .= thermo_params.reflectance_ir[begin]
        thermo_params.emissivity[2:end]      .= thermo_params.emissivity[begin]
    elseif length(thermo_params.inertia) == n_face
        # Do nothing
    else
        throw(ArgumentError("The length of the thermophysical parameters is invalid."))
    end
end


"""
    broadcast_thermo_params!(thermo_params::ThermoParams, shape::ShapeModel)

Broadcast the thermophysical parameters to all faces if the values are uniform globally.

# Arguments
- `thermo_params` : Thermophysical parameters
- `shape`         : Shape model
"""
broadcast_thermo_params!(thermo_params::ThermoParams, shape::ShapeModel) = broadcast_thermo_params!(thermo_params, length(shape.faces))


"""
    struct SingleTPM <: ThermoPhysicalModel

# Fields
- `shape`          : Shape model
- `thermo_params`  : Thermophysical parameters

- `flux`           : Flux on each face. Matrix of size (Number of faces, 3). Three components are:
    - `flux[:, 1]`     : F_sun,  flux of direct sunlight
    - `flux[:, 2]`     : F_scat, flux of scattered light
    - `flux[:, 3]`     : F_rad,  flux of thermal emission from surrounding surface
- `temperature`    : Temperature matrix `(n_depth, n_face)` according to the number of depth cells `n_depth` and the number of faces `n_face`.

- `face_forces`    : Thermal force on each face
- `force`          : Thermal recoil force at body-fixed frame (Yarkovsky effect)
- `torque`         : Thermal recoil torque at body-fixed frame (YORP effect)

- `SELF_SHADOWING` : Flag to consider self-shadowing
- `SELF_HEATING`   : Flag to consider self-heating
- `SOLVER`         : Solver of heat conduction equation
- `BC_UPPER`       : Boundary condition at the upper boundary
- `BC_LOWER`       : Boundary condition at the lower boundary

# TODO:
- roughness_maps   ::ShapeModel[]
"""
struct SingleTPM{P<:AbstractThermoParams, S<:HeatConductionSolver, BU<:BoundaryCondition, BL<:BoundaryCondition} <: ThermoPhysicalModel
    shape          ::ShapeModel
    thermo_params  ::P

    flux           ::Matrix{Float64}  # (n_face, 3)
    temperature    ::Matrix{Float64}  # (n_depth, n_face)

    face_forces    ::Vector{SVector{3, Float64}}
    force          ::MVector{3, Float64}
    torque         ::MVector{3, Float64}

    SELF_SHADOWING ::Bool
    SELF_HEATING   ::Bool
    SOLVER         ::S
    BC_UPPER       ::BU
    BC_LOWER       ::BL
end


"""
    SingleTPM(shape, thermo_params; SELF_SHADOWING=true, SELF_HEATING=true) -> stpm

Construct a thermophysical model for a single asteroid (`SingleTPM`).

# Arguments
- `shape`          : Shape model
- `thermo_params`  : Thermophysical parameters

# Keyword arguments
- `SELF_SHADOWING` : Flag to consider self-shadowing
- `SELF_HEATING`   : Flag to consider self-heating
- `SOLVER`         : Solver of heat conduction equation
- `BC_UPPER`       : Boundary condition at the upper boundary
- `BC_LOWER`       : Boundary condition at the lower boundary
"""
function SingleTPM(shape, thermo_params; SELF_SHADOWING, SELF_HEATING, SOLVER, BC_UPPER, BC_LOWER)

    broadcast_thermo_params!(thermo_params, shape)

    n_depth = thermo_params.n_depth
    n_face = length(shape.faces)

    flux = zeros(n_face, 3)
    temperature = zeros(n_depth, n_face)

    face_forces = zeros(SVector{3, Float64}, n_face)
    force  = zero(MVector{3, Float64})
    torque = zero(MVector{3, Float64})

    return SingleTPM(shape, thermo_params, flux, temperature, face_forces, force, torque, SELF_SHADOWING, SELF_HEATING, SOLVER, BC_UPPER, BC_LOWER)
end


"""
    struct BinaryTPM{M1, M2} <: ThermoPhysicalModel

# Fields
- `pri`              : TPM for the primary
- `sec`              : TPM for the secondary
- `MUTUAL_SHADOWING` : Flag to consider mutual shadowing
- `MUTUAL_HEATING`   : Flag to consider mutual heating
"""
struct BinaryTPM{M1, M2} <: ThermoPhysicalModel
    pri              ::M1
    sec              ::M2

    MUTUAL_SHADOWING ::Bool
    MUTUAL_HEATING   ::Bool
end


"""
    BinaryTPM(pri, sec; MUTUAL_SHADOWING=true, MUTUAL_HEATING=true) -> btpm

Construct a thermophysical model for a binary asteroid (`BinaryTPM`).
"""
function BinaryTPM(pri, sec; MUTUAL_SHADOWING, MUTUAL_HEATING)
    return BinaryTPM(pri, sec, MUTUAL_SHADOWING, MUTUAL_HEATING)
end


"""
Return surface temperature of a single asteroid corrsponding to each face.
"""
surface_temperature(stpm::SingleTPM) = stpm.temperature[begin, :]


# ****************************************************************
#                     Output data format
# ****************************************************************

"""
    struct SingleTPMResult

Output data format for `SingleTPM` 

# Fields
## Saved at all time steps
- `times`  : Timesteps, given the same vector as `ephem.time` [s]
- `E_in`   : Input energy per second on the whole surface [W]
- `E_out`  : Output enegey per second from the whole surface [W]
- `E_cons` : Energy conservation ratio [-], ratio of total energy going out to total energy coming in in the last rotation cycle
- `force`  : Thermal force on the asteroid [N]
- `torque` : Thermal torque on the asteroid [N ⋅ m]

## Saved only at the time steps desired by the user
- `times_to_save`          : Timesteps to save temperature and thermal force on every face [s]
- `depth_nodes`            : Depths of the calculation nodes for 1-D heat conduction [m], a vector of size `n_depth`
- `surface_temperature`    : Surface temperature [K], a matrix in size of `(n_face, n_time)`.
    - `n_face` : Number of faces
    - `n_time` : Number of time steps to save surface temperature
- `subsurface_temperature` : Temperature [K] as a function of depth [m] and time [s], `Dict` with face ID as key and a matrix `(n_depth, n_time)` as an entry.
    - `n_depth` : The number of the depth nodes
    - `n_time` : The number of time steps to save temperature
- `face_forces`            : Thermal force on every face of the shape model [N], a matrix in size of `(n_face, n_time)`.
    - `n_face` : Number of faces
    - `n_time` : Number of time steps to save surface temperature
"""
struct SingleTPMResult
    times  ::Vector{Float64}
    E_in   ::Vector{Float64}
    E_out  ::Vector{Float64}
    E_cons ::Vector{Float64}
    force  ::Vector{SVector{3, Float64}}
    torque ::Vector{SVector{3, Float64}}

    times_to_save          ::Vector{Float64}
    depth_nodes            ::Vector{Float64}
    surface_temperature    ::Matrix{Float64}
    subsurface_temperature ::Dict{Int, Matrix{Float64}}
    face_forces            ::Matrix{SVector{3, Float64}}
end


"""
Outer constructor of `SingleTPMResult`

# Arguments
- `stpm`          : Thermophysical model for a single asteroid
- `ephem`         : Ephemerides
- `times_to_save` : Timesteps to save temperature
- `face_ID`       : Face indices to save subsurface temperature
"""
function SingleTPMResult(stpm::SingleTPM, ephem, times_to_save::Vector{Float64}, face_ID::Vector{Int})
    n_step = length(ephem.time)             # Number of time steps
    n_step_to_save = length(times_to_save)  # Number of time steps to save temperature
    n_face = length(stpm.shape.faces)       # Number of faces of the shape model

    E_in   = zeros(n_step)
    E_out  = zeros(n_step)
    E_cons = fill(NaN, n_step)
    force  = zeros(SVector{3, Float64}, n_step)
    torque = zeros(SVector{3, Float64}, n_step)

    depth_nodes = stpm.thermo_params.Δz * (0:stpm.thermo_params.n_depth-1)
    surface_temperature = zeros(n_face, n_step_to_save)
    subsurface_temperature = Dict{Int,Matrix{Float64}}(
        i => zeros(stpm.thermo_params.n_depth, n_step_to_save) for i in face_ID
    )
    face_forces = zeros(SVector{3, Float64}, n_face, n_step_to_save)

    return SingleTPMResult(
        ephem.time,
        E_in,
        E_out,
        E_cons,
        force,
        torque,
        times_to_save,
        depth_nodes,
        surface_temperature,
        subsurface_temperature,
        face_forces,
    )
end


"""
    struct BinaryTPMResult

Output data format for `BinaryTPM`

# Fields
- `pri` : TPM result for the primary
- `sec` : TPM result for the secondary
"""
struct BinaryTPMResult
    pri::SingleTPMResult
    sec::SingleTPMResult
end


"""
Outer constructor of `BinaryTPMResult`

# Arguments
- `btpm`          : Thermophysical model for a binary asteroid
- `ephem`         : Ephemerides
- `times_to_save` : Timesteps to save temperature (Common to both the primary and the secondary)
- `face_ID_pri`   : Face indices to save subsurface temperature of the primary
- `face_ID_sec`   : Face indices to save subsurface temperature of the secondary
"""
function BinaryTPMResult(btpm::BinaryTPM, ephem, times_to_save::Vector{Float64}, face_ID_pri::Vector{Int}, face_ID_sec::Vector{Int})
    result_pri = SingleTPMResult(btpm.pri, ephem, times_to_save, face_ID_pri)
    result_sec = SingleTPMResult(btpm.sec, ephem, times_to_save, face_ID_sec)

    return BinaryTPMResult(result_pri, result_sec)
end


"""
    update_TPM_result!(result::SingleTPMResult, stpm::SingleTPM, i_time::Integer)

Save the results of TPM at the time step `i_time` to `result`.

# Arguments
- `result` : Output data format for `SingleTPM`
- `stpm`   : Thermophysical model for a single asteroid
- `i_time` : Time step to save data
"""
function update_TPM_result!(result::SingleTPMResult, stpm::SingleTPM, i_time::Integer)
    result.E_in[i_time]   = energy_in(stpm)
    result.E_out[i_time]  = energy_out(stpm)
    result.force[i_time]  = stpm.force
    result.torque[i_time] = stpm.torque

    P  = stpm.thermo_params.period # Rotation period
    t  = result.times[i_time]      # Current time
    t₀ = result.times[begin]   # Time at the beginning of the simulation

    if t > t₀ + P  # Note that `E_cons` cannot be calculated during the first rotation
        n_step_in_period = count(@. t - P ≤ result.times < t)  # Number of time steps within the last rotation

        ΣE_in  = sum(result.E_in[n-1]  * (result.times[n] - result.times[n-1]) for n in (i_time - n_step_in_period + 1):i_time)
        ΣE_out = sum(result.E_out[n-1] * (result.times[n] - result.times[n-1]) for n in (i_time - n_step_in_period + 1):i_time)

        result.E_cons[i_time] = ΣE_out / ΣE_in
    end

    if t in result.times_to_save  # In the step of saving temperature
        i_time_save = findfirst(isequal(t), result.times_to_save)

        result.surface_temperature[:, i_time_save] .= surface_temperature(stpm)

        for (i, temperature) in result.subsurface_temperature
            temperature[:, i_time_save] .= stpm.temperature[:, i]
        end

        result.face_forces[:, i_time_save] .= stpm.face_forces
    end
end


"""
    update_TPM_result!(result::BinaryTPMResult, btpm::BinaryTPM, ephem, i_time::Integer)

Save the results of TPM at the time step `i_time` to `result`.

# Arguments
- `result` : Output data format for `BinaryTPM`
- `btpm`   : Thermophysical model for a binary asteroid
- `ephem`  : Ephemerides
- `i_time`     : Time step
"""
function update_TPM_result!(result::BinaryTPMResult, btpm::BinaryTPM, i_time::Integer)
    update_TPM_result!(result.pri, btpm.pri, i_time)
    update_TPM_result!(result.sec, btpm.sec, i_time)
end


"""
    export_TPM_results(dirpath, result::SingleTPMResult)

Export the result of `SingleTPM` to CSV files. 
The output files are saved in the following directory structure:

    dirpath
    ├── physical_quantities.csv
    ├── subsurface_temperature.csv
    ├── surface_temperature.csv
    └── thermal_force.csv

# Arguments
- `dirpath` : Path to the directory to save CSV files.
- `result`  : Output data format for `SingleTPM`
"""
function export_TPM_results(dirpath, result::SingleTPMResult)
    
    df = DataFrame()
    df.time     = result.times
    df.E_in     = result.E_in
    df.E_out    = result.E_out
    df.E_cons   = result.E_cons
    df.force_x  = [f[1] for f in result.force]
    df.force_y  = [f[2] for f in result.force]
    df.force_z  = [f[3] for f in result.force]
    df.torque_x = [τ[1] for τ in result.torque]
    df.torque_y = [τ[2] for τ in result.torque]
    df.torque_z = [τ[3] for τ in result.torque]
    
    CSV.write(joinpath(dirpath, "physical_quantities.csv"), df)

    ##= Surface temperature =##
    filepath = joinpath(dirpath, "surface_temperature.csv")
    df = hcat(
        DataFrame(time = result.times_to_save),
        DataFrame(
            result.surface_temperature',
            ["face_$(i)" for i in 1:size(result.surface_temperature, 1)]
        ),
    )

    CSV.write(filepath, df)

    ##= Subsurface temperature =##
    filepath = joinpath(dirpath, "subsurface_temperature.csv")
    
    nrows = length(result.depth_nodes) * length(result.times_to_save)
    df = DataFrame(
        time  = reshape([t for _ in result.depth_nodes, t in result.times_to_save], nrows),
        depth = reshape([d for d in result.depth_nodes, _ in result.times_to_save], nrows),
    )

    # Add a column for each face
    for (i, subsurface_temperature) in collect(result.subsurface_temperature)
        df[:, "face_$(i)"] =
            reshape(subsurface_temperature, length(subsurface_temperature))
    end

    # Sort the columns by the face ID
    keys_sorted = sort(names(df[:, 3:end]), by=x->parse(Int, replace(x, r"[^0-9]" => "")))
    df = df[:, ["time", "depth", keys_sorted...]]

    CSV.write(filepath, df)

    ##= Thermal force on every face of the shape model =##
    filepath = joinpath(dirpath, "thermal_force.csv")

    n_face = size(result.face_forces, 1)  # Number of faces of the shape model
    n_step = size(result.face_forces, 2)  # Number of time steps to save temperature
    nrows = n_face * n_step

    df = DataFrame(
        time = reshape([t for _ in 1:n_face, t in result.times_to_save], nrows),
        face = reshape([i for i in 1:n_face, _ in result.times_to_save], nrows),
    )
    df.x = reshape([f[1] for f in result.face_forces], nrows)  # x-component of the thermal force
    df.y = reshape([f[2] for f in result.face_forces], nrows)  # y-component of the thermal force
    df.z = reshape([f[3] for f in result.face_forces], nrows)  # z-component of the thermal force

    CSV.write(filepath, df)
end


"""
    export_TPM_results(dirpath, result::BinaryTPMResult)

Export the result of `BinaryTPM` to CSV files. 
The output files are saved in the following directory structure:

    dirpath
    ├── pri
    │   ├── physical_quantities.csv
    │   ├── subsurface_temperature.csv
    │   ├── surface_temperature.csv
    │   └── thermal_force.csv
    └── sec
        ├── physical_quantities.csv
        ├── subsurface_temperature.csv
        ├── surface_temperature.csv
        └── thermal_force.csv

# Arguments
- `dirpath` : Path to the directory to save CSV files.
- `result`  : Output data format for `BinaryTPM`
"""
function export_TPM_results(dirpath, result::BinaryTPMResult)
    dirpath_pri = joinpath(dirpath, "pri")
    dirpath_sec = joinpath(dirpath, "sec")

    mkpath(dirpath_pri)
    mkpath(dirpath_sec)

    export_TPM_results(dirpath_pri, result.pri)
    export_TPM_results(dirpath_sec, result.sec)
end


# ****************************************************************
#                   Initialize temperatures
# ****************************************************************


"""
    subsolar_temperature(r☉) -> Tₛₛ

Subsolar temperature [K] on an asteroid at a heliocentric distance `r☉` [m],
assuming radiative equilibrium with zero conductivity.
"""
subsolar_temperature(r☉, params::AbstractThermoParams) = subsolar_temperature(r☉, params.reflectance_vis, params.emissivity)

function subsolar_temperature(r☉, R_vis, ε)
    Φ = SOLAR_CONST / SPICE.convrt(norm(r☉), "m", "au")^2  # Energy flux at the solar distance [W/m²]
    Tₛₛ = ((1 - R_vis) * Φ / (ε * σ_SB))^(1/4)

    return Tₛₛ
end


"""
    init_temperature!(stpm::SingleTPM, T₀::Real)

Initialize all temperature cells at the given temperature `T₀`

# Arguments
- `stpm` : Thermophysical model for a single asteroid
- `T₀`   : Initial temperature of all cells [K]
"""
function init_temperature!(stpm::SingleTPM, T₀::Real)
    stpm.temperature .= T₀
end


"""
    init_temperature!(btpm::BinaryTPM, T₀::Real)

Initialize all temperature cells at the given temperature `T₀`

# Arguments
- `btpm` : Thermophysical model for a binary asteroid
- `T₀`   : Initial temperature of all cells [K]
"""
function init_temperature!(btpm::BinaryTPM, T₀::Real)
    init_temperature!(btpm.pri, T₀)
    init_temperature!(btpm.sec, T₀)
end


# ****************************************************************
#                    Thermophysical modeling
# ****************************************************************


# """
# """
# function run_TPM!(shape::ShapeModel, orbit::OrbitalElements, spin::SpinParams, thermo_params::AbstractThermoParams, savepath="tmp.jld2")
#     @unpack P, Δt, t_begin, t_end = thermo_params
    
#     init_temperatures_zero!(shape, thermo_params)

#     ts = (t_begin:Δt:t_end) * P
#     timestamp = prep_timestamp(ts)
#     # surface_temperature_table = zeros(length(shape.faces), Int(1/thermo_params.Δt)-1)

#     for (i, t) in enumerate(ts)
#         update_orbit!(orbit, t)
#         update_spin!(spin, t)
            
#         r̂☉ = normalize(orbit.r) * -1  # Shift the origin from the sun to the body
#         r̂☉ = orbit_to_body(r̂☉, spin)
        
#         update_flux_sun!(shape, orbit.F☉, r̂☉)
#         update_flux_scat_single!(shape, thermo_params)
#         update_flux_rad_single!(shape, thermo_params)
        
#         update_force!(shape, thermo_params)
#         sum_force_torque!(shape)
        
#         f = SVector{3}(shape.force)   # Body-fixed frame
#         τ = SVector{3}(shape.torque)  # Body-fixed frame

#         f = body_to_orbit(f, spin)  # Orbital plane frame
#         τ = body_to_orbit(τ, spin)  # Orbital plane frame

#         E_in, E_out, E_cons = energy_io(shape, thermo_params)

#         save_timestamp!(timestamp, i, orbit.u, orbit.ν, spin.ϕ, f..., τ..., E_in, E_out, E_cons)
        
#         update_temperatures!(shape, thermo_params)
#     end
#     mean_energy_cons_frac!(timestamp, spin)
#     jldsave(savepath; shape, orbit, spin, thermo_params, timestamp)

#     timestamp
# end

"""
    run_TPM!(stpm::SingleTPM, ephem, savepath)

Run TPM for a single asteroid.

# Arguments
- `stpm`          : Thermophysical model for a single asteroid
- `ephem`         : Ephemerides
    - `ephem.time` : Ephemeris times
    - `ephem.sun`  : Sun's position in the asteroid-fixed frame (Not normalized)
- `times_to_save` : Timesteps to save temperature
- `face_ID`       : Face indices where to save subsurface termperature

# Keyword arguments
- `show_progress` : Flag to show the progress meter
"""
function run_TPM!(stpm::SingleTPM, ephem, times_to_save::Vector{Float64}, face_ID::Vector{Int}; show_progress=true)

    result = SingleTPMResult(stpm, ephem, times_to_save, face_ID)

    ## ProgressMeter setting
    if show_progress
        p = Progress(length(ephem.time); dt=1, desc="Running TPM...", showspeed=true)
        ProgressMeter.ijulia_behavior(:clear)
    end
    
    for i_time in eachindex(ephem.time)
        r☉ = ephem.sun[i_time]

        update_flux_sun!(stpm, r☉)
        update_flux_scat_single!(stpm)
        update_flux_rad_single!(stpm)
        
        update_thermal_force!(stpm)

        update_TPM_result!(result, stpm, i_time)  # Save data
        
        ## Update the progress meter
        if show_progress
            showvalues = [
                ("Timestep ", i_time),
                ("E_cons   ", result.E_cons[i_time]),
            ]
            ProgressMeter.next!(p; showvalues)
        end

        i_time == length(ephem.time) && break  # Stop to update the temperature at the final step
        Δt = ephem.time[i_time+1] - ephem.time[i_time]
        update_temperature!(stpm, Δt)
    end
    
    return result
end

"""
    run_TPM!(btpm::BinaryTPM, ephem, savepath)

Run TPM for a binary asteroid.

# Arguments
- `btpm`          : Thermophysical model for a binary asteroid
- `ephem`         : Ephemerides
    - `time` : Ephemeris times
    - `sun1` : Sun's position in the primary's frame
    - `sun2` : Sun's position in the secondary's frame
    - `sec`  : Secondary's position in the primary's frame
    - `P2S`  : Rotation matrix from primary to secondary frames
    - `S2P`  : Rotation matrix from secondary to primary frames
- `times_to_save` : Timesteps to save temperature
- `face_ID_pri`   : Face indices where to save subsurface termperature for the primary
- `face_ID_sec`   : Face indices where to save subsurface termperature for the secondary

# Keyword arguments
- `show_progress` : Flag to show the progress meter
"""
function run_TPM!(btpm::BinaryTPM, ephem, times_to_save::Vector{Float64}, face_ID_pri::Vector{Int}, face_ID_sec::Vector{Int}; show_progress=true)

    result = BinaryTPMResult(btpm, ephem, times_to_save, face_ID_pri, face_ID_sec)

    ## ProgressMeter setting
    if show_progress
        p = Progress(length(ephem.time); dt=1, desc="Running TPM...", showspeed=true)
        ProgressMeter.ijulia_behavior(:clear)
    end
    
    for i_time in eachindex(ephem.time)
        r☉₁ = ephem.sun1[i_time]  # Sun's position in the primary's frame
        r☉₂ = ephem.sun2[i_time]  # Sun's position in the secondary's frame
        rₛ  = ephem.sec[i_time]   # Secondary's position in the primary's frame
        R₂₁ = ephem.S2P[i_time]   # Rotation matrix from secondary to primary frames

        ## Update enegey flux
        update_flux_sun!(btpm, r☉₁, r☉₂)
        mutual_shadowing!(btpm, r☉₁, rₛ, R₂₁)  # Mutual-shadowing (eclipse)
        update_flux_scat_single!(btpm)
        update_flux_rad_single!(btpm)
        mutual_heating!(btpm, rₛ, R₂₁)         # Mutual-heating

        update_thermal_force!(btpm)

        update_TPM_result!(result, btpm, i_time)  # Save data

        ## Update the progress meter
        if show_progress
            showvalues = [
                ("Timestep             ", i_time),
                ("E_cons for primary   ", result.pri.E_cons[i_time]),
                ("E_cons for secondary ", result.sec.E_cons[i_time]),
            ]
            ProgressMeter.next!(p; showvalues)
        end
        
        ## Update temperature distribution
        i_time == length(ephem.time) && break  # Stop to update the temperature at the final step
        Δt = ephem.time[i_time+1] - ephem.time[i_time]
        update_temperature!(btpm, Δt)
    end

    return result
end

