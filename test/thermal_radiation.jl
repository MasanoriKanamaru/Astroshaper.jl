@testset "thermal_radiation" begin

    msg = """\n
    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    |                Test: thermal_radiation                 |
    ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    """
    println(msg)

    ## Calculate the intensity of blackbody radiation of at a wavelength of 6e-7 m and a temperature of 5850 K,
    ## and compare it with the value calculated by Planck.jl.
    ## cf. https://github.com/JuliaAstro/Planck.jl/blob/main/src/Planck.jl
    @test AsteroidThermoPhysicalModels.blackbody_radiance(6e-7, 5850) ≈ 2.583616647617974e13

    ## Check the value of Stefan-Boltzmann law at a temperature of 5850 K
    @test AsteroidThermoPhysicalModels.blackbody_radiance(5850) ≈ AsteroidThermoPhysicalModels.σ_SB * 5850^4

    ######## Thermal radiance from the local terrain model ########

    ##= Load shape model =##
    path_obj = joinpath("shape", "fractal_v2572_f5000.obj")
    shape = AsteroidThermoPhysicalModels.load_shape_obj(path_obj; scale=1, find_visible_facets=true)
    n_face = length(shape.faces)  # Number of faces

    ##= Ephemerides =##
    P = SPICE.convrt(8, "hours", "seconds")  # Rotation period of the asteroid [s]
    
    ncycles = 2  # Number of cycles to perform TPM
    nsteps_in_cycle = 360  # Number of time steps in one rotation period
    
    ## TPM simulation time duration (ephemerides time)
    et_begin = 0.0  # Start time of TPM
    et_end   = et_begin + P * ncycles  # End time of TPM
    et_range = range(et_begin, et_end; length=nsteps_in_cycle*ncycles+1)

    ## Rotation matrix between the global (asteroid-fixed) frame and the local frame at `(lat, lon)`.
    local_to_global(lat, lon) = RotZYZ(lon, π/2 - lat, π/2)
    global_to_local(lat, lon) = inv(local_to_global(lat, lon))
    
    ## Solar position in the local frame of the roughness map
    lat = deg2rad(0)  # Latitude of the local frame [rad]
    lon = deg2rad(0)  # Longitude of the local frame [rad]

    r☉₀ = [SPICE.convrt(-1, "au", "m"), 0, 0]  # Sun's position at the initial time step in the asteoid-fixed frame

    """
    - `time` : Ephemeris times
    - `sun`  : Sun's position in the local frame
    """
    ephem = (
        time = collect(et_range),
        sun  = [global_to_local(lat, lon) * inv(RotZ(2π * et / P)) * r☉₀ for et in et_range],
    )
    
    ##= Thermal properties =##
    k  = 0.1     # Thermal conductivity [W/m/K]
    ρ  = 1270.0  # Density [kg/m³]
    Cₚ = 600.0   # Heat capacity [J/kg/K]
    
    l = AsteroidThermoPhysicalModels.thermal_skin_depth(P, k, ρ, Cₚ)
    Γ = AsteroidThermoPhysicalModels.thermal_inertia(k, ρ, Cₚ)

    R_vis = 0.04  # Reflectance in visible light [-]
    R_ir  = 0.0   # Reflectance in thermal infrared [-]
    ε     = 1.0   # Emissivity [-]

    z_max = 0.6   # Depth of the lower boundary of a heat conduction equation [m]
    n_depth = 41  # Number of depth steps
    Δz = z_max / (n_depth - 1)  # Depth step width [m]

    thermo_params = AsteroidThermoPhysicalModels.ThermoParams(P, l, Γ, R_vis, R_ir, ε, z_max, Δz, n_depth)

    ##= Setting of TPM =##
    stpm = AsteroidThermoPhysicalModels.SingleTPM(shape, thermo_params;
        SELF_SHADOWING = true,
        SELF_HEATING   = true,
        SOLVER         = AsteroidThermoPhysicalModels.ForwardEulerSolver(thermo_params),
        BC_UPPER       = AsteroidThermoPhysicalModels.RadiationBoundaryCondition(),
        BC_LOWER       = AsteroidThermoPhysicalModels.InsulationBoundaryCondition(),
    )
    AsteroidThermoPhysicalModels.init_temperature!(stpm, 200)

    times_to_save = ephem.time[end-nsteps_in_cycle:end]  # Save temperature during the final rotation
    face_ID = [49, 340, 648]  # Face indices to save subsurface temperature
    
    result = run_TPM!(stpm, ephem, times_to_save, face_ID)

    ##= Check the thermal radiation from the local terrain model =##
    obs_above = SVector{3, Float64}(0, 0, 1000)  # Observer is just above the local terrain model
    obs_east  = RotY(+π/6) * obs_above           # Observed from 30° east
    obs_west  = RotY(-π/6) * obs_above           # Observed from 30° west

    emissivities = fill(1.0, length(shape.faces))
    temperatures = result.surface_temperature[:, 181]

    ## Expected values of the thermal radiation
    ## - Observation from 30° east   : 19.89394442347112  [W/m²]
    ## - Observation from just above : 18.01010231251351  [W/m²]
    ## - Observation from 30° west   : 12.406927167050457 [W/m²]
    @test AsteroidThermoPhysicalModels.thermal_radiance(shape, emissivities, temperatures, obs_east)  ≈ 19.89394442347112
    @test AsteroidThermoPhysicalModels.thermal_radiance(shape, emissivities, temperatures, obs_above) ≈ 18.01010231251351
    @test AsteroidThermoPhysicalModels.thermal_radiance(shape, emissivities, temperatures, obs_west)  ≈ 12.406927167050457
end
