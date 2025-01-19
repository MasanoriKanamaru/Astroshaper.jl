# See https://github.com/Astroshaper/Astroshaper-examples/tree/main/TPM_Ryugu for more information.
@testset "TPM_zero-conductivity" begin
    DIR_OUTPUT = joinpath(@__DIR__, "output")
    rm(DIR_OUTPUT; recursive=true, force=true)
    mkpath(DIR_OUTPUT)

    msg = """\n
    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    |              Test: TPM_zero-conductivity               |
    ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    """
    println(msg)

    ##= Ephemerides =##
    P = SPICE.convrt(8, "hours", "seconds")  # Rotation period of the asteroid [s]
    n_cycle = 2                              # Number of cycles to perform TPM
    n_step_in_cycle = 72                     # Number of time steps in one rotation period

    et_begin = 0.0                     # Start time of TPM
    et_end   = et_begin + P * n_cycle  # End time of TPM
    et_range = range(et_begin, et_end; length=n_step_in_cycle*n_cycle+1)

    """
    - `time` : Ephemeris times
    - `sun`  : Sun's position in the asteroid-fixed frame
    """
    ephem = (
        time = collect(et_range),
        sun  = [inv(RotZ(2π * et / P)) * [SPICE.convrt(1, "au", "m"), 0, 0] for et in et_range],
    )

    ##= Load obj file =##
    path_obj = joinpath("shape", "ryugu_test.obj")
    shape = AsteroidThermoPhysicalModels.load_shape_obj(path_obj; scale=1000, find_visible_facets=true)
    n_face = length(shape.faces)  # Number of faces

    ##= Thermal properties: zero-conductivity case =##
    l = 0.0  # Thermal skin depth [m]
    Γ = 0.0  # Thermal inertia [J m⁻² K⁻¹ s⁻¹/2]

    R_vis = 0.1  # Reflectance in visible light [-]
    R_ir  = 0.0  # Reflectance in thermal infrared [-]
    ε     = 1.0  # Emissivity [-]

    z_max = 0.6   # Depth of the lower boundary of a heat conduction equation [m]
    n_depth = 41  # Number of depth steps
    Δz = z_max / (n_depth - 1)  # Depth step width [m]

    thermo_params = AsteroidThermoPhysicalModels.ThermoParams(P, l, Γ, R_vis, R_ir, ε, z_max, Δz, n_depth)

    ##= Setting of TPM =##
    stpm = AsteroidThermoPhysicalModels.SingleTPM(shape, thermo_params;
        SELF_SHADOWING = true,
        SELF_HEATING   = false,
        SOLVER         = AsteroidThermoPhysicalModels.ForwardEulerSolver(thermo_params),
        BC_UPPER       = AsteroidThermoPhysicalModels.RadiationBoundaryCondition(),
        BC_LOWER       = AsteroidThermoPhysicalModels.InsulationBoundaryCondition(),
    )
    AsteroidThermoPhysicalModels.init_temperature!(stpm, 0)

    ##= Run TPM =##
    times_to_save = ephem.time[end-n_step_in_cycle:end]  # Save temperature during the final rotation
    face_ID = [1, 2]  # Face indices to save subsurface temperature

    result = AsteroidThermoPhysicalModels.run_TPM!(stpm, ephem, times_to_save, face_ID)
    
    ##= Save TPM result =##
    @testset "Save TPM result" begin
        AsteroidThermoPhysicalModels.export_TPM_results(DIR_OUTPUT, result)

        @test isfile(joinpath(DIR_OUTPUT, "physical_quantities.csv"))
        @test isfile(joinpath(DIR_OUTPUT, "subsurface_temperature.csv"))
        @test isfile(joinpath(DIR_OUTPUT, "surface_temperature.csv"))
        @test isfile(joinpath(DIR_OUTPUT, "thermal_force.csv"))
    end
end
