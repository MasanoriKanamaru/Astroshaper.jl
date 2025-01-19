# The following tests are almost the same as `TPM_Ryugu.jl`.
# The only difference is that the thermophysical properties vary depending on the location of the asteroid.
@testset "non-uniform_thermoparams" begin
    DIR_OUTPUT = joinpath(@__DIR__, "output")
    rm(DIR_OUTPUT; recursive=true, force=true)
    mkpath(DIR_OUTPUT)

    msg = """\n
    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    |             Test: non-uniform_thermoparams             |
    ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    """
    println(msg)


    ##= Download Files =##
    paths_kernel = [
        "lsk/naif0012.tls",
        "pck/hyb2_ryugu_shape_v20190328.tpc",
        "fk/hyb2_ryugu_v01.tf",
        "spk/2162173_Ryugu.bsp",
    ]
    paths_shape = [
        "SHAPE_SFM_49k_v20180804.obj",
    ]

    for path_kernel in paths_kernel
        url_kernel = "https://data.darts.isas.jaxa.jp/pub/hayabusa2/spice_bundle/spice_kernels/$(path_kernel)"
        filepath = joinpath("kernel", path_kernel)
        mkpath(dirname(filepath))
        isfile(filepath) || Downloads.download(url_kernel, filepath)
    end

    for path_shape in paths_shape
        url_shape = "https://data.darts.isas.jaxa.jp/pub/hayabusa2/paper/Watanabe_2019/$(path_shape)"
        filepath = joinpath("shape", path_shape)
        mkpath(dirname(filepath))
        isfile(filepath) || Downloads.download(url_shape, filepath)
    end

    ##= Load data with SPICE =##
    for path_kernel in paths_kernel
        filepath = joinpath("kernel", path_kernel)
        SPICE.furnsh(filepath)
    end

    ##= Ephemerides =##
    P = SPICE.convrt(7.63262, "hours", "seconds")  # Rotation period of Ryugu

    n_cycle = 2  # Number of cycles to perform TPM
    n_step_in_cycle = 120  # Number of time steps in one rotation period

    et_begin = SPICE.utc2et("2018-07-01T00:00:00")  # Start time of TPM
    et_end   = et_begin + P * n_cycle  # End time of TPM
    et_range = range(et_begin, et_end; length=n_step_in_cycle*n_cycle+1)

    """
    - `time` : Ephemeris times
    - `sun`  : Sun's position in the RYUGU_FIXED frame
    """
    ephem = (
        time = collect(et_range),
        sun  = [SVector{3}(SPICE.spkpos("SUN", et, "RYUGU_FIXED", "None", "RYUGU")[1]) * 1000 for et in et_range],
    )

    SPICE.kclear()

    ##= Load obj file =##
    path_obj = joinpath("shape", "ryugu_test.obj")  # Small model for test
    # path_obj = joinpath("shape", "SHAPE_SFM_49k_v20180804.obj")
    
    shape = AsteroidThermoPhysicalModels.load_shape_obj(path_obj; scale=1000, find_visible_facets=true)
    n_face = length(shape.faces)  # Number of faces

    ##= Thermal properties =##
    """
    When thermophysical properties vary from face to face

    - "Northern" hemisphere:
        - reflectance for visible light : R_vis = 0.04 [-]
        - Thermal conductivity : k   = 0.1  [W/m/K]
        - Emissivity           : ε   = 1.0  [-]
    - "Southern" hemisphere:
        - reflectance for visible light : R_vis = 0.1  [-]
        - Thermal conductivity : k   = 0.3  [W/m/K]
        - Emissivity           : ε   = 0.9  [-]
    """

    k  = [r[3] > 0 ? 0.1 : 0.3  for r in shape.face_centers]  # Thermal conductivity [W/m/K]
    ρ  = 1270.0  # Density [kg/m³]
    Cₚ = 600.0   # Heat capacity [J/kg/K]
    
    l = AsteroidThermoPhysicalModels.thermal_skin_depth(P, k, ρ, Cₚ)  # Thermal skin depth [m]
    Γ = AsteroidThermoPhysicalModels.thermal_inertia(k, ρ, Cₚ)        # Thermal inertia [tiu]

    R_vis = [r[3] > 0 ? 0.04 : 0.1 for r in shape.face_centers]  # Reflectance in visible light [-]
    R_ir  = [r[3] > 0 ? 1.0 : 0.9  for r in shape.face_centers]  # Reflectance in thermal infrared [-]
    ε     = fill(1.0, n_face)   # Emissivity [-]

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

    ##= Run TPM =##
    times_to_save = ephem.time[end-n_step_in_cycle:end]  # Save temperature during the final rotation
    face_ID = [1, 2, 3, 4, 10]  # Face indices to save subsurface temperature

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
