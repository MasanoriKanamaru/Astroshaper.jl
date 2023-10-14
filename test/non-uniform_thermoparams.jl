# The following tests are almost the same as `TPM_Ryugu.jl`.
# The only difference is that the thermophysical properties vary depending on the location of the asteroid.
@testset "non-uniform_thermoparams" begin
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
    P = SPICE.convrt(7.63262, "hours", "seconds")   # Rotation period of Ryugu
    et_begin = SPICE.utc2et("2018-07-01T00:00:00")  # Start time of TPM
    et_end   = et_begin + 2P                        # End time of TPM
    step     = P / 360                              # Time step of TPM, corresponding to 1 deg rotation
    et_range = et_begin : step : et_end
    @show length(et_range)

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
    path_obj = joinpath("shape", "ryugu_test.obj")   # Small model for test
    path_jld = joinpath("shape", "ryugu_test.jld2")  # Small model for test
    # path_obj = joinpath("shape", "SHAPE_SFM_49k_v20180804.obj")
    # path_jld = joinpath("shape", "SHAPE_SFM_49k_v20180804.jld2")
    if isfile(path_jld) && ENABLE_JLD
        shape = AsteroidThermoPhysicalModels.load_shape_jld(path_jld)
    else
        shape = AsteroidThermoPhysicalModels.load_shape_obj(path_obj; scale=1000, find_visible_facets=true)
        AsteroidThermoPhysicalModels.save_shape_jld(path_jld, shape)
    end

    ##= Thermal properties =##
    """
    When thermophysical properties vary from face to face

    - "Northern" hemisphere:
        - Bond albedo          : A_B = 0.04 [-]
        - Thermal conductivity : k   = 0.1  [W/m/K]
        - Emissivity           : ε   = 1.0  [-]
    - "Southern" hemisphere:
        - Bond albedo          : A_B = 0.1  [-]
        - Thermal conductivity : k   = 0.3  [W/m/K]
        - Emissivity           : ε   = 0.9  [-]
    """

    P  = SPICE.convrt(7.63262, "hours", "seconds")
    k  = [r[3] > 0 ? 0.1 : 0.3  for r in shape.face_centers]
    ρ  = 1270.0
    Cₚ = 600.0
    
    l = AsteroidThermoPhysicalModels.thermal_skin_depth(P, k, ρ, Cₚ)
    Γ = AsteroidThermoPhysicalModels.thermal_inertia(k, ρ, Cₚ)

    thermo_params = AsteroidThermoPhysicalModels.thermoparams(
        P       = P,
        l       = l,
        Γ       = Γ,
        A_B     = [r[3] > 0 ? 0.04 : 0.1 for r in shape.face_centers],
        A_TH    = 0.0,
        ε       = [r[3] > 0 ? 1.0 : 0.9  for r in shape.face_centers],
        z_max   = 0.6,
        Nz      = 41,
    )

    ##= Setting of TPM =##
    stpm = AsteroidThermoPhysicalModels.SingleTPM(shape, thermo_params;
        SELF_SHADOWING = true,
        SELF_HEATING   = true,
        SOLVER         = AsteroidThermoPhysicalModels.ForwardEulerSolver(thermo_params),
        BC_UPPER       = AsteroidThermoPhysicalModels.RadiationBoundaryCondition(),
        BC_LOWER       = AsteroidThermoPhysicalModels.InsulationBoundaryCondition(),
    )
    AsteroidThermoPhysicalModels.init_temperature!(stpm, 200)

    ##= Run TPM and save the result =##
    savepath = "non-uniform_thermoparams.jld2"
    AsteroidThermoPhysicalModels.run_TPM!(stpm, ephem, savepath)
end
