# See https://github.com/Astroshaper/Astroshaper-examples/tree/main/TPM_Didymos for more information.
@testset "TPM_Didymos" begin
    DIR_OUTPUT = joinpath(@__DIR__, "output")
    rm(DIR_OUTPUT; recursive=true, force=true)
    mkpath(DIR_OUTPUT)

    msg = """\n
    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    |                   Test: TPM_Didymos                    |
    ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    """
    println(msg)

    ##= SPICE kernels =##
    paths_kernel = [
        "fk/hera_v10.tf",
        "lsk/naif0012.tls",
        "pck/hera_didymos_v06.tpc",
        "spk/de432s.bsp",
        "spk/didymos_hor_000101_500101_v01.bsp",
        "spk/didymos_gmv_260901_311001_v01.bsp",
    ]

    ##= Shape models =##
    paths_shape = [
        "g_50677mm_rad_obj_didy_0000n00000_v001.obj",
        "g_08438mm_lgt_obj_dimo_0000n00000_v002.obj",
    ]

    ##= Download SPICE kernels =##
    for path_kernel in paths_kernel
        url_kernel = "https://s2e2.cosmos.esa.int/bitbucket/projects/SPICE_KERNELS/repos/hera/raw/kernels/$(path_kernel)?at=refs%2Ftags%2Fv161_20230929_001"
        filepath = joinpath("kernel", path_kernel)
        mkpath(dirname(filepath))
        isfile(filepath) || Downloads.download(url_kernel, filepath)
    end

    ##= Download shape models =##
    for path_shape in paths_shape
        url_kernel = "https://s2e2.cosmos.esa.int/bitbucket/projects/SPICE_KERNELS/repos/hera/raw/kernels/dsk/$(path_shape)?at=refs%2Ftags%2Fv161_20230929_001"
        filepath = joinpath("shape", path_shape)
        mkpath(dirname(filepath))
        isfile(filepath) || Downloads.download(url_kernel, filepath)
    end

    ##= Load the SPICE kernels =##
    for path_kernel in paths_kernel
        filepath = joinpath("kernel", path_kernel)
        SPICE.furnsh(filepath)
    end

    ##= Ephemerides =##
    P₁ = SPICE.convrt(2.2593, "hours", "seconds")  # Rotation period of Didymos
    P₂ = SPICE.convrt(11.93 , "hours", "seconds")  # Rotation period of Dimorphos

    n_cycle = 2  # Number of cycles to perform TPM
    n_step_in_cycle = 72  # Number of time steps in one rotation period

    et_begin = SPICE.utc2et("2027-02-18T00:00:00")  # Start time of TPM
    et_end   = et_begin + P₂ * n_cycle  # End time of TPM
    et_range = range(et_begin, et_end; length=n_step_in_cycle*n_cycle+1)

    """
    - `time` : Ephemeris times
    - `sun1` : Sun's position in the primary's frame (DIDYMOS_FIXED)
    - `sun2` : Sun's position in the secondary's frame (DIMORPHOS_FIXED)
    - `sec`  : Secondary's position in the primary's frame (DIDYMOS_FIXED)
    - `P2S`  : Rotation matrix from primary to secondary frames
    - `S2P`  : Rotation matrix from secondary to primary frames
    """
    ephem = (
        time = collect(et_range),
        sun1 = [SVector{3}(SPICE.spkpos("SUN"      , et, "DIDYMOS_FIXED"  , "None", "DIDYMOS"  )[1]) * 1000 for et in et_range],
        sun2 = [SVector{3}(SPICE.spkpos("SUN"      , et, "DIMORPHOS_FIXED", "None", "DIMORPHOS")[1]) * 1000 for et in et_range],
        sec  = [SVector{3}(SPICE.spkpos("DIMORPHOS", et, "DIDYMOS_FIXED"  , "None", "DIDYMOS"  )[1]) * 1000 for et in et_range],
        P2S  = [RotMatrix{3}(SPICE.pxform("DIDYMOS_FIXED"  , "DIMORPHOS_FIXED", et)) for et in et_range],
        S2P  = [RotMatrix{3}(SPICE.pxform("DIMORPHOS_FIXED", "DIDYMOS_FIXED"  , et)) for et in et_range],
    )

    SPICE.kclear()

    ##= Load the shape models =##
    path_shape1_obj = joinpath("shape", "g_50677mm_rad_obj_didy_0000n00000_v001.obj")
    path_shape2_obj = joinpath("shape", "g_08438mm_lgt_obj_dimo_0000n00000_v002.obj")
    
    shape1 = AsteroidThermoPhysicalModels.load_shape_obj(path_shape1_obj; scale=1000, find_visible_facets=true)
    shape2 = AsteroidThermoPhysicalModels.load_shape_obj(path_shape2_obj; scale=1000, find_visible_facets=true)

    n_face_shape1 = length(shape1.faces)  # Number of faces of Didymos
    n_face_shape2 = length(shape2.faces)  # Number of faces of Dimorphos
    
    ##= Thermal properties of Didymos & Dimorphos [c.f. Michel+2016; Naidu+2020] =##
    k  = 0.125   # Thermal conductivity [W/m/K]
    ρ  = 2170.0  # Density [kg/m³]
    Cₚ = 600.0   # Heat capacity [J/kg/K]

    l₁ = AsteroidThermoPhysicalModels.thermal_skin_depth(P₁, k, ρ, Cₚ)  # Thermal skin depth for Didymos
    l₂ = AsteroidThermoPhysicalModels.thermal_skin_depth(P₂, k, ρ, Cₚ)  # Thermal skin depth for Dimorphos
    Γ = AsteroidThermoPhysicalModels.thermal_inertia(k, ρ, Cₚ)          # Thermal inertia for Didymos and Dimorphos [tiu]

    R_vis = 0.059  # Reflectance in visible light [-]
    R_ir  = 0.0    # Reflectance in thermal infrared [-]
    ε     = 0.9    # Emissivity [-]

    z_max = 0.6   # Depth of the lower boundary of a heat conduction equation [m]
    n_depth = 41  # Number of depth steps
    Δz = z_max / (n_depth - 1)  # Depth step width [m]

    thermo_params1 = AsteroidThermoPhysicalModels.ThermoParams(P₁, l₁, Γ, R_vis, R_ir, ε, z_max, Δz, n_depth)
    thermo_params2 = AsteroidThermoPhysicalModels.ThermoParams(P₂, l₂, Γ, R_vis, R_ir, ε, z_max, Δz, n_depth)

    ##= Setting of TPM =##
    stpm1 = AsteroidThermoPhysicalModels.SingleTPM(shape1, thermo_params1;
        SELF_SHADOWING = true,
        SELF_HEATING   = true,
        SOLVER         = AsteroidThermoPhysicalModels.ForwardEulerSolver(thermo_params1),
        BC_UPPER       = AsteroidThermoPhysicalModels.RadiationBoundaryCondition(),
        BC_LOWER       = AsteroidThermoPhysicalModels.InsulationBoundaryCondition(),
    )

    stpm2 = AsteroidThermoPhysicalModels.SingleTPM(shape2, thermo_params2;
        SELF_SHADOWING = true,
        SELF_HEATING   = true,
        SOLVER         = AsteroidThermoPhysicalModels.ForwardEulerSolver(thermo_params2),
        BC_UPPER       = AsteroidThermoPhysicalModels.RadiationBoundaryCondition(),
        BC_LOWER       = AsteroidThermoPhysicalModels.InsulationBoundaryCondition(),
    )

    btpm  = AsteroidThermoPhysicalModels.BinaryTPM(stpm1, stpm2; MUTUAL_SHADOWING=true, MUTUAL_HEATING=true)
    AsteroidThermoPhysicalModels.init_temperature!(btpm, 200.)
    
    ##= Run TPM =##
    times_to_save = ephem.time[end-n_step_in_cycle:end]  # Save temperature during the final rotation
    face_ID_pri = [1, 2, 3, 4, 10]  # Face indices to save subsurface temperature of the primary
    face_ID_sec = [1, 2, 3, 4, 20]  # Face indices to save subsurface temperature of the secondary

    result = AsteroidThermoPhysicalModels.run_TPM!(btpm, ephem, times_to_save, face_ID_pri, face_ID_sec)

    ##= Save TPM result =##
    @testset "Save TPM result" begin
        AsteroidThermoPhysicalModels.export_TPM_results(DIR_OUTPUT, result)

        @test isfile(joinpath(DIR_OUTPUT, "pri", "physical_quantities.csv"))
        @test isfile(joinpath(DIR_OUTPUT, "pri", "subsurface_temperature.csv"))
        @test isfile(joinpath(DIR_OUTPUT, "pri", "surface_temperature.csv"))
        @test isfile(joinpath(DIR_OUTPUT, "pri", "thermal_force.csv"))

        @test isfile(joinpath(DIR_OUTPUT, "sec", "physical_quantities.csv"))
        @test isfile(joinpath(DIR_OUTPUT, "sec", "subsurface_temperature.csv"))
        @test isfile(joinpath(DIR_OUTPUT, "sec", "surface_temperature.csv"))
        @test isfile(joinpath(DIR_OUTPUT, "sec", "thermal_force.csv"))
    end
end
