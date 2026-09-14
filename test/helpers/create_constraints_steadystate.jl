@testset "create_constraints_steadystate" begin
    function residual(core)
        nlp = EMP.ExaModels.ExaModel(core)
        c = similar(nlp.meta.x0, nlp.meta.ncon)
        EMP.ExaModels.NLPModels.cons!(nlp, nlp.meta.x0, c)
        return c
    end

    # dfdz(theta, z, cv, cvfixed, u, t): Jacobian of the right-hand side in z, moved here from src
    function dfdz_function(PEinfo)
        arguments = EMP._get_arguments(PEinfo)
        rules = EMP._get_substitutions(PEinfo, arguments)
        rhs = [
            EMP.Symbolics.fixpoint_sub(equation.rhs, rules; fold = EMP.Symbolics.Val(true))
            for equation in EMP.MTK.equations(PEinfo.model.sys)
        ]
        return EMP.Symbolics.build_function(
            EMP.Symbolics.sparsejacobian(rhs, collect(arguments.z)),
            arguments...;
            expression = Val{false},
            nanmath = false
        )[1]
    end

    @testset "the steady-state constraints of $model" for model in filter(model -> EMP._has_zss(peinfo(model)), MODELS)
        PEinfo = peinfo(model)
        Nz, Nss = EMP._get_Nz(PEinfo), EMP._get_Nss(PEinfo)
        cvfixed = EMP._get_cvfixed(PEinfo, PEinfo.preeq_conditions)
        times = EMP._get_event_times(PEinfo.events, PEinfo.preeq_conditions, PEinfo.parameters, PEinfo.model.parametermap)
        u = [
            something(
                EMP._get_u_value(PEinfo, id, times[:,ssidx], Inf),
                EMP._get_u_start(PEinfo, PEinfo.preeq_conditions[ssidx], id)
            )
            for id in EMP._get_u_ids(PEinfo), ssidx in 1:Nss
        ]
        W, b, keep_rows = EMP._get_conservation_laws(PEinfo, cvfixed, u)
        dfdz, f = dfdz_function(PEinfo), EMP._get_f(PEinfo)

        @test length(W) == Nss && length(b) == Nss && length(keep_rows) == Nss
        for ssidx in 1:Nss
            zss = PEinfo.zss0[ssidx]
            @test size(W[ssidx]) == (length(b[ssidx]), Nz)
            @test length(keep_rows[ssidx]) + length(b[ssidx]) == Nz
            @test maximum(abs, W[ssidx] * zss - b[ssidx]) <= 1e-8 * maximum(abs, zss)

            J = dfdz(PEinfo.theta0, zss, Float64[], cvfixed[:,ssidx], u[:,ssidx], 0.0)
            # unit laws are absent states, their rows of J vanish outside the absent columns, other laws are exact
            units = [k for k in axes(W[ssidx], 1) if count(!iszero, W[ssidx][k,:]) == 1]
            absent = [findfirst(!iszero, W[ssidx][k,:]) for k in units]
            others = setdiff(axes(W[ssidx], 1), units)
            @test maximum(abs, W[ssidx][others,:] * J; init = 0.0) <= 1e-6 * maximum(abs, J)
            @test maximum(abs, J[absent, setdiff(1:Nz, absent)]; init = 0.0) <= 1e-6 * maximum(abs, J)
        end

        core = isempty(PEinfo.nodes) ?
            EMP._create_constraints_steadystate(EMP._create_variables_steadystate(EMP.ExaModels.ExaCore(), PEinfo), PEinfo) :
            EMP._create_zss_constraints(EMP._create_variables(EMP.EMC.CollocationExaCore(PEinfo.nodes, PEinfo.K), PEinfo), PEinfo)
        @test core.ncon == Nz * Nss
        @test maximum(abs, residual(core)) <= 1e-4 * (1 + maximum(maximum(abs, zss) for zss in PEinfo.zss0))

        # the kept rows of f at a perturbed point, sorted since the two kernel paths order rows differently
        rows = [(v, ssidx) for ssidx in 1:Nss for v in keep_rows[ssidx]]
        nlp = EMP.ExaModels.ExaModel(core)
        x = nlp.meta.x0 .+ 0.1 .* randn(length(nlp.meta.x0)) .* max.(abs.(nlp.meta.x0), 1.0)
        c = similar(x, nlp.meta.ncon)
        EMP.ExaModels.NLPModels.cons!(nlp, x, c)
        theta, zss = block(x, core.theta), block(x, core.zss)
        expected = [f[v](theta, zss[:,ssidx], (), cvfixed[:,ssidx], u[:,ssidx], 0.0) for (v, ssidx) in rows]
        @test sort(c[1:length(rows)]) ≈ sort(expected) rtol = 1e-10
    end

    @testset "rejects a condition variable with pre-equilibration" begin
        yaml = revise_model("Brannmark_JBC2010", :conditions, content -> replace(content,
            "Dose_001\tinsulin, 0.01 nM\t0.000\t0.010" => "Dose_001\tinsulin, 0.01 nM\t0.000\tk1a"))
        PEinfo = EMP._get_PEtabInfo(yaml)
        @test EMP._has_cv(PEinfo) && EMP._has_zss(PEinfo)
        core = EMP._create_variables(EMP.EMC.CollocationExaCore(PEinfo.nodes, PEinfo.K), PEinfo)
        @test_throws ArgumentError EMP._create_zss_constraints(core, PEinfo)
    end
end
