@testset "create_constraints" begin
    mesh_models = filter(model -> !isempty(peinfo(model).nodes), MODELS)
    core_of(PEinfo) = EMP._create_variables(EMP.EMC.CollocationExaCore(PEinfo.nodes, PEinfo.K), PEinfo)
    function residual(core)
        model = EMP.ExaModels.ExaModel(core)
        c = similar(model.meta.x0, model.meta.ncon)
        EMP.ExaModels.NLPModels.cons!(model, model.meta.x0, c)
        return c
    end

    paths = Set{Symbol}()
    @testset "assembles the constraints of $model" for model in mesh_models
        PEinfo = peinfo(model)
        Nz, Nc = EMP._get_Nz(PEinfo), EMP._get_Nc(PEinfo)
        N, K = length(PEinfo.nodes[1]) - 1, PEinfo.K
        Ncv, Nss = EMP._get_Ncv(PEinfo), EMP._get_Nss(PEinfo)
        core = core_of(PEinfo)
        forms, _ = EMP._analyze_rhs(core, PEinfo)
        push!(paths, length(forms) + 1 < Nz ? :grouped : :perstate)
        core = EMP._create_constraints(core, PEinfo)

        collocation, continuity, ic, cv = Nz * Nc * N * K, Nz * Nc * (N - 1), Nz * Nc, Ncv * Nc
        @test core.ncon == collocation + continuity + ic + cv + Nz * Nss

        c, scale, from = residual(core), maximum(abs, PEinfo.z0), 0
        @test maximum(abs, c[from+1:from+collocation]) <= 1e-3 * scale
        from += collocation
        @test maximum(abs, c[from+1:from+continuity]) <= 1e-10 * scale
        from += continuity
        @test maximum(abs, c[from+1:from+ic]) <= 1e-8 * scale
        from += ic
        @test cv == 0 || all(iszero, c[from+1:from+cv])
    end
    @test paths == Set([:grouped, :perstate])

    @testset "the right-hand side of $model matches the ODE problem" for model in mesh_models
        PEinfo = peinfo(model)
        Nz, Nc = EMP._get_Nz(PEinfo), EMP._get_Nc(PEinfo)
        N = length(PEinfo.nodes[1]) - 1
        f, u = EMP._get_f(PEinfo), EMP._get_u(PEinfo)
        cvfixed = EMP._get_cvfixed(PEinfo, PEinfo.conditions)
        u_ids = EMP._get_u_ids(PEinfo)

        for cidx in 1:Nc, i in unique([1, cld(N, 2)])
            ssidx = PEinfo.preeq_idxs[cidx]
            op = EMP._get_op(PEinfo, PEinfo.theta0, PEinfo.conditions[cidx], ssidx == 0 ? nothing : PEinfo.zss0[ssidx])
            key = Dict(EMP._get_id(symbol) => symbol for symbol in keys(op))
            for (uidx, id) in enumerate(u_ids)
                op[key[id]] = u[uidx,cidx,i]
            end
            t, z = PEinfo.nodes[cidx][i], PEinfo.z0[:,cidx,i,1]
            prob = EMP.ODE.ODEProblem(PEinfo.model.sys, op, (0.0, 1.0); build_initializeprob = false)
            du = similar(z)
            prob.f(du, z, prob.p, t)
            ours = [f[v](PEinfo.theta0, z, PEinfo.cv0[:,cidx], cvfixed[:,cidx], u[:,cidx,i], t) for v in 1:Nz]
            @test ours ≈ du rtol = 1e-8
        end
    end

    @testset "the condition targets of $model" for model in mesh_models
        PEinfo = peinfo(model)
        Nc = EMP._get_Nc(PEinfo)
        conditions_table = EMP._read_tsv(EMP._read_yaml(find_yaml(model)).conditions)
        cv_ids, cvfixed_ids = EMP._get_cv_ids(PEinfo), EMP._get_cvfixed_ids(PEinfo)
        target_ids = unique(id for condition in [PEinfo.conditions; PEinfo.preeq_conditions] for id in condition.target_ids)

        @test sort([cv_ids; cvfixed_ids]) == sort(target_ids)
        @test isempty(intersect(cv_ids, cvfixed_ids))

        cvfixed = EMP._get_cvfixed(PEinfo, PEinfo.conditions)
        @test size(cvfixed) == (length(cvfixed_ids) + length(EMP._get_ifelses(PEinfo)), Nc)
        row(cidx) = EMP._get_index(PEinfo.conditions[cidx].condition_id, conditions_table.conditionId)
        @test all(
            cvfixed[cvfixedidx,cidx] == something(tryparse(Float64, conditions_table[Symbol(id)][row(cidx)]), cvfixed[cvfixedidx,cidx])
            for (cvfixedidx, id) in enumerate(cvfixed_ids), cidx in 1:Nc
        )
    end

    @testset "fills an empty condition cell with the model default" begin
        function blank(condition_id, id)
            return function (content)
                lines = String.(filter(!isempty, split(content, '\n')))
                icol = EMP._get_index(id, split(lines[1], '\t'))
                irow = findfirst(line -> first(split(line, '\t')) == condition_id, lines)
                cells = String.(split(lines[irow], '\t'))
                cells[icol] = ""
                lines[irow] = join(cells, '\t')
                return join(lines, '\n')
            end
        end
        cidx_of(petab, condition_id) = EMP._get_index(condition_id, [condition.condition_id for condition in petab.conditions])

        petab = EMP._parse_yaml(revise_model("Armistead_CellDeathDis2024", :conditions, blank("mutant", "S_on")))
        cidx = cidx_of(petab, "mutant")
        @test !("S_on" in petab.conditions[cidx].target_ids)
        cvfixed = EMP._get_cvfixed(petab, petab.conditions)
        @test cvfixed[EMP._get_index("S_on", EMP._get_cvfixed_ids(petab)),cidx] == EMP._get_default(petab.model, "S_on") == 1.0

        petab = EMP._parse_yaml(revise_model("Bruno_JExpBot2016", :conditions, blank("model1_data2", "init_bcar")))
        cidx = cidx_of(petab, "model1_data2")
        @test "init_bcar" in EMP._get_cv_ids(petab)
        cv = EMP._get_cv(petab, EMP._get_theta0(petab))
        @test cv[EMP._get_index("init_bcar", EMP._get_cv_ids(petab)),cidx] == EMP._get_default(petab.model, "init_bcar")
    end

    @testset "the event values of $model" for model in filter(model -> !isempty(peinfo(model).events), mesh_models)
        PEinfo = peinfo(model)
        Nc, N = EMP._get_Nc(PEinfo), length(peinfo(model).nodes[1]) - 1
        u, u_ids = EMP._get_u(PEinfo), EMP._get_u_ids(PEinfo)

        @test all(
            u[uidx,cidx,i] == u[uidx,cidx,i+1] || PEinfo.nodes[cidx][i+1] in PEinfo.event_times[:,cidx]
            for uidx in eachindex(u_ids), cidx in 1:Nc, i in 1:(N - 1)
        )
        start(condition, id) = id in condition.target_ids ?
            condition.target_values[EMP._get_index(id, condition.target_ids)] : EMP._get_default(PEinfo.model, id)
        @test all(
            u[uidx,cidx,1] == start(PEinfo.conditions[cidx], id)
            for (uidx, id) in enumerate(u_ids), cidx in 1:Nc if !any(iszero, PEinfo.event_times[:,cidx])
        )
    end

    @testset "rewrites an ifelse as a max or min" begin
        EMP.Symbolics.@variables x c
        unwrap = EMP.Symbolics.unwrap
        value(expr, xv, cv) = EMP.SymbolicUtils.unwrap_const(EMP.Symbolics.fixpoint_sub(expr, Dict(x => xv, c => cv); fold = Val(true)))
        for term in (ifelse(c < x, x - c, 0), ifelse(x > c, x - c, 0), ifelse(x < c, x - c, 0), ifelse(x <= c, 2x, x + c))
            rewritten = EMP._get_maxmin(unwrap(term))
            @test EMP.Symbolics.operation(rewritten) in (max, min)
            @test all(value(rewritten, xv, cv) == value(unwrap(term), xv, cv) for xv in (0.0, 3.0), cv in (1.0, 2.0))
        end
        @test EMP.Symbolics.operation(EMP._get_maxmin(unwrap(ifelse(c < x, x - c, 0)))) === max
        @test EMP.Symbolics.operation(EMP._get_maxmin(unwrap(ifelse(x < c, x - c, 0)))) === min
        @test_throws ArgumentError EMP._get_maxmin(unwrap(ifelse(x < c, x, 2c)))
    end

    @testset "groups the right-hand side terms" begin
        EMP.Symbolics.@variables theta[1:5] z[1:5] s
        unwrap = EMP.Symbolics.unwrap

        @test length(EMP._get_terms(z[1] * theta[1] + z[2])) == 2
        @test length(EMP._get_terms(z[1] * theta[1])) == 1
        @test EMP._get_terms(3.0) == [3.0]

        @test EMP._get_leaf(unwrap(theta[3])) == (:theta, 3)
        @test EMP._get_leaf(unwrap(z[2])) == (:z, 2)
        @test EMP._get_leaf(2.5) == (:data, 2.5)
        @test EMP._get_leaf(unwrap(s)) == (:t, 0)
        @test isnothing(EMP._get_leaf(unwrap(z[1] * theta[1])))

        @test EMP._get_form(unwrap(theta[1] * z[2])) == EMP._get_form(unwrap(theta[4] * z[5]))
        @test EMP._get_form(unwrap(theta[1] * z[2])) == EMP._get_form(unwrap(z[2] * theta[1]))
        @test EMP._get_form(unwrap(theta[1] * z[2])) != EMP._get_form(unwrap(theta[1] * theta[2]))

        @test EMP._get_children(unwrap(z[1]^3))[1] === (*)
        @test length(EMP._get_children(unwrap(z[1]^3))[2]) == 3
        @test EMP._get_children(unwrap(z[1]^5))[1] === (^)

        @test EMP._count_leaves(unwrap(z[1])) == 1
        @test EMP._count_leaves(unwrap(z[1] + z[2])) == 2
    end
end
