# Creates the steady-state / cv auxiliary variable constraints
function _create_constraints_steadystate(
        core::ExaCore,
        PEinfo::PEtabInfo
    )
    core = _create_zss_constraints(core, PEinfo)

    core = _create_cv_constraints(core, PEinfo)

    return core
end

# Create steady-state constraints f(zss) = 0 
# and conservation constraints W zss = W z(t=0) for every pre-equilibration condition
function _create_zss_constraints(
        core::ExaCore,
        PEinfo::PEtabInfo
    )
    _has_cv(PEinfo) && throw(ArgumentError("cv with pre-equilibration is not supported"))

    # Unpack variables and model functions
    zss, theta = core.zss, core.theta
    Nz, Nss = _get_Nz(PEinfo), _get_Nss(PEinfo)

    # Create steady-state constraint iterator, u after every event
    cvfixed, u = _get_cvfixed(PEinfo, PEinfo.preeq_conditions), _get_u_ss(PEinfo)

    # Conservation laws W[ssidx] zss[:,ssidx] = b[ssidx] and the rows of f they do not replace
    W, b, keep_rows = _get_conservation_laws(PEinfo, cvfixed, u)

    # TODO (REVIEW) One kernel per term form of the right-hand sides when that is fewer than one per state
    forms, slots, arguments = _analyze_rhs_steadystate(PEinfo, cvfixed, u)
    if length(forms) + 1 < Nz
        # TODO (REVIEW) Create base rows 0 = 0 of the kept (v, ssidx), the terms of f are added below
        rows = [(v, ssidx) for ssidx in 1:Nss for v in keep_rows[ssidx]]
        pos = zeros(Int, Nz, Nss)
        pos[CartesianIndex.(rows)] = eachindex(rows)
        ExaModels.@add_con(core, con, length(rows))

        # TODO (REVIEW) Add f to the rows of every state containing the term form, one call per form
        for (expr, terms) in forms
            f = Symbolics.build_function(
                expr,
                arguments.theta, slots.z, slots.zidxs..., slots.js, slots.vs, slots.data, arguments.t;
                expression = Val{false},
                nanmath = false
            )
            itr = [
                (pos[v,ssidx], ssidx, js, vs, data[ssidx])
                for (v, js, vs, cvidxs, data) in terms, ssidx in 1:Nss if pos[v,ssidx] > 0
            ]
            ExaModels.@add_con!(core, con,
                row => f(theta, zss, ssidx, js, vs, data_ss, 0.0)
                for (row, ssidx, js, vs, data_ss) in itr
            )
        end
    else
        # Create steady-state constraints
        f = _get_f(PEinfo)
        itr = [
            (ssidx, Tuple(cvfixed[:,ssidx]), Tuple(u[:,ssidx]))
            for ssidx in 1:_get_Nss(PEinfo)
        ]
        for v in 1:_get_Nz(PEinfo)
            itr_v = [row for row in itr if v in keep_rows[row[1]]]
            isempty(itr_v) && continue
            ExaModels.@add_con(core,
                f[v](theta[:], zss[:,ssidx], (), cvfixed_ss, u_ss, 0.0)
                for (ssidx, cvfixed_ss, u_ss) in itr_v
            )
        end
    end

    # Create conservation constraints
    itr_b = [(ssidx, k, b[ssidx][k]) for ssidx in 1:_get_Nss(PEinfo) for k in eachindex(b[ssidx])]
    isempty(itr_b) && return core
    con = ExaModels.@add_con(core,
        -b_sk
        for (ssidx, k, b_sk) in itr_b
    )
    itr_W = [
        (pos, W[ssidx][k,v], v, ssidx)
        for (pos, (ssidx, k, _)) in enumerate(itr_b) for v in 1:_get_Nz(PEinfo)
    ]
    ExaModels.@add_con!(core, con,
        pos => W_skv * zss[v,ssidx]
        for (pos, W_skv, v, ssidx) in itr_W
    )

    return core
end

# ----- helper functions -----

# TODO (REVIEW) u[uidx,ssidx]: value of event target u_ids[uidx] after every event of pre-equilibration condition ssidx
function _get_u_ss(PEinfo)
    times = _get_event_times(PEinfo.events, PEinfo.preeq_conditions, PEinfo.parameters, PEinfo.model.parametermap)
    return [
        something(
            _get_u_value(PEinfo, id, times[:,ssidx], Inf),
            _get_u_start(PEinfo, PEinfo.preeq_conditions[ssidx], id)
        )
        for id in _get_u_ids(PEinfo), ssidx in 1:_get_Nss(PEinfo)
    ]
end

# TODO (REVIEW) Right-hand-side terms grouped by form at the steady states, terms = [(v, js, vs, cvidxs, data[ssidx])]
function _analyze_rhs_steadystate(PEinfo, cvfixed, u)
    arguments = _get_arguments(PEinfo)
    terms = _get_rhs_terms(PEinfo, arguments)
    slots = _get_slots_steadystate(PEinfo, maximum(_count_slots(term) for (v, term) in terms))
    points = [(ssidx,) for ssidx in 1:_get_Nss(PEinfo)]
    return _get_forms(terms, arguments, slots, cvfixed, u, points), slots, arguments
end

# TODO (REVIEW) Slot arrays of a steady-state form: zss[v,ssidx], index tuples js, vs, data
function _get_slots_steadystate(PEinfo, n)
    Nz, Nss = _get_Nz(PEinfo), _get_Nss(PEinfo)
    Symbolics.@variables zss[1:Nz, 1:Nss] ssidx::Int js[1:n]::Int vs[1:n]::Int data[1:n]
    return (; z = zss, zidxs = (ssidx,), js, vs, data)
end

# dfdz(theta, z, cv, cvfixed, u, t): Jacobian of the right-hand side in z
function _get_dfdz(PEinfo)
    arguments = _get_arguments(PEinfo)
    rules = _get_substitutions(PEinfo, arguments)
    rhs = [
        Symbolics.fixpoint_sub(equation.rhs, rules; fold = Val(true)) 
        for equation in MTK.equations(PEinfo.model.sys)
    ]
    return Symbolics.build_function(
        Symbolics.sparsejacobian(rhs, collect(arguments.z)),
        arguments...;
        expression = Val{false},
        nanmath = false
    )[1]
end

# z(t=0) of the pre-equilibration condition with targets cvfixed_s, a number for every state
function _get_zic(arguments, zic_sym, cvfixed_s)
    rules = Dict(
        arguments.cvfixed[cvfixedidx] => value 
        for (cvfixedidx, value) in enumerate(cvfixed_s)
    )
    zic = [
        Symbolics.value(Symbolics.substitute(expr, rules)) 
        for expr in zic_sym
    ]
    all(value -> value isa Number, zic) || throw(
        ArgumentError("pre-equilibration initial state depends on theta, which is not supported")
    )
    return Float64.(zic)
end

# W[ssidx], b[ssidx], keep_rows[ssidx]: left null space of df/dz at (theta0, zss0), b = W z(t=0), rows of f kept
function _get_conservation_laws(PEinfo, cvfixed_ss, u_ss; tol = 1e-12)
    Nz = _get_Nz(PEinfo)
    arguments = _get_arguments(PEinfo)
    zic_sym = _get_zic_sym(PEinfo, arguments)
    dfdz = _get_dfdz(PEinfo)
    W, b, keep_rows = Matrix{Float64}[], Vector{Float64}[], Vector{Int}[]
    for ssidx in 1:_get_Nss(PEinfo)
        J = dfdz(PEinfo.theta0, PEinfo.zss0[ssidx], Float64[], cvfixed_ss[:,ssidx], u_ss[:,ssidx], 0.0)
        F = LinearAlgebra.svd(Matrix(J))
        r = count(<(tol * F.S[1]), F.S)
        W_s = Matrix(transpose(F.U[:,end-r+1:end]))
        drop = LinearAlgebra.qr(W_s, LinearAlgebra.ColumnNorm()).p[1:r]
        push!(W, W_s)
        push!(b, W_s * _get_zic(arguments, zic_sym, cvfixed_ss[:,ssidx]))
        push!(keep_rows, setdiff(1:Nz, drop))
    end
    return W, b, keep_rows
end
