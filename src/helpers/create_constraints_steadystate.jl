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
    Nz, Nss, Ntheta = _get_Nz(PEinfo), _get_Nss(PEinfo), _get_Ntheta(PEinfo)

    # Create steady-state constraint iterator, u after every event
    cvfixed, u = _get_cvfixed(PEinfo, PEinfo.preeq_conditions), _get_u_ss(PEinfo)

    # Conservation laws W[ssidx] zss[:,ssidx] = b[ssidx] and the rows of f they do not replace
    W, b, keep_rows = _get_conservation_laws(PEinfo, cvfixed, u)

    # One kernel per term form of the right-hand sides when that is fewer than one per state
    forms, slots, arguments = _analyze_rhs_steadystate(PEinfo, cvfixed, u)
    if length(forms) + 1 < Nz
        rows = [(v, ssidx) for ssidx in 1:Nss for v in keep_rows[ssidx]]
        pos = zeros(Int, Nz, Nss)
        pos[CartesianIndex.(rows)] = eachindex(rows)
        ExaModels.@add_con(core, con, length(rows))

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
                f[v](_colonindex(theta, Ntheta), _colonindex(zss, Nz, ssidx), (), cvfixed_ss, u_ss, 0.0)
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
        for (pos, (ssidx, k, _)) in enumerate(itr_b) for v in 1:_get_Nz(PEinfo) if W[ssidx][k,v] != 0
    ]
    ExaModels.@add_con!(core, con,
        pos => W_skv * zss[v,ssidx]
        for (pos, W_skv, v, ssidx) in itr_W
    )

    return core
end

# ----- helper functions -----

# u[uidx,ssidx]: value of event target u_ids[uidx] after every event of pre-equilibration condition ssidx
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

# Right-hand-side terms grouped by form at the steady states, terms = [(v, js, vs, cvidxs, data[ssidx])]
function _analyze_rhs_steadystate(PEinfo, cvfixed, u)
    arguments = _get_arguments(PEinfo)
    terms = _get_rhs_terms(PEinfo, arguments)
    slots = _get_slots_steadystate(PEinfo, maximum(_count_slots(term) for (v, term) in terms))
    points = [(ssidx,) for ssidx in 1:_get_Nss(PEinfo)]
    return _get_forms(terms, arguments, slots, cvfixed, u, points), slots, arguments
end

# Slot arrays of a steady-state form: zss[v,ssidx], index tuples js, vs, data
function _get_slots_steadystate(PEinfo, n)
    Nz, Nss = _get_Nz(PEinfo), _get_Nss(PEinfo)
    Symbolics.@variables zss[1:Nz, 1:Nss] ssidx::Int js[1:n]::Int vs[1:n]::Int data[1:n]
    return (; z = zss, zidxs = (ssidx,), js, vs, data)
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

# W[ssidx], b[ssidx], keep_rows[ssidx]: left null space of the stoichiometry alive in condition ssidx, b = W z(t=0), rows of f kept
function _get_conservation_laws(PEinfo, cvfixed_ss, u_ss)
    Nz = _get_Nz(PEinfo)
    arguments = _get_arguments(PEinfo)
    zic_sym = _get_zic_sym(PEinfo, arguments)
    N, monomials = _get_stoichiometry(_get_rhs_terms(PEinfo, arguments), Nz)
    isdead = _get_isdead(monomials)
    laws = Dict{BitVector, Tuple{Matrix{Float64}, Vector{Int}}}()
    W, b, keep_rows = Matrix{Float64}[], Vector{Float64}[], Vector{Int}[]
    for ssidx in 1:_get_Nss(PEinfo)
        zic = _get_zic(arguments, zic_sym, cvfixed_ss[:,ssidx])
        dead = _get_dead_monomials(N, isdead, cvfixed_ss[:,ssidx], u_ss[:,ssidx], zic)
        W_s, keep_rows_s = get!(() -> _get_left_nullspace(N[:, .!dead]), laws, dead)
        push!(W, W_s)
        push!(b, W_s * zic)
        push!(keep_rows, keep_rows_s)
    end
    return W, b, keep_rows
end

# N[v,j]: coefficient of monomial j in the right-hand side of state v, and one term of every monomial
function _get_stoichiometry(terms, Nz)
    index, monomials = Dict{String, Int}(), []
    rows, cols, vals = Int[], Int[], Float64[]
    for (v, term) in terms
        coefficient, key = _get_coefficient(term)
        iszero(coefficient) && continue
        j = get!(() -> (push!(monomials, term); length(monomials)), index, key)
        push!(rows, v)
        push!(cols, j)
        push!(vals, coefficient)
    end
    return SparseArrays.sparse(rows, cols, vals, Nz, length(monomials)), monomials
end

# Numeric coefficient of a term and the key of its monomial
function _get_coefficient(term)
    leaf = _get_leaf(term)
    isnothing(leaf) || return leaf[1] === :data ? (Float64(leaf[2]), "1") : (1.0, string(term))
    op, args = _get_children(term)
    if op === (/)
        coefficient, key = _get_coefficient(args[1])
        return coefficient, string(key, "/", args[2])
    end
    op === (*) || return 1.0, string(term)
    isnumber(arg) = SymbolicUtils.unwrap_const(arg) isa Number
    coefficient = prod(Float64(SymbolicUtils.unwrap_const(arg)) for arg in args if isnumber(arg); init = 1.0)
    return coefficient, join(sort!([string(arg) for arg in args if !isnumber(arg)]), "*")
end

# dead[j]: monomial j is zero in a condition by its data values and the absent states, a state is absent when z(t=0) = 0 and every monomial of its row is dead
function _get_dead_monomials(N, isdead, cvfixed_s, u_s, zic)
    Nt = SparseArrays.sparse(transpose(N))
    absent = iszero.(zic)
    while true
        dead = BitVector(isdead(j, cvfixed_s, u_s, absent) for j in axes(N, 2))
        still = BitVector(
            absent[v] && all(dead[j] for j in SparseArrays.rowvals(Nt)[SparseArrays.nzrange(Nt, v)])
            for v in axes(N, 1)
        )
        still == absent && return dead
        absent = still
    end
end

# isdead(j, cvfixed_s, u_s, absent): monomial j is zero at the data values with the absent states at 0, cached by its own leaves
function _get_isdead(monomials)
    leaves = [
        [_get_leaf(x) => x for x in Symbolics.get_variables(monomial) if _get_leaf(x)[1] in (:cvfixed, :u, :z)]
        for monomial in monomials
    ]
    caches = [Dict{Vector{Float64}, Bool}() for _ in monomials]
    value((kind, index), cvfixed_s, u_s, absent) =
        kind === :cvfixed ? cvfixed_s[index] :
        kind === :u       ? u_s[index]       :
        absent[index]     ? 0.0 : NaN
    function isdead(j, cvfixed_s, u_s, absent)
        isempty(leaves[j]) && return false
        values = [value(leaf, cvfixed_s, u_s, absent) for (leaf, x) in leaves[j]]
        return get!(caches[j], values) do
            rules = Dict(x => v for ((leaf, x), v) in zip(leaves[j], values) if !isnan(v))
            result = Symbolics.value(Symbolics.substitute(monomials[j], rules))
            result isa Number && iszero(result)
        end
    end
    return isdead
end

# W: left null space of N by sparse QR with unit-scaled columns, keep_rows: rows not replaced by a law
function _get_left_nullspace(N)
    Nz = size(N, 1)
    F = SparseArrays.qr(N * LinearAlgebra.Diagonal(1 ./ vec(maximum(abs, N; dims = 1))))
    r = Nz - LinearAlgebra.rank(F)
    r == 0 && return zeros(0, Nz), collect(1:Nz)
    W = zeros(r, Nz)
    Q = F.Q * Matrix{Float64}(LinearAlgebra.I, Nz, Nz)
    W[:, F.prow] = transpose(Q[:, end-r+1:end])
    drop = LinearAlgebra.qr(W, LinearAlgebra.ColumnNorm()).p[1:r]
    return W, setdiff(1:Nz, drop)
end
