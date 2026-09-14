# Creates objective function / y, sigma auxiliary variable constraints
function _create_objective(
        core::ExaModels.ExaCore,
        PEinfo::PEtabInfo
    )
    core = _create_y(core, PEinfo)

    core = _create_sigma(core, PEinfo)

    core = _create_nllh(core, PEinfo)

    core = _create_prior(core, PEinfo)

    return core
end

# Create y (observables at the measurements) and its constraints, y = fy(theta, z, cv, cvfixed)
function _create_y(
        core::ExaModels.ExaCore,
        PEinfo::PEtabInfo
    )
    # Resolve the observable formula of every measurement
    arguments, yvalue = _get_arguments(PEinfo), _get_yvalue()
    exprs = _get_y_exprs(PEinfo, arguments, yvalue)

    # Create zsum for the sums of states inside the formulas
    zsum_sym, zsum0, exprs = _bind_zsum(core, PEinfo, arguments, exprs)

    # Create ExaModels variable
    y0 = _get_starts(PEinfo, arguments, yvalue, exprs, zeros(_get_Nm(PEinfo)), zsum_sym, zsum0)
    ExaModels.@add_var(core,
        y,
        1:_get_Nm(PEinfo);
        start = y0,
    )

    # Create constraints
    core = _create_y_constraints(core, PEinfo, arguments, exprs)

    return core
end

# Create zsum (sums of states inside a denominator, a function, or a large product of an observable) and its constraints, one per sum and measurement point
function _create_zsum(
        core::EMC.CollocationExaCore,
        PEinfo
    )
    # Find the sums, one key per sum and point
    arguments = _get_arguments(PEinfo)
    keys = _get_zsum_keys(PEinfo, arguments)
    index, zsum_sym = _get_zsum_index(keys)

    # Create ExaModels variable
    cvfixed = _get_cvfixed(PEinfo, PEinfo.conditions)
    cache = Dict{Any, Any}()
    zsum0 = Float64[
        get!(cache, sum) do
            Symbolics.build_function(
                sum,
                arguments.theta, arguments.z, arguments.cv, arguments.cvfixed;
                expression = Val{false},
                nanmath = false
            )
        end(PEinfo.theta0, PEinfo.z0[:,cidx,i,k+1], PEinfo.cv0[:,cidx], cvfixed[:,cidx])
        for (sum, cidx, i, k) in keys
    ]
    ExaModels.@add_var(core,
        zsum,
        1:length(keys);
        start = zsum0,
    )
    isempty(keys) && return core

    # Create constraints zsum[q] = sum, one call per term form
    items = [
        (q, _bind_sums(term, false, [], (cidx, i, k), index, zsum_sym), cidx, i, k, PEinfo.nodes[cidx][i + (k > 0)])
        for (q, (sum, cidx, i, k)) in enumerate(keys) for term in _get_terms(sum)
    ]
    ExaModels.@add_con(core, con, zsum[q] for q in 1:length(keys))
    for (f, rows) in _get_form_groups(core, PEinfo, arguments, items)
        core = _create_term_constraints(core, con, f, rows)
    end

    return core
end

# Keys (sum, cidx, i, k) of zsum: the sums bound in the fzic initial conditions and the observable formulas at their points, then inside the terms of every key until no key is new
function _get_zsum_keys(PEinfo, arguments)
    points = _get_measurement_points(PEinfo)
    exprs = _get_y_exprs(PEinfo, arguments, _get_yvalue())
    items = [
        _get_ic_items(PEinfo, arguments);
        [(exprs[m], (measurement.cidx, points[m]...)) for (m, measurement) in enumerate(PEinfo.measurements)]
    ]
    sums = []
    for (expr, point) in items
        _bind_sums(expr, false, sums, point, nothing, _get_zsum(1))
    end
    keys, q = unique(sums), 0
    while q < length(keys)
        q += 1
        for term in _get_terms(keys[q][1])
            _bind_sums(term, false, sums, keys[q][2:end], nothing, _get_zsum(1))
        end
        keys = unique(sums)
    end
    return keys
end

# index[key] => q of zsum[q], and the symbolic zsum
_get_zsum_index(keys) = Dict(key => q for (q, key) in enumerate(keys)), _get_zsum(length(keys))

# exprs with their sums bound to zsum, its symbol and start values
function _bind_zsum(core::EMC.CollocationExaCore, PEinfo, arguments, exprs)
    index, zsum_sym = _get_zsum_index(_get_zsum_keys(PEinfo, arguments))
    points = _get_measurement_points(PEinfo)
    exprs = [
        _bind_sums(exprs[m], false, [], (measurement.cidx, points[m]...), index, zsum_sym)
        for (m, measurement) in enumerate(PEinfo.measurements)
    ]
    zsum0 = Array(core.x0)[core.zsum.offset .+ (1:_get_Nsum(core))]
    return zsum_sym, zsum0, exprs
end

_bind_zsum(core::ExaModels.ExaCore, PEinfo, arguments, exprs) = _get_zsum(0), Float64[], exprs

# Create y constraints, one kernel per form up to 8 leaves and one per term form above
function _create_y_constraints(
        core::EMC.CollocationExaCore,
        PEinfo,
        arguments,
        exprs
    )
    # Unpack variables
    y = core.y

    # Create constraints for the formulas up to 8 leaves
    ms = [m for m in 1:_get_Nm(PEinfo) if _count_leaves(exprs[m]) <= 8]
    for (f, rows) in _get_groups(core, PEinfo, arguments, exprs, ms)
        core = _create_measurement_constraints(core, y, f, rows)
    end

    # Create constraints for the formulas above 8 leaves, y[m] minus its terms
    ms = [m for m in 1:_get_Nm(PEinfo) if _count_leaves(exprs[m]) > 8]
    isempty(ms) && return core
    ExaModels.@add_con(core, con, y[m] for m in ms)
    points = _get_measurement_points(PEinfo)
    items = [
        (row, term, PEinfo.measurements[m].cidx, points[m]..., PEinfo.measurements[m].time)
        for (row, m) in enumerate(ms) for term in _get_y_terms(exprs[m])
    ]
    for (f, rows) in _get_form_groups(core, PEinfo, arguments, items)
        core = _create_term_constraints(core, con, f, rows)
    end

    return core
end

function _create_y_constraints(
        core::ExaModels.ExaCore,
        PEinfo,
        arguments,
        exprs
    )
    for (f, rows) in _get_groups(core, PEinfo, arguments, exprs, 1:_get_Nm(PEinfo))
        core = _create_measurement_constraints(core, core.y, f, rows)
    end

    return core
end

# Create sigma (noise at the measurements) and its constraints, sigma = fsigma(theta, cv, cvfixed, y)
function _create_sigma(
        core::ExaModels.ExaCore,
        PEinfo::PEtabInfo
    )
    # Resolve the noise formula of every measurement, the observable formula substituted by y
    arguments, yvalue = _get_arguments(PEinfo), _get_yvalue()
    sigmas, rows = _get_sigmas(PEinfo, arguments, yvalue)
    maximum(rows) == 0 && return core

    # Unpack variables
    theta, cv, y = core.theta, core.cv, core.y
    Ntheta, Ncv = _get_Ntheta(PEinfo), _get_Ncv(PEinfo)

    # Create sigma iterator, one group per observable and cells
    y0 = Array(core.x0)[y.offset .+ (1:_get_Nm(PEinfo))]
    sigma0 = _get_starts(PEinfo, arguments, yvalue, sigmas, y0)
    cvfixed = _get_cvfixed(PEinfo, PEinfo.conditions)
    groups = [
        (
            _get_fsigma(sigmas[first(ms)], arguments, yvalue),
            [(rows[m], m, PEinfo.measurements[m].cidx, Tuple(cvfixed[:,PEinfo.measurements[m].cidx])) for m in ms]
        )
        for ms in _get_key_groups(PEinfo, [m for m in 1:_get_Nm(PEinfo) if rows[m] > 0])
    ]

    # Create ExaModels variable
    ExaModels.@add_var(core,
        sigma,
        1:maximum(rows);
        start = [sigma0[m] for m in 1:_get_Nm(PEinfo) if rows[m] > 0],
    )

    # Create constraints
    for (fsigma, itr) in groups
        ExaModels.@add_con(core,
            sigma[row] - fsigma(_colonindex(theta, Ntheta), _colonindex(cv, Ncv, cidx), cvfixed_m, (y[m],))
            for (row, m, cidx, cvfixed_m) in itr
        )
    end

    return core
end

# Create the Gaussian negative log-likelihood, residuals on the observableTransformation scale with PEtab.jl's constants
function _create_nllh(
        core::ExaModels.ExaCore,
        PEinfo::PEtabInfo
    )
    # Unpack variables
    y = core.y

    # Resolve sigma of every measurement
    arguments, yvalue = _get_arguments(PEinfo), _get_yvalue()
    sigmas, rows = _get_sigmas(PEinfo, arguments, yvalue)
    values = [Symbolics.value(sigma) for sigma in sigmas]

    # Split the measurements by what sigma is
    itr_fixed = [m for m in 1:_get_Nm(PEinfo) if values[m] isa Number]
    itr_sigma = [m for m in 1:_get_Nm(PEinfo) if rows[m] > 0]
    itr_theta = setdiff(1:_get_Nm(PEinfo), itr_fixed, itr_sigma)

    # Create residuals for sigma = fixed value
    for transform in (:lin, :log, :log10)
        itr = [
            (m, _get_ymeas(PEinfo.measurements[m], transform), Float64(values[m]))
            for m in itr_fixed if _get_transform(PEinfo, m) == transform
        ]
        isempty(itr) && continue
        if transform == :lin
            ExaModels.@add_obj(core, 0.5 * ((y[m] - ymeas) / value)^2 for (m, ymeas, value) in itr)
        elseif transform == :log
            ExaModels.@add_obj(core, 0.5 * ((log(y[m]) - ymeas) / value)^2 for (m, ymeas, value) in itr)
        else
            ExaModels.@add_obj(core, 0.5 * ((log10(y[m]) - ymeas) / value)^2 for (m, ymeas, value) in itr)
        end
    end

    # Create residuals and log(sigma) for sigma = theta expression, one log term per distinct row
    for (f, rows_f) in _get_groups(core, PEinfo, arguments, sigmas, itr_theta)
        core = _create_theta_sigma_objective(core, PEinfo, y, f, rows_f)
    end

    # Create residuals and log(sigma) for sigma = variable
    if !isempty(itr_sigma)
        sigma = core.sigma
        for transform in (:lin, :log, :log10)
            itr = [
                (rows[m], m, _get_ymeas(PEinfo.measurements[m], transform))
                for m in itr_sigma if _get_transform(PEinfo, m) == transform
            ]
            isempty(itr) && continue
            if transform == :lin
                ExaModels.@add_obj(core, 0.5 * ((y[m] - ymeas) / sigma[row])^2 for (row, m, ymeas) in itr)
            elseif transform == :log
                ExaModels.@add_obj(core, 0.5 * ((log(y[m]) - ymeas) / sigma[row])^2 for (row, m, ymeas) in itr)
            else
                ExaModels.@add_obj(core, 0.5 * ((log10(y[m]) - ymeas) / sigma[row])^2 for (row, m, ymeas) in itr)
            end
        end
        ExaModels.@add_obj(core, log(sigma[row]) for row in rows[itr_sigma])
    end

    # Create the measurement constants, log(sigma) of a fixed sigma among them
    constant = sum(
        _get_constant(PEinfo.measurements[m], _get_transform(PEinfo, m)) + (m in itr_fixed ? log(Float64(values[m])) : 0.0)
        for m in 1:_get_Nm(PEinfo)
    )
    ExaModels.@add_obj(core, value for value in [constant])

    return core
end

# Create the negative log prior of every estimated parameter with an objectivePriorType
function _create_prior(
        core::ExaModels.ExaCore,
        PEinfo::PEtabInfo
    )
    # Unpack variables
    theta = core.theta

    # Create prior iterator, (j, parameter) per estimated parameter with a prior
    estimated = [parameter for parameter in PEinfo.parameters if parameter.estimate]
    rows = [(j, parameter) for (j, parameter) in enumerate(estimated) if parameter.prior_type != :none]
    for (j, parameter) in rows
        parameter.prior_type in (:parameterScaleNormal, :parameterScaleLaplace, :normal, :laplace, :uniform) ||
            throw(ArgumentError("objectivePriorType '$(parameter.prior_type)' of '$(parameter.parameter_id)' is not supported"))
    end

    # Create objective for parameterScaleNormal, on theta
    itr = [(j, parameter.prior_parameters[1], parameter.prior_parameters[2]) for (j, parameter) in rows if parameter.prior_type == :parameterScaleNormal]
    if !isempty(itr)
        ExaModels.@add_obj(core,
            0.5 * ((theta[j] - mu) / sd)^2 + log(sd) + 0.5 * log(2pi)
            for (j, mu, sd) in itr
        )
    end

    # Create objective for parameterScaleLaplace, on theta
    itr = [(j, parameter.prior_parameters[1], parameter.prior_parameters[2]) for (j, parameter) in rows if parameter.prior_type == :parameterScaleLaplace]
    if !isempty(itr)
        ExaModels.@add_obj(core,
            abs(theta[j] - mu) / b + log(2 * b)
            for (j, mu, b) in itr
        )
    end

    # Create objective for normal and laplace, on _linscale(theta)
    for scale in (:log10, :log, :lin)
        itr = [(j, parameter.prior_parameters[1], parameter.prior_parameters[2]) for (j, parameter) in rows if parameter.prior_type == :normal && parameter.scale == scale]
        if !isempty(itr)
            ExaModels.@add_obj(core,
                0.5 * ((_linscale(theta[j], scale) - mu) / sd)^2 + log(sd) + 0.5 * log(2pi)
                for (j, mu, sd) in itr
            )
        end
        itr = [(j, parameter.prior_parameters[1], parameter.prior_parameters[2]) for (j, parameter) in rows if parameter.prior_type == :laplace && parameter.scale == scale]
        if !isempty(itr)
            ExaModels.@add_obj(core,
                abs(_linscale(theta[j], scale) - mu) / b + log(2 * b)
                for (j, mu, b) in itr
            )
        end
    end

    # Create objective for uniform, the constant log(ub - lb)
    itr = [(j, parameter.prior_parameters[1], parameter.prior_parameters[2]) for (j, parameter) in rows if parameter.prior_type == :uniform]
    if !isempty(itr)
        ExaModels.@add_obj(core,
            log(ub - lb)
            for (j, lb, ub) in itr
        )
    end

    return core
end

# ----- helper functions -----

_get_Nm(PEinfo::Union{NamedTuple, PEtabInfo}) = length(PEinfo.measurements)
_get_Ny(PEinfo::PEtabInfo) = length(PEinfo.observables)

# Observable and cells of a measurement, the key its formulas are resolved and grouped by
_get_key(measurement::PEtabMeasurement) =
    (measurement.yidx, measurement.observable_parameters, measurement.noise_parameters)

# Measurements of ms with the same key, in the order they first occur
function _get_key_groups(PEinfo, ms)
    groups = Dict{Any, Vector{Int}}()
    for m in ms
        push!(get!(groups, _get_key(PEinfo.measurements[m]), Int[]), m)
    end
    return [groups[key] for key in unique(_get_key(PEinfo.measurements[m]) for m in ms)]
end

# yvalue[1]: the observable of a measurement, standing in for y[m] inside its noise formula
function _get_yvalue()
    Symbolics.@variables yvalue[1:1]
    return yvalue
end

_get_transform(PEinfo, m) = PEinfo.observables[PEinfo.measurements[m].yidx].transform

# zsum[q]: a sum of states inside an observable formula at a measurement point
function _get_zsum(Nsum)
    Symbolics.@variables zsum[1:Nsum]
    return zsum
end

_get_Nsum(core::EMC.CollocationExaCore) = core.zsum.length
_get_Nsum(core::ExaModels.ExaCore) = 0

# Sums with more than 8 leaves in a denominator, a function, or a product with more than 8 other leaves replaced by zsum: collected into sums without an index, else zsum[index[sum, point]]
function _bind_sums(x, under, sums, point, index, zsum)
    isnothing(_get_leaf(x)) || return x
    op, args = SymbolicUtils.operation(x), collect(SymbolicUtils.arguments(x))
    if op === (+) && under && _count_leaves(x) > 8
        isnothing(index) && push!(sums, (x, point...))
        return zsum[isnothing(index) ? 1 : index[(x, point...)]]
    end
    if op === (/)
        den = _bind_sums(args[2], true, sums, point, index, zsum)
        num = _bind_sums(args[1], under || _count_leaves(den) > 8, sums, point, index, zsum)
        return num / den
    end
    unders = op === (*) ? [under || _count_leaves(x) - _count_leaves(a) > 8 for a in args] : fill(op !== (+), length(args))
    return op((_bind_sums(a, under_a, sums, point, index, zsum) for (a, under_a) in zip(args, unders))...)
end

# Top-level terms of an observable formula, a numerator sum distributed over its denominator
function _get_y_terms(expr)
    expr isa Number && return [expr]
    if SymbolicUtils.iscall(expr) && SymbolicUtils.operation(expr) === (/)
        num, den = SymbolicUtils.arguments(expr)
        return [term / den for term in _get_terms(SymbolicUtils.expand(num))]
    end
    terms = _get_terms(SymbolicUtils.expand(expr))
    return length(terms) == 1 ? terms : [t for term in terms for t in _get_y_terms(term)]
end

# Measured value on the observableTransformation scale
_get_ymeas(measurement::PEtabMeasurement, transform) =
    transform == :log   ? log(measurement.measurement)   :
    transform == :log10 ? log10(measurement.measurement) : measurement.measurement

# Measurement constant of the Gaussian negative log-likelihood, PEtab.jl's change of variables
_get_constant(measurement::PEtabMeasurement, transform) =
    0.5 * log(2pi) +
    (transform == :lin ? 0.0 : log(measurement.measurement)) +
    (transform == :log10 ? log(log(10)) : 0.0)

# exprs[m]: observable formula of measurement m resolved into the arguments
_get_y_exprs(PEinfo, arguments, yvalue) =
    _get_exprs(PEinfo, arguments, yvalue, [Meta.parse(observable.observable_formula) for observable in PEinfo.observables])

# exprs[m]: formulas[yidx] of measurement m, its cells and every id resolved into the arguments
function _get_exprs(PEinfo, arguments, yvalue, formulas)
    rules = _get_substitutions(PEinfo, arguments)
    symbols = _get_symbols(PEinfo, arguments)
    haskey(symbols, "yvalue") && throw(ArgumentError("'yvalue' is reserved for the observable inside a noise formula"))
    symbols["yvalue"] = yvalue[1]
    scales = [parameter.scale for parameter in PEinfo.parameters if parameter.estimate]
    value(cell) = cell isa Int ? _linscale(arguments.theta[cell], scales[cell]) : cell
    cache = Dict{Any, Any}()
    return [
        get!(cache, _get_key(measurement)) do
            observable = PEinfo.observables[measurement.yidx]
            cells = Dict{String, Any}(
                "$(prefix)Parameter$(n)_$(observable.observable_id)" => value(cell)
                for (prefix, parameters) in
                    ("observable" => measurement.observable_parameters, "noise" => measurement.noise_parameters)
                for (n, cell) in enumerate(parameters)
            )
            Symbolics.unwrap(
                Symbolics.fixpoint_sub(
                    _parse_formula(formulas[measurement.yidx], merge(symbols, cells)),
                    rules;
                    fold = Val(true)
                )
            )
        end
        for measurement in PEinfo.measurements
    ]
end

# sigmas[m]: noise formula of measurement m with its observable formula substituted by yvalue.
# rows[m]: its sigma row, 0 when sigma reads no observable
function _get_sigmas(PEinfo, arguments, yvalue)
    sigmas = _get_exprs(PEinfo, arguments, yvalue, [
        _substitute_observable(Meta.parse(observable.noise_formula), Meta.parse(observable.observable_formula))
        for observable in PEinfo.observables
    ])
    rows, row = zeros(Int, _get_Nm(PEinfo)), 0
    for (m, sigma) in enumerate(sigmas)
        kinds = _get_kinds(sigma)
        :z in kinds && throw(ArgumentError(
            "noise formula of '$(PEinfo.observables[PEinfo.measurements[m].yidx].observable_id)' reads a state outside its observable formula, which is not supported"
        ))
        :yvalue in kinds && (row += 1; rows[m] = row)
    end
    return sigmas, rows
end

# Every subexpression of expression equal to observable replaced by the yvalue placeholder
_substitute_observable(expression, observable) =
    expression == observable ? :yvalue :
    expression isa Expr ? Expr(expression.head, (_substitute_observable(argument, observable) for argument in expression.args)...) :
    expression

# fy(theta, z, cv, cvfixed): an observable formula
_get_fy(expr, arguments) = Symbolics.build_function(
    expr, arguments.theta, arguments.z, arguments.cv, arguments.cvfixed;
    expression = Val{false},
    nanmath = false
)

# fsigma(theta, cv, cvfixed, yvalue): a noise formula that reads the observable
_get_fsigma(sigma, arguments, yvalue) = Symbolics.build_function(
    sigma, arguments.theta, arguments.cv, arguments.cvfixed, yvalue;
    expression = Val{false},
    nanmath = false
)

# Leaf kinds a resolved expression reads
function _get_kinds(x)
    leaf = _get_leaf(x)
    isnothing(leaf) || return Set([leaf[1]])
    return union((_get_kinds(argument) for argument in SymbolicUtils.arguments(x))...)
end

# Symbol of every formula id: model states, assignment rules and parameters, and parameters-table ids outside the model as theta or values
function _get_symbols(PEinfo, arguments)
    (; sys) = PEinfo.model
    symbols = Dict{String, Any}(
        _get_id(symbol) => symbol
        for symbol in [MTK.unknowns(sys); MTK.parameters(sys); [equation.lhs for equation in MTK.observed(sys)]]
    )
    j = 0
    for parameter in PEinfo.parameters
        parameter.estimate && (j += 1)
        haskey(symbols, parameter.parameter_id) && continue
        symbols[parameter.parameter_id] = parameter.estimate ? _linscale(arguments.theta[j], parameter.scale) : parameter.value
    end
    return symbols
end

# Symbolic expression of a table formula, ids resolved through symbols
_parse_formula(number::Number, symbols) = Symbolics.Num(number)

function _parse_formula(id::Symbol, symbols)
    haskey(symbols, string(id)) || throw(ArgumentError("'$id' is not a model symbol, a parameter, or a placeholder"))
    return symbols[string(id)]
end

function _parse_formula(expression::Expr, symbols)
    expression.head == :call || throw(ArgumentError("formula '$expression' is not supported"))
    return getfield(Base, expression.args[1])((_parse_formula(argument, symbols) for argument in expression.args[2:end])...)
end

# points[m]: (i, k) of the collocation point z[:,cidx,i,k] at the time of measurement m, node 1 is (1, 0), node i + 1 is (i, K), a steady state is (0, 0)
function _get_measurement_points(PEinfo)
    points = Vector{Tuple{Int, Int}}(undef, _get_Nm(PEinfo))
    for (m, measurement) in enumerate(PEinfo.measurements)
        isinf(measurement.time) && (points[m] = (0, 0); continue)
        node = _get_index(measurement.time, PEinfo.nodes[measurement.cidx])
        points[m] = node == 1 ? (1, 0) : (node - 1, PEinfo.K)
    end
    return points
end

# z0_measurements[:,m]: initial guess of the states at measurement m, z0 at its collocation point or zss0 of its condition
function _get_z0_measurements(PEinfo)
    z0_measurements = Matrix{Float64}(undef, _get_Nz(PEinfo), _get_Nm(PEinfo))
    points = _get_measurement_points(PEinfo)
    for (m, measurement) in enumerate(PEinfo.measurements)
        if isinf(measurement.time)
            z0_measurements[:,m] = PEinfo.zss0[PEinfo.preeq_idxs[measurement.cidx]]
        else
            i, k = points[m]
            z0_measurements[:,m] = PEinfo.z0[:,measurement.cidx,i,k+1]
        end
    end
    return z0_measurements
end

# starts[m]: exprs[m] at the initial guess, its observable read from y0 and its sums from zsum0
function _get_starts(PEinfo, arguments, yvalue, exprs, y0, zsum = _get_zsum(0), zsum0 = Float64[])
    z0 = _get_z0_measurements(PEinfo)
    cvfixed = _get_cvfixed(PEinfo, PEinfo.conditions)
    cache = Dict{Any, Any}()
    return Float64[
        get!(cache, exprs[m]) do
            Symbolics.build_function(
                exprs[m],
                arguments.theta, arguments.z, arguments.cv, arguments.cvfixed, yvalue, zsum;
                expression = Val{false},
                nanmath = false
            )
        end(PEinfo.theta0, z0[:,m], PEinfo.cv0[:,measurement.cidx], cvfixed[:,measurement.cidx], (y0[m],), zsum0)
        for (m, measurement) in enumerate(PEinfo.measurements)
    ]
end

# Groups of measurements ms sharing a form: (f, rows) with rows = (m, cidx, i, k, js, vs, cvidxs, qs, data, t)
function _get_groups(core::EMC.CollocationExaCore, PEinfo, arguments, exprs, ms)
    points = _get_measurement_points(PEinfo)
    items = [(m, exprs[m], PEinfo.measurements[m].cidx, points[m]..., PEinfo.measurements[m].time) for m in ms]
    return _get_form_groups(core, PEinfo, arguments, items)
end

# Groups of items (row, expr, cidx, i, k, t) sharing a form: (f, rows) with rows = (row, cidx, i, k, js, vs, cvidxs, qs, data, t)
function _get_form_groups(core, PEinfo, arguments, items)
    isempty(items) && return []
    cvfixed, u = _get_cvfixed(PEinfo, PEinfo.conditions), _get_u(PEinfo)
    slots = _get_slots(core, PEinfo, maximum(_count_slots(expr) for (row, expr, cidx, i, k, t) in items); Nsum = _get_Nsum(core))
    forms, occurrences = Dict{String, Any}(), []
    for (row, expr, cidx, i, k, t) in items
        form = _get_form(expr)
        expr, leaves = _get_expr(expr, arguments, slots)
        haskey(forms, form) || (forms[form] = expr)
        push!(occurrences, (
            form, row, cidx, i, k,
            Tuple(leaves.js), Tuple(leaves.vs), Tuple(leaves.cvidxs), Tuple(leaves.qs),
            Tuple(_get_data(leaf, cvfixed, u, cidx, i) for leaf in leaves.data),
            t,
        ))
    end
    return [
        (
            Symbolics.build_function(
                forms[form],
                arguments.theta, slots.z, slots.cv, slots.zsum, slots.m, slots.i, slots.k,
                slots.js, slots.vs, slots.cvidxs, slots.qs, slots.data, arguments.t;
                expression = Val{false},
                nanmath = false
            ),
            [occurrence[2:end] for occurrence in occurrences if occurrence[1] == form]
        )
        for form in unique(first.(occurrences))
    ]
end

# Groups of measurements ms sharing an observable and cells: (f, rows) with rows = (m, ssidx, cvfixed_m)
# now groups of measurements ms sharing a form at the steady state: (f, rows) with rows = (m, ssidx, js, vs, data)
function _get_groups(core::ExaModels.ExaCore, PEinfo, arguments, exprs, ms)
    isempty(ms) && return []
    cvfixed, u = _get_cvfixed(PEinfo, PEinfo.preeq_conditions), _get_u_ss(PEinfo)
    slots = _get_slots_steadystate(PEinfo, maximum(_count_slots(exprs[m]) for m in ms))
    forms, occurrences = Dict{String, Any}(), []
    for m in ms
        ssidx = PEinfo.preeq_idxs[PEinfo.measurements[m].cidx]
        form = _get_form(exprs[m])
        expr, leaves = _get_expr(exprs[m], arguments, slots)
        haskey(forms, form) || (forms[form] = expr)
        push!(occurrences, (
            form, m, ssidx,
            Tuple(leaves.js), Tuple(leaves.vs), Tuple(_get_data(leaf, cvfixed, u, ssidx) for leaf in leaves.data),
        ))
    end
    return [
        (
            Symbolics.build_function(
                forms[form],
                arguments.theta, slots.z, slots.zidxs..., slots.js, slots.vs, slots.data, arguments.t;
                expression = Val{false},
                nanmath = false
            ),
            [occurrence[2:end] for occurrence in occurrences if occurrence[1] == form]
        )
        for form in unique(first.(occurrences))
    ]
end

# Create constraints variable[m] = f(theta, z, cv, cvfixed) at the collocation point of every measurement of the group
function _create_measurement_constraints(
        core::EMC.CollocationExaCore,
        variable,
        f,
        rows
    )
    # Unpack variables
    z, theta, cv, zsum = core.z, core.theta, core.cv, core.zsum

    # Create constraints
    ExaModels.@add_con(core,
        variable[m] - f(theta, z, cv, zsum, cidx, i, k, js, vs, cvidxs, qs, data_m, t)
        for (m, cidx, i, k, js, vs, cvidxs, qs, data_m, t) in rows
    )

    return core
end

# Create constraint augmentations row => -f(theta, z, cv, zsum) of the terms of the group
function _create_term_constraints(
        core::EMC.CollocationExaCore,
        con,
        f,
        rows
    )
    # Unpack variables
    z, theta, cv, zsum = core.z, core.theta, core.cv, core.zsum

    # Create constraint augmentations
    ExaModels.@add_con!(core, con,
        row => -f(theta, z, cv, zsum, cidx, i, k, js, vs, cvidxs, qs, data_m, t)
        for (row, cidx, i, k, js, vs, cvidxs, qs, data_m, t) in rows
    )

    return core
end

# Create constraints variable[m] = f(theta, zss, (), cvfixed) at the steady state of every measurement of the group
function _create_measurement_constraints(
        core::ExaModels.ExaCore,
        variable,
        f,
        rows
    )
    # Unpack variables
    zss, theta = core.zss, core.theta

    # Create constraints
    ExaModels.@add_con(core,
        variable[m] - f(theta, zss, ssidx, js, vs, data_m, 0.0)
        for (m, ssidx, js, vs, data_m) in rows
    )

    return core
end

# Create the residuals of a sigma form and its log(sigma), one log term per distinct row
function _create_theta_sigma_objective(
        core::EMC.CollocationExaCore,
        PEinfo,
        y,
        f,
        rows
    )
    # Unpack variables
    z, theta, cv, zsum = core.z, core.theta, core.cv, core.zsum

    # Create residuals
    for transform in (:lin, :log, :log10)
        itr = [
            (_get_ymeas(PEinfo.measurements[m], transform), m, cidx, i, k, js, vs, cvidxs, qs, data_m, t)
            for (m, cidx, i, k, js, vs, cvidxs, qs, data_m, t) in rows if _get_transform(PEinfo, m) == transform
        ]
        isempty(itr) && continue
        if transform == :lin
            ExaModels.@add_obj(core,
                0.5 * ((y[m] - ymeas) / f(theta, z, cv, zsum, cidx, i, k, js, vs, cvidxs, qs, data_m, t))^2
                for (ymeas, m, cidx, i, k, js, vs, cvidxs, qs, data_m, t) in itr
            )
        elseif transform == :log
            ExaModels.@add_obj(core,
                0.5 * ((log(y[m]) - ymeas) / f(theta, z, cv, zsum, cidx, i, k, js, vs, cvidxs, qs, data_m, t))^2
                for (ymeas, m, cidx, i, k, js, vs, cvidxs, qs, data_m, t) in itr
            )
        else
            ExaModels.@add_obj(core,
                0.5 * ((log10(y[m]) - ymeas) / f(theta, z, cv, zsum, cidx, i, k, js, vs, cvidxs, qs, data_m, t))^2
                for (ymeas, m, cidx, i, k, js, vs, cvidxs, qs, data_m, t) in itr
            )
        end
    end

    # Create log(sigma), one term per distinct row since sigma reads no state
    counts, first_rows = Dict{Any, Int}(), Dict{Any, Any}()
    for row in rows
        (m, cidx, i, k, js, vs, cvidxs, qs, data_m, t) = row
        key = (cidx, js, vs, cvidxs, qs, data_m)
        counts[key] = get(counts, key, 0) + 1
        get!(first_rows, key, row)
    end
    itr = [
        (counts[key], first_rows[key][2:end]...)
        for key in unique((cidx, js, vs, cvidxs, qs, data_m) for (m, cidx, i, k, js, vs, cvidxs, qs, data_m, t) in rows)
    ]
    ExaModels.@add_obj(core,
        count * log(f(theta, z, cv, zsum, cidx, i, k, js, vs, cvidxs, qs, data_m, t))
        for (count, cidx, i, k, js, vs, cvidxs, qs, data_m, t) in itr
    )

    return core
end

# Create the residuals of a sigma group and its log(sigma) at the steady state
function _create_theta_sigma_objective(
        core::ExaModels.ExaCore,
        PEinfo,
        y,
        f,
        rows
    )
    # Unpack variables
    zss, theta = core.zss, core.theta

    # Create residuals
    for transform in (:lin, :log, :log10)
        itr = [
            (_get_ymeas(PEinfo.measurements[m], transform), m, ssidx, js, vs, data_m)
            for (m, ssidx, js, vs, data_m) in rows if _get_transform(PEinfo, m) == transform
        ]
        isempty(itr) && continue
        if transform == :lin
            ExaModels.@add_obj(core,
                0.5 * ((y[m] - ymeas) / f(theta, zss, ssidx, js, vs, data_m, 0.0))^2
                for (ymeas, m, ssidx, js, vs, data_m) in itr
            )
        elseif transform == :log
            ExaModels.@add_obj(core,
                0.5 * ((log(y[m]) - ymeas) / f(theta, zss, ssidx, js, vs, data_m, 0.0))^2
                for (ymeas, m, ssidx, js, vs, data_m) in itr
            )
        else
            ExaModels.@add_obj(core,
                0.5 * ((log10(y[m]) - ymeas) / f(theta, zss, ssidx, js, vs, data_m, 0.0))^2
                for (ymeas, m, ssidx, js, vs, data_m) in itr
            )
        end
    end

    # Create log(sigma), one term per condition since sigma reads no state
    counts = Dict{Any, Int}()
    for (m, ssidx, js, vs, data_m) in rows
        counts[(ssidx, js, vs, data_m)] = get(counts, (ssidx, js, vs, data_m), 0) + 1
    end
    itr = [(counts[key], key...) for key in unique((ssidx, js, vs, data_m) for (m, ssidx, js, vs, data_m) in rows)]
    ExaModels.@add_obj(core,
        count * log(f(theta, zss, ssidx, js, vs, data_m, 0.0))
        for (count, ssidx, js, vs, data_m) in itr
    )

    return core
end
