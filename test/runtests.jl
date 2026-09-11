using ExaModelsPEtab
using MadNLP
using Test

const EMP = ExaModelsPEtab

include("backends.jl")

const MODELS = [
    "Alkan_SciSignal2018",
    "Armistead_CellDeathDis2024",
    "Bachmann_MSB2011",
    "Bertozzi_PNAS2020",
    "Blasi_CellSystems2016",
    "Boehm_JProteomeRes2014",
    "Borghans_BiophysChem1997",
    "Brannmark_JBC2010",
    "Bruno_JExpBot2016",
    "Chen_MSB2009",
    "Crauste_CellSystems2017",
    "Elowitz_Nature2000",
    "Fiedler_BMCSystBiol2016",
    # "Froehlich_CellSystems2018", # too large
    "Fujita_SciSignal2010",
    "Giordano_Nature2020",
    "Isensee_JCB2018",
    "Lang_PLOSComputBiol2024",
    "Laske_PLOSComputBiol2019",
    "Lucarelli_CellSystems2018",
    "Okuonghae_ChaosSolitonsFractals2020",
    "Perelson_Science1996",
    "Rahman_MBS2016",
    "Raia_CancerResearch2011",
    "Raimundez_PCB2020",
    "SalazarCavazos_MBoC2020",
    "Schwen_PONE2014",
    "Smith_BMCSystBiol2013",
    "Sneyd_PNAS2002",
    "Weber_BMC2015",
    "Zhao_QuantBiol2020",
    "Zheng_PNAS2012",
]

const UNSUPPORTED = Dict(
    "Liu_IFACPapersOnLine2025" => "SBML event on a state variable",
    "Oliveira_NatCommun2021" => "event time is an estimated parameter",
    "Beer_MolBioSystems2014" => "event time maps to an estimated parameter",
)

function find_yaml(model)
    dir = joinpath(@__DIR__, "Benchmark-Models-PEtab", model)
    path = joinpath(dir, model * ".yaml")
    return isfile(path) ? path : only(filter(endswith(".yaml"), readdir(dir; join = true)))
end

const MODELS_SOLVED = [
    "Blasi_CellSystems2016",
    "Bertozzi_PNAS2020",
    "Armistead_CellDeathDis2024",
    "Boehm_JProteomeRes2014",
]

const PEINFO = Dict{String, EMP.PEtabInfo}()
peinfo(model) = get!(() -> EMP._get_PEtabInfo(find_yaml(model)), PEINFO, model)

const EXAMODEL = Dict{String, Any}()
examodel(model) = get!(() -> examodel_petab(find_yaml(model)), EXAMODEL, model)

block(w, variable) = reshape(w[variable.offset .+ (1:variable.length)], EMP.ExaModels.size(variable.size)...)

function revise_model(model, table, edit)
    dir = mktempdir()
    cp(dirname(find_yaml(model)), dir; force = true)
    path = joinpath(dir, basename(EMP._read_yaml(find_yaml(model))[table]))
    write(path, edit(read(path, String)))
    return joinpath(dir, basename(find_yaml(model)))
end

@testset "ExaModelsPEtab" begin
    @testset "helpers" begin
        for file in [
            "structs",
            "utils",
            "get_PEtabInfo",
            "create_variables",
            "create_constraints",
            "create_objective",
            "create_variables_steadystate",
            "create_constraints_steadystate"
        ]
            include("helpers/$file.jl")
        end
    end

    @testset "api" begin
        for file in [
            "examodel_petab",
            "evaluate_objective"
        ]
            include("api/$file.jl")
        end
    end
end
