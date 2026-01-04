load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "GenerateParam",
    srcs = [
        "GenerateParamBFV.cpp",
        "GenerateParamBGV.cpp",
        "GenerateParamCKKS.cpp",
    ],
    hdrs = ["GenerateParam.h"],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Analysis/DimensionAnalysis",
        "@heir//lib/Analysis/LevelAnalysis",
        "@heir//lib/Analysis/NoiseAnalysis",
        "@heir//lib/Analysis/NoiseAnalysis:Noise",
        "@heir//lib/Analysis/NoiseAnalysis/BFV:NoiseByBoundCoeffModel",
        "@heir//lib/Analysis/NoiseAnalysis/BFV:NoiseByVarianceCoeffModel",
        "@heir//lib/Analysis/NoiseAnalysis/BFV:NoiseCanEmbModel",
        "@heir//lib/Analysis/NoiseAnalysis/BGV:NoiseByBoundCoeffModel",
        "@heir//lib/Analysis/NoiseAnalysis/BGV:NoiseBySymbolCoeffModel",
        "@heir//lib/Analysis/NoiseAnalysis/BGV:NoiseByVarianceCoeffModel",
        "@heir//lib/Analysis/NoiseAnalysis/BGV:NoiseCanEmbModel",
        "@heir//lib/Analysis/RangeAnalysis",
        "@heir//lib/Analysis/SecretnessAnalysis",
        "@heir//lib/Dialect:ModuleAttributes",
        "@heir//lib/Dialect/BGV/IR:Dialect",
        "@heir//lib/Dialect/CKKS/IR:Dialect",
        "@heir//lib/Dialect/Mgmt/IR:MgmtAttributes",
        "@heir//lib/Dialect/Mgmt/IR:MgmtOps",
        "@heir//lib/Dialect/Mgmt/Transforms:AnnotateMgmt",
        "@heir//lib/Dialect/Secret/IR:Dialect",
        "@heir//lib/Parameters/BGV:Params",
        "@heir//lib/Parameters/CKKS:Params",
        "@heir//lib/Utils:LogArithmetic",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
)

add_heir_transforms(
    generated_target_name = "pass_inc_gen",
    pass_name = "GenerateParam",
    td_file = "GenerateParam.td",
)
