load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "PopulateScale",
    srcs = [
        "PopulateScaleBGV.cpp",
        "PopulateScaleCKKS.cpp",
    ],
    hdrs = ["PopulateScale.h"],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Analysis/DimensionAnalysis",
        "@heir//lib/Analysis/LevelAnalysis",
        "@heir//lib/Analysis/NoiseAnalysis",
        "@heir//lib/Analysis/NoiseAnalysis/BGV:NoiseByBoundCoeffModel",
        "@heir//lib/Analysis/NoiseAnalysis/BGV:NoiseByVarianceCoeffModel",
        "@heir//lib/Analysis/ScaleAnalysis",
        "@heir//lib/Analysis/SecretnessAnalysis",
        "@heir//lib/Dialect:ModuleAttributes",
        "@heir//lib/Dialect/BGV/IR:Dialect",
        "@heir//lib/Dialect/Mgmt/IR:MgmtOps",
        "@heir//lib/Dialect/Secret/IR:Dialect",
        "@heir//lib/Utils:APIntUtils",
        "@heir//lib/Utils:AttributeUtils",
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
    pass_name = "PopulateScale",
    td_file = "PopulateScale.td",
)
