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
        "PopulateScalePlaintext.cpp",
    ],
    hdrs = ["PopulateScale.h"],
    deps = [
        ":PopulateScalePatterns",
        ":pass_inc_gen",
        "@heir//lib/Analysis/ScaleAnalysis",
        "@heir//lib/Analysis/SecretnessAnalysis",
        "@heir//lib/Dialect:ModuleAttributes",
        "@heir//lib/Dialect/BGV/IR:Dialect",
        "@heir//lib/Dialect/CKKS/IR:Dialect",
        "@heir//lib/Dialect/Mgmt/Transforms:AnnotateMgmt",
        "@heir//lib/Parameters:PlaintextParams",
        "@heir//lib/Parameters/BGV:Params",
        "@heir//lib/Parameters/CKKS:Params",
        "@heir//lib/Utils:APIntUtils",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "PopulateScalePatterns",
    srcs = [
        "PopulateScalePatterns.cpp",
    ],
    hdrs = ["PopulateScalePatterns.h"],
    deps = [
        "@heir//lib/Dialect/Mgmt/IR:Dialect",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

add_heir_transforms(
    generated_target_name = "pass_inc_gen",
    pass_name = "PopulateScale",
    td_file = "PopulateScale.td",
)
