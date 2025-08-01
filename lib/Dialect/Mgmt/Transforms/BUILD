load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Transforms",
    hdrs = ["Passes.h"],
    deps = [
        ":AnnotateMgmt",
        ":pass_inc_gen",
        "@heir//lib/Dialect/Mgmt/IR:Dialect",
    ],
)

cc_library(
    name = "Utils",
    srcs = ["Utils.cpp"],
    hdrs = ["Utils.h"],
    deps = [
        "@heir//lib/Dialect:ModuleAttributes",
        "@heir//lib/Dialect/Mgmt/IR:Dialect",
        "@heir//lib/Dialect/Secret/IR:Dialect",
        "@heir//lib/Transforms/PropagateAnnotation",
        "@heir//lib/Utils:AttributeUtils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "AnnotateMgmt",
    srcs = ["AnnotateMgmt.cpp"],
    hdrs = ["AnnotateMgmt.h"],
    deps = [
        ":Utils",
        ":pass_inc_gen",
        "@heir//lib/Analysis/DimensionAnalysis",
        "@heir//lib/Analysis/LevelAnalysis",
        "@heir//lib/Analysis/ScaleAnalysis",
        "@heir//lib/Analysis/SecretnessAnalysis",
        "@heir//lib/Dialect/Mgmt/IR:Dialect",
        "@heir//lib/Dialect/Secret/IR:SecretPatterns",
        "@heir//lib/Utils",
        "@heir//lib/Utils:AttributeUtils",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

add_heir_transforms(
    header_filename = "Passes.h.inc",
    pass_name = "Mgmt",
    td_file = "Passes.td",
)
