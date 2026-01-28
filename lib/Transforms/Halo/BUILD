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
        ":BootstrapLoopIterArgs",
        ":PartialUnrollForLevelConsumption",
        ":ReconcileMixedSecretnessIterArgs",
        ":pass_inc_gen",
        "@heir//lib/Dialect/Mgmt/IR:Dialect",
        "@heir//lib/Dialect/Secret/IR:Dialect",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:SCFDialect",
    ],
)

cc_library(
    name = "Patterns",
    srcs = ["Patterns.cpp"],
    hdrs = [
        "Patterns.h",
    ],
    deps = [
        "@heir//lib/Analysis/LevelAnalysis",
        "@heir//lib/Analysis/SecretnessAnalysis",
        "@heir//lib/Dialect/Mgmt/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineAnalysis",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AffineUtils",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SCFTransforms",
        "@llvm-project//mlir:SCFUtils",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "ReconcileMixedSecretnessIterArgs",
    srcs = ["ReconcileMixedSecretnessIterArgs.cpp"],
    hdrs = [
        "ReconcileMixedSecretnessIterArgs.h",
    ],
    deps = [
        ":Patterns",
        ":pass_inc_gen",
        "@heir//lib/Analysis/SecretnessAnalysis",
        "@heir//lib/Dialect/Secret/IR:Dialect",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "BootstrapLoopIterArgs",
    srcs = ["BootstrapLoopIterArgs.cpp"],
    hdrs = [
        "BootstrapLoopIterArgs.h",
    ],
    deps = [
        ":Patterns",
        ":pass_inc_gen",
        "@heir//lib/Analysis/SecretnessAnalysis",
        "@heir//lib/Dialect/Mgmt/IR:MgmtOps",
        "@heir//lib/Dialect/Secret/IR:Dialect",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "PartialUnrollForLevelConsumption",
    srcs = ["PartialUnrollForLevelConsumption.cpp"],
    hdrs = [
        "PartialUnrollForLevelConsumption.h",
    ],
    deps = [
        ":Patterns",
        ":pass_inc_gen",
        "@heir//lib/Analysis/LevelAnalysis",
        "@heir//lib/Analysis/SecretnessAnalysis",
        "@heir//lib/Dialect/Mgmt/IR:MgmtOps",
        "@heir//lib/Dialect/Secret/IR:Dialect",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

add_heir_transforms(
    generated_target_name = "pass_inc_gen",
    pass_name = "Halo",
    td_file = "Passes.td",
)
