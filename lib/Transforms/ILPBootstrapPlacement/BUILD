load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "ILPBootstrapPlacement",
    srcs = ["ILPBootstrapPlacement.cpp"],
    hdrs = [
        "ILPBootstrapPlacement.h",
    ],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Analysis/ILPBootstrapPlacementAnalysis",
        "@heir//lib/Analysis/SecretnessAnalysis",
        "@heir//lib/Dialect/Mgmt/IR:Dialect",
        "@heir//lib/Dialect/Mgmt/Transforms:AnnotateMgmt",
        "@heir//lib/Dialect/Secret/IR:Dialect",
        "@heir//lib/Transforms/SecretInsertMgmt:Pipeline",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
)

add_heir_transforms(
    generated_target_name = "pass_inc_gen",
    pass_name = "ILPBootstrapPlacement",
)
