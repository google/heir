# Kernel dialect transforms

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
        ":PrepareLinearTransforms",
        ":pass_inc_gen",
        "@heir//lib/Dialect/Kernel/IR:Dialect",
    ],
)

cc_library(
    name = "PrepareLinearTransforms",
    srcs = ["PrepareLinearTransforms.cpp"],
    hdrs = ["PrepareLinearTransforms.h"],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect:ModuleAttributes",
        "@heir//lib/Dialect/Kernel/IR:Dialect",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@heir//lib/Target/CompilationTarget",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

add_heir_transforms(
    header_filename = "Passes.h.inc",
    pass_name = "Kernel",
    td_file = "Passes.td",
)
