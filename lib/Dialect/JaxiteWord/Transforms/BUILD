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
        ":JaxiteCkksParameterSelection",
        ":pass_inc_gen",
        "@heir//lib/Dialect/JaxiteWord/IR:Dialect",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "JaxiteCkksParameterSelection",
    srcs = ["JaxiteCkksParameterSelection.cpp"],
    hdrs = ["JaxiteCkksParameterSelection.h"],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/CKKS/IR:Dialect",
        "@heir//lib/Dialect/JaxiteWord/IR:Dialect",
        "@heir//lib/Parameters:RLWEParams",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)

add_heir_transforms(
    header_filename = "Passes.h.inc",
    pass_name = "JaxiteWord",
    td_file = "Passes.td",
)
