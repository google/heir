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
        ":ConvertToMac",
        ":pass_inc_gen",
        "@heir//lib/Dialect/ModArith/IR:Dialect",
    ],
)

cc_library(
    name = "ConvertToMac",
    srcs = ["ConvertToMac.cpp"],
    hdrs = ["ConvertToMac.h"],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/ModArith/IR:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

add_heir_transforms(
    generated_target_name = "pass_inc_gen",
    pass_name = "Passes",
    td_file = "Passes.td",
)
