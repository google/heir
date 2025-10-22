load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "OrionToCKKS",
    srcs = ["OrionToCKKS.cpp"],
    hdrs = ["OrionToCKKS.h"],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/CKKS/IR:Dialect",
        "@heir//lib/Dialect/Orion/IR:Dialect",
        "@heir//lib/Utils:ConversionUtils",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)

add_heir_transforms(
    header_filename = "OrionToCKKS.h.inc",
    pass_name = "OrionToCKKS",
    td_file = "OrionToCKKS.td",
)
