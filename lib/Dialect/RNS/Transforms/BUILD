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
        ":LowerConvertBasis",
        ":pass_inc_gen",
        "@heir//lib/Dialect/RNS/IR:Dialect",
    ],
)

cc_library(
    name = "LowerConvertBasis",
    srcs = ["LowerConvertBasis.cpp"],
    hdrs = ["LowerConvertBasis.h"],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/ModArith/IR:Dialect",
        "@heir//lib/Dialect/RNS/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

add_heir_transforms(
    header_filename = "Passes.h.inc",
    pass_name = "RNS",
    td_file = "Passes.td",
)
