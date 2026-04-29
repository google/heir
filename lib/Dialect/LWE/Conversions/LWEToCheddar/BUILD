load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "LWEToCheddar",
    srcs = ["LWEToCheddar.cpp"],
    hdrs = [
        "LWEToCheddar.h",
    ],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect:ModuleAttributes",
        "@heir//lib/Dialect/CKKS/IR:Dialect",
        "@heir//lib/Dialect/Cheddar/IR:Dialect",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@heir//lib/Utils",
        "@heir//lib/Utils:ConversionUtils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
    ],
)

add_heir_transforms(
    header_filename = "LWEToCheddar.h.inc",
    pass_name = "LWEToCheddar",
    td_file = "LWEToCheddar.td",
)
