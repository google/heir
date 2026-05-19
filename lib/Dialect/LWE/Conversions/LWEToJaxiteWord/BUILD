load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "LWEToJaxiteWord",
    srcs = ["LWEToJaxiteWord.cpp"],
    hdrs = ["LWEToJaxiteWord.h"],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/BGV/IR:Dialect",
        "@heir//lib/Dialect/CKKS/IR:Dialect",
        "@heir//lib/Dialect/JaxiteWord/IR:Dialect",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@heir//lib/Dialect/ModArith/IR:Dialect",
        "@heir//lib/Dialect/RNS/IR:Dialect",
        "@heir//lib/Utils:ConversionUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
)

add_heir_transforms(
    header_filename = "LWEToJaxiteWord.h.inc",
    pass_name = "LWEToJaxiteWord",
    td_file = "LWEToJaxiteWord.td",
)
