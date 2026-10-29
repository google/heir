load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

add_heir_transforms(
    header_filename = "LWEToCheddarGPU.h.inc",
    pass_name = "LWEToCheddarGPU",
    td_file = "Passes.td",
)

cc_library(
    name = "LWEToCheddarGPU",
    srcs = ["LWEToCheddarGPU.cpp"],
    hdrs = ["LWEToCheddarGPU.h"],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect:ModuleAttributes",
        "@heir//lib/Dialect/CKKS/IR:Dialect",
        "@heir//lib/Dialect/CheddarGPU/IR:Dialect",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@heir//lib/Utils/CheddarRef:params",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
    ],
)
