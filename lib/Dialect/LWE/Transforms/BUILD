load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Transforms",
    hdrs = [
        "Passes.h",
    ],
    deps = [
        ":AddDebugPort",
        ":ImplementTrivialEncryptionAsAddition",
        ":pass_inc_gen",
        "@heir//lib/Dialect/LWE/IR:Dialect",
    ],
)

cc_library(
    name = "AddDebugPort",
    srcs = ["AddDebugPort.cpp"],
    hdrs = [
        "AddDebugPort.h",
    ],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/Debug/IR:Dialect",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "ImplementTrivialEncryptionAsAddition",
    srcs = ["ImplementTrivialEncryptionAsAddition.cpp"],
    hdrs = ["ImplementTrivialEncryptionAsAddition.h"],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect:FuncUtils",
        "@heir//lib/Dialect:ModuleAttributes",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@heir//lib/Dialect/Mgmt/IR:Dialect",
        "@heir//lib/Dialect/ModArith/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
    ],
)

add_heir_transforms(
    header_filename = "Passes.h.inc",
    pass_name = "LWE",
    td_file = "Passes.td",
)
