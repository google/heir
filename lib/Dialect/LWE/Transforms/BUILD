load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")

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
        ":AddClientInterface",
        ":AddDebugPort",
        ":SetDefaultParameters",
        ":pass_inc_gen",
        "@heir//lib/Dialect/LWE/IR:Dialect",
    ],
)

cc_library(
    name = "AddClientInterface",
    srcs = ["AddClientInterface.cpp"],
    hdrs = [
        "AddClientInterface.h",
    ],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@heir//lib/Dialect/Polynomial/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
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
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FunctionInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "SetDefaultParameters",
    srcs = ["SetDefaultParameters.cpp"],
    hdrs = [
        "SetDefaultParameters.h",
    ],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FunctionInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

add_heir_transforms(
    header_filename = "Passes.h.inc",
    pass_name = "LWE",
    td_file = "Passes.td",
)
