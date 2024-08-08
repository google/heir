load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

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
        "@heir//lib/Dialect/BGV/IR:Dialect",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
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
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

gentbl_cc_library(
    name = "pass_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=LWE",
            ],
            "Passes.h.inc",
        ),
        (
            ["-gen-pass-doc"],
            "LWEPasses.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Passes.td",
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)
