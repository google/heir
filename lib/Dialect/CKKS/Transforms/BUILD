load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Transforms",
    hdrs = ["Passes.h"],
    deps = [
        ":DecomposeRelinearize",
        ":pass_inc_gen",
        "@heir//lib/Dialect/CKKS/IR:Dialect",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "Patterns",
    srcs = ["Patterns.cpp"],
    hdrs = ["Patterns.h"],
    deps = [
        "@heir//lib/Dialect/CKKS/IR:Dialect",
        "@heir//lib/Utils:AttributeUtils",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "DecomposeRelinearize",
    srcs = ["DecomposeRelinearize.cpp"],
    hdrs = ["DecomposeRelinearize.h"],
    deps = [
        ":Patterns",
        ":pass_inc_gen",
        "@heir//lib/Dialect/CKKS/IR:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)

gentbl_cc_library(
    name = "pass_inc_gen",
    tbl_outs = {
        "Passes.h.inc": [
            "-gen-pass-decls",
            "-name=CKKS",
        ],
        "CKKSPasses.md": ["-gen-pass-doc"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Passes.td",
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)
