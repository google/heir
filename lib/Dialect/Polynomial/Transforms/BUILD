# Polynomial pass tablegen and headers.

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")
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
        ":NTTRewrites",
        ":pass_inc_gen",
        "@heir//lib/Dialect/Polynomial/IR:Dialect",
    ],
)

cc_library(
    name = "NTTRewrites",
    srcs = ["NTTRewrites.cpp"],
    hdrs = [
        "NTTRewrites.h",
    ],
    deps = [
        ":ntt_rewrites_inc_gen",
        ":pass_inc_gen",
        "@heir//lib/Dialect/ModArith/IR:Dialect",
        "@heir//lib/Dialect/Polynomial/IR:Dialect",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:TransformUtils",
    ],
)

gentbl_cc_library(
    name = "pass_inc_gen",
    tbl_outs = {
        "Passes.h.inc": [
            "-gen-pass-decls",
            "-name=Polynomial",
        ],
        "PolynomialPasses.md": ["-gen-pass-doc"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Passes.td",
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "ntt_rewrites_inc_gen",
    includes = ["../../../.."],
    tbl_outs = {"NTTRewrites.cpp.inc": ["-gen-rewriters"]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "NTTRewrites.td",
    deps = [
        "@heir//lib/Dialect/ModArith/IR:td_files",
        "@heir//lib/Dialect/Polynomial/IR:td_files",
        "@heir//lib/Utils/DRR",
        "@llvm-project//mlir:ArithOpsTdFiles",
    ],
)
