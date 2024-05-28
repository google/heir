# Polynomial pass tablegen and headers.

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
        ":NTTRewrites",
        ":pass_inc_gen",
        "@heir//lib/Dialect/Polynomial/IR:Dialect",
        "@llvm-project//mlir:IR",
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
        "@heir//lib/Dialect/Polynomial/IR:Dialect",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:TransformUtils",
    ],
)

gentbl_cc_library(
    name = "pass_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=Polynomial",
            ],
            "Passes.h.inc",
        ),
        (
            ["-gen-pass-doc"],
            "PolynomialPasses.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Passes.td",
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "ntt_rewrites_inc_gen",
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "NTTRewrites.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "NTTRewrites.td",
    deps = [
        "@heir//lib/DRR",
        "@heir//lib/Dialect/Polynomial/IR:ops_inc_gen",
        "@heir//lib/Dialect/Polynomial/IR:td_files",
        "@llvm-project//mlir:ArithOpsTdFiles",
    ],
)
