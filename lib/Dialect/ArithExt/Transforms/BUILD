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
        ":pass_inc_gen",
        "@heir//lib/Dialect/ArithExt/IR:Dialect",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "RemToArithExt",
    srcs = ["RemToArithExt.cpp"],
    hdrs = [
        "RemToArithExt.h",
    ],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/ArithExt/IR:Dialect",
        "@heir//lib/Dialect/Polynomial/IR:PolynomialAttributes",
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
                "-name=ArithExt",
            ],
            "Passes.h.inc",
        ),
        (
            ["-gen-pass-doc"],
            "ArithExtPasses.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Passes.td",
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)
