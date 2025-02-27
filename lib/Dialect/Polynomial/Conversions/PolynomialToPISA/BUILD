load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "PolynomialToPISA",
    srcs = ["PolynomialToPISA.cpp"],
    hdrs = ["PolynomialToPISA.h"],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/ModArith/IR:Dialect",
        "@heir//lib/Dialect/PISA/IR:Dialect",
        "@heir//lib/Dialect/Polynomial/IR:Dialect",
        "@heir//lib/Utils:ConversionUtils",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)

gentbl_cc_library(
    name = "pass_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=PolynomialToPISA",
            ],
            "PolynomialToPISA.h.inc",
        ),
        (
            ["-gen-pass-doc"],
            "PolynomialToPISA.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "PolynomialToPISA.td",
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)
