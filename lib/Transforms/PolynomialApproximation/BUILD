load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "ApproximateRelu",
    srcs = ["ApproximateRelu.cpp"],
    hdrs = ["ApproximateRelu.h"],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Approximation:Sign",
        "@heir//lib/Dialect/PolyExt/IR:Dialect",
        "@heir//lib/Dialect/Polynomial/IR:Dialect",
        "@heir//lib/Dialect/Polynomial/IR:Polynomial",
        "@heir//lib/Dialect/Polynomial/IR:PolynomialAttributes",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:TosaDialect",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
    ],
)

gentbl_cc_library(
    name = "pass_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=PolynomialApproximation",
            ],
            "PolynomialApproximation.h.inc",
        ),
        (
            ["-gen-pass-doc"],
            "PolynomialApproximationPasses.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Passes.td",
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)
