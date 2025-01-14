load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "CGGIToTfheRust",
    srcs = ["CGGIToTfheRust.cpp"],
    hdrs = [
        "CGGIToTfheRust.h",
    ],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/CGGI/IR:Dialect",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@heir//lib/Dialect/TfheRust/IR:Dialect",
        "@heir//lib/Utils",
        "@heir//lib/Utils:ConversionUtils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
    ],
)

gentbl_cc_library(
    name = "pass_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=CGGIToTfheRust",
            ],
            "CGGIToTfheRust.h.inc",
        ),
        (
            ["-gen-pass-doc"],
            "CGGIToTfheRust.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "CGGIToTfheRust.td",
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)
