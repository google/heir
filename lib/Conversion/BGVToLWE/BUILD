load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "BGVToLWE",
    srcs = ["BGVToLWE.cpp"],
    hdrs = [
        "BGVToLWE.h",
    ],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Conversion:Utils",
        "@heir//lib/Dialect/BGV/IR:Dialect",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:PolynomialDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
    ],
    alwayslink = 1,
)

gentbl_cc_library(
    name = "pass_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=BGVToLWE",
            ],
            "BGVToLWE.h.inc",
        ),
        (
            ["-gen-pass-doc"],
            "BGVToLWE.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "BGVToLWE.td",
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)
