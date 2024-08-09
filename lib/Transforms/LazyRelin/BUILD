load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "LazyRelin",
    srcs = ["LazyRelin.cpp"],
    hdrs = [
        "LazyRelin.h",
    ],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Analysis/LazyRelinAnalysis",
        "@heir//lib/Dialect/BGV/IR:Dialect",
        "@com_google_ortools//ortools/base",
        "@com_google_ortools//ortools/linear_solver",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",

        # "@llvm-project//mlir:AffineDialect",
        # "@llvm-project//mlir:AffineUtils",
    ],
)
# LazyRelin tablegen and headers.

gentbl_cc_library(
    name = "pass_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=LazyRelin",
            ],
            "LazyRelin.h.inc",
        ),
        (
            ["-gen-pass-doc"],
            "LazyRelinPasses.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "LazyRelin.td",
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)
