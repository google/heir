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
        ":PolyMulToNTT",
        ":pass_inc_gen",
        "@heir//lib/Dialect/Polynomial/IR:Dialect",
    ],
)

cc_library(
    name = "PolyMulToNTT",
    srcs = [
        "NTTSolver.cpp",
        "PolyMulToNTT.cpp",
    ],
    hdrs = [
        "NTTSolver.h",
        "PolyMulToNTT.h",
    ],
    deps = [
        ":pass_inc_gen",
        "@com_google_ortools//ortools/sat:cp_model",
        "@com_google_ortools//ortools/sat:cp_model_solver",
        "@heir//lib/Dialect/Polynomial/IR:Dialect",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:TensorDialect",
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
