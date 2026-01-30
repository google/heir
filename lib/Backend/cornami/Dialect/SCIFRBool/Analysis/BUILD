load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Analysis",
    hdrs = [
        "Passes.h",
    ],
    deps = [
        ":CGGIEstimator",
        ":DialectResourceMap",
        ":PerfCounter",
        ":pass_inc_gen",
        "@heir//lib/Backend/cornami/Dialect/SCIFRBool/IR:Dialect",
        "@heir//lib/Dialect/CGGI/IR:Dialect",
        "@heir//lib/Dialect/CKKS/IR:Dialect",
        "@heir//lib/Dialect/Openfhe/IR:Dialect",
    ],
)

cc_library(
    name = "CGGIEstimator",
    srcs = [
        "CGGIEstimator.cpp",
    ],
    hdrs = [
        "CGGIEstimator.h",
    ],
    deps = [
        ":PerfCounter",
        ":pass_inc_gen",
        "@heir//lib/Backend/cornami/Dialect/SCIFRBool/Analysis:DialectResourceMap",
        "@heir//lib/Dialect/CGGI/IR:Dialect",
        "@heir//lib/Dialect/CKKS/IR:Dialect",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@heir//lib/Utils/Graph",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
    ],
)

cc_library(
    name = "PerfCounter",
    srcs = [
        "PerfCounter.cpp",
    ],
    hdrs = [
        "PerfCounter.h",
    ],
    deps = [
    ],
)

cc_library(
    name = "DialectResourceMap",
    srcs = [
        "DialectResourceMap.cpp",
    ],
    hdrs = [
        "DialectResourceMap.h",
    ],
    deps = [
        "@heir//lib/Utils/Graph",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
    ],
)

gentbl_cc_library(
    name = "pass_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=SCIFRBoolAnalysis",
            ],
            "Passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Passes.td",
    deps = [
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)
