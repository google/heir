load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

gentbl_cc_library(
    name = "pass_inc_gen_v2",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=CGGIEmitter",
            ],
            "SCIFRBoolEmitter.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "SCIFRBoolEmitter.td",
    deps = [
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "pass_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=SCIFRBoolEstimationAnalysis",
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

cc_library(
    name = "CGGIEmitter",
    srcs = [
        "SCIFRBoolEmitter.cpp",
    ],
    hdrs = [
        "SCIFRBoolEmitter.h",
    ],
    deps = [
        ":pass_inc_gen",
        ":pass_inc_gen_v2",
        "@heir//lib/Dialect/CGGI/IR:Dialect",
        "@heir//lib/Dialect/CKKS/IR:Dialect",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@heir//lib/Dialect/SCIFRBool/IR:Dialect",
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
    name = "EstimationAnalysis",
    hdrs = [
        "Passes.h",
    ],
    deps = [
        ":CGGIEstimator",
        ":DialectResourceMap",
        ":PerfCounter",
        ":pass_inc_gen",
        ":pass_inc_gen_v2",
        "@heir//lib/Dialect/CGGI/IR:Dialect",
        "@heir//lib/Dialect/CKKS/IR:Dialect",
        "@heir//lib/Dialect/Openfhe/IR:Dialect",
        "@heir//lib/Dialect/SCIFRBool/IR:Dialect",
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
        ":DialectResourceMap",
        ":PerfCounter",
        ":pass_inc_gen",
        ":pass_inc_gen_v2",
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
