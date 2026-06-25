load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "PreprocessingToOpenfhe",
    srcs = ["PreprocessingToOpenfhe.cpp"],
    hdrs = ["PreprocessingToOpenfhe.h"],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Analysis/PreprocessingStorageLayoutAnalysis",
        "@heir//lib/Dialect/Openfhe/IR:Dialect",
        "@heir//lib/Dialect/Preprocessing/Conversions:Util",
        "@heir//lib/Dialect/Preprocessing/IR:Dialect",
        "@heir//lib/Utils",
        "@heir//lib/Utils:ConversionUtils",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

gentbl_cc_library(
    name = "pass_inc_gen",
    tbl_outs = {
        "PreprocessingToOpenfhe.h.inc": [
            "-gen-pass-decls",
            "-name=PreprocessingToOpenfhe",
        ],
        "PreprocessingToOpenfhe.md": ["-gen-pass-doc"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "PreprocessingToOpenfhe.td",
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)
