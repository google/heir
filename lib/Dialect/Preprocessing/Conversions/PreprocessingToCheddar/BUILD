load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "PreprocessingToCheddar",
    srcs = ["PreprocessingToCheddar.cpp"],
    hdrs = ["PreprocessingToCheddar.h"],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Analysis/PreprocessingStorageLayoutAnalysis",
        "@heir//lib/Dialect/Cheddar/IR:Dialect",
        "@heir//lib/Dialect/Preprocessing/Conversions:Util",
        "@heir//lib/Dialect/Preprocessing/IR:Dialect",
        "@heir//lib/Utils",
        "@heir//lib/Utils:ConversionUtils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:DestinationStyleOpInterface",
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
    tbl_outs = {
        "PreprocessingToCheddar.h.inc": [
            "-gen-pass-decls",
            "-name=PreprocessingToCheddar",
        ],
        "PreprocessingToCheddar.md": ["-gen-pass-doc"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "PreprocessingToCheddar.td",
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)
