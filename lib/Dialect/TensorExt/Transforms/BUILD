load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

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
        ":CollapseInsertionChains",
        ":InsertRotate",
        ":RotateAndReduce",
        ":pass_inc_gen",
        "@heir//lib/Dialect/TensorExt/IR:Dialect",
    ],
)

cc_library(
    name = "InsertRotate",
    srcs = ["InsertRotate.cpp"],
    hdrs = [
        "InsertRotate.h",
    ],
    deps = [
        ":insert_rotate_inc_gen",
        ":pass_inc_gen",
        "@heir//lib/Analysis/TargetSlotAnalysis",
        "@heir//lib/Dialect/TensorExt/IR:Dialect",
        "@heir//lib/Dialect/TensorExt/IR:canonicalize_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "CollapseInsertionChains",
    srcs = ["CollapseInsertionChains.cpp"],
    hdrs = [
        "CollapseInsertionChains.h",
    ],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect:Utils",
        "@heir//lib/Dialect/TensorExt/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "RotateAndReduce",
    srcs = ["RotateAndReduce.cpp"],
    hdrs = [
        "RotateAndReduce.h",
    ],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Analysis/RotationAnalysis",
        "@heir//lib/Dialect/Secret/IR:Dialect",
        "@heir//lib/Dialect/TensorExt/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
    ],
)
# TensorExt pass tablegen and headers.

gentbl_cc_library(
    name = "pass_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=TensorExt",
            ],
            "Passes.h.inc",
        ),
        (
            ["-gen-pass-doc"],
            "TensorExtPasses.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Passes.td",
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "insert_rotate_inc_gen",
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "InsertRotate.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "InsertRotate.td",
    deps = [
        "@heir//lib/DRR",
        "@heir//lib/Dialect/TensorExt/IR:ops_inc_gen",
        "@heir//lib/Dialect/TensorExt/IR:td_files",
        "@llvm-project//mlir:ArithOpsTdFiles",
        "@llvm-project//mlir:TensorOpsTdFiles",
    ],
)
