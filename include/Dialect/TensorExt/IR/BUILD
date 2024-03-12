# TensorExt tablegen and headers

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

exports_files(
    [
        "TensorExtDialect.h",
        "TensorExtOps.h",
        "TensorExtPatterns.h",
    ],
)

td_library(
    name = "td_files",
    srcs = [
        "TensorExtCanonicalization.td",
        "TensorExtDialect.td",
        "TensorExtOps.td",
    ],
    # include from the heir-root to enable fully-qualified include-paths
    includes = ["../../../.."],
    deps = [
        "@heir//include/DRR",
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "dialect_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-dialect-decls",
            ],
            "TensorExtDialect.h.inc",
        ),
        (
            [
                "-gen-dialect-defs",
            ],
            "TensorExtDialect.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "TensorExtDialect.td",
    deps = [
        ":td_files",
    ],
)

gentbl_cc_library(
    name = "ops_inc_gen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "TensorExtOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "TensorExtOps.cpp.inc",
        ),
        (
            ["-gen-op-doc"],
            "TensorExtOps.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "TensorExtOps.td",
    deps = [
        ":dialect_inc_gen",
        ":td_files",
        "@heir//include/Dialect/Polynomial/IR:td_files",
    ],
)

gentbl_cc_library(
    name = "canonicalize_inc_gen",
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "TensorExtCanonicalization.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "TensorExtCanonicalization.td",
    deps = [
        ":ops_inc_gen",
        ":td_files",
        "@llvm-project//mlir:ArithOpsTdFiles",
        "@llvm-project//mlir:TensorOpsTdFiles",
    ],
)
