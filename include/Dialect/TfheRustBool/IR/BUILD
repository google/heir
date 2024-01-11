# TfheRustBool, an exit dialect to the boolean tfhe-rs API

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

exports_files(
    [
        "TfheRustBoolDialect.h",
        "TfheRustBoolOps.h",
        "TfheRustBoolTypes.h",
    ],
)

td_library(
    name = "td_files",
    srcs = [
        "TfheRustBoolDialect.td",
        "TfheRustBoolOps.td",
        "TfheRustBoolTypes.td",
    ],
    deps = [
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "dialect_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-dialect-decls",
            ],
            "TfheRustBoolDialect.h.inc",
        ),
        (
            [
                "-gen-dialect-defs",
            ],
            "TfheRustBoolDialect.cpp.inc",
        ),
        (
            [
                "-gen-dialect-doc",
            ],
            "TfheRustBoolDialect.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "TfheRustBoolDialect.td",
    deps = [
        ":td_files",
    ],
)

gentbl_cc_library(
    name = "types_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-typedef-decls",
            ],
            "TfheRustBoolTypes.h.inc",
        ),
        (
            [
                "-gen-typedef-defs",
            ],
            "TfheRustBoolTypes.cpp.inc",
        ),
        (
            ["-gen-typedef-doc"],
            "TfheRustBoolTypes.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "TfheRustBoolTypes.td",
    deps = [
        ":dialect_inc_gen",
        ":td_files",
    ],
)

gentbl_cc_library(
    name = "ops_inc_gen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "TfheRustBoolOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "TfheRustBoolOps.cpp.inc",
        ),
        (
            ["-gen-op-doc"],
            "TfheRustBoolOps.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "TfheRustBoolOps.td",
    deps = [
        ":dialect_inc_gen",
        ":td_files",
        ":types_inc_gen",
    ],
)
