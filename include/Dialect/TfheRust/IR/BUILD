# TfheRust, an exit dialect to the tfhe-rs API

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

exports_files(
    [
        "TfheRustDialect.h",
        "TfheRustOps.h",
        "TfheRustTypes.h",
    ],
)

td_library(
    name = "td_files",
    srcs = [
        "TfheRustDialect.td",
        "TfheRustOps.td",
        "TfheRustTypes.td",
    ],
    deps = [
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
            "TfheRustDialect.h.inc",
        ),
        (
            [
                "-gen-dialect-defs",
            ],
            "TfheRustDialect.cpp.inc",
        ),
        (
            [
                "-gen-dialect-doc",
            ],
            "TfheRustDialect.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "TfheRustDialect.td",
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
            "TfheRustTypes.h.inc",
        ),
        (
            [
                "-gen-typedef-defs",
            ],
            "TfheRustTypes.cpp.inc",
        ),
        (
            ["-gen-typedef-doc"],
            "TfheRustTypes.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "TfheRustTypes.td",
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
            "TfheRustOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "TfheRustOps.cpp.inc",
        ),
        (
            ["-gen-op-doc"],
            "TfheRustOps.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "TfheRustOps.td",
    deps = [
        ":dialect_inc_gen",
        ":td_files",
        ":types_inc_gen",
    ],
)
