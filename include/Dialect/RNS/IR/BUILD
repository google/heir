# RNS tablegen and headers

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

exports_files(
    [
        "RNSDialect.h",
        "RNSTypes.h",
        "RNSOps.h",
    ],
)

td_library(
    name = "td_files",
    srcs = [
        "RNSDialect.td",
        "RNSOps.td",
        "RNSTypes.td",
    ],
    # include from the heir-root to enable fully-qualified include-paths
    includes = ["../../../.."],
    deps = [
        "@llvm-project//mlir:BuiltinDialectTdFiles",
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
            "RNSDialect.h.inc",
        ),
        (
            [
                "-gen-dialect-defs",
            ],
            "RNSDialect.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "RNSDialect.td",
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
            "RNSTypes.h.inc",
        ),
        (
            [
                "-gen-typedef-defs",
            ],
            "RNSTypes.cpp.inc",
        ),
        (
            ["-gen-typedef-doc"],
            "RNSTypes.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "RNSTypes.td",
    deps = [
        ":dialect_inc_gen",
        ":td_files",
        "@heir//include/Dialect/Polynomial/IR:td_files",
    ],
)

gentbl_cc_library(
    name = "ops_inc_gen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "RNSOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "RNSOps.cpp.inc",
        ),
        (
            ["-gen-op-doc"],
            "RNSOps.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "RNSOps.td",
    deps = [
        ":dialect_inc_gen",
        ":td_files",
        ":types_inc_gen",
        "@heir//include/Dialect/Polynomial/IR:td_files",
    ],
)
