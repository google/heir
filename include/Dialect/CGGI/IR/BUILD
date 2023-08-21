load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

exports_files(
    [
        "CGGIDialect.h",
        "CGGIOps.h",
    ],
)

td_library(
    name = "td_files",
    srcs = [
        "CGGIDialect.td",
        "CGGIOps.td",
    ],
    includes = ["@heir//include"],
    deps = [
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "dialect_inc_gen",
    tbl_outs = [
        (
            ["-gen-dialect-decls"],
            "CGGIDialect.h.inc",
        ),
        (
            ["-gen-dialect-defs"],
            "CGGIDialect.cpp.inc",
        ),
        (
            ["-gen-dialect-doc"],
            "CGGIDialect.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "CGGIDialect.td",
    deps = [
        ":td_files",
    ],
)

gentbl_cc_library(
    name = "ops_inc_gen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "CGGIOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "CGGIOps.cpp.inc",
        ),
        (
            ["-gen-op-doc"],
            "CGGIOps.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "CGGIOps.td",
    deps = [
        ":dialect_inc_gen",
        ":td_files",
    ],
)
