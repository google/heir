# HEIR, a dialect for encrypted programs

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

exports_files(
    [
        "HEIRDialect.h",
        "HEIROps.h",
    ],
)

td_library(
    name = "td_files",
    srcs = [
        "HEIRDialect.td",
        "HEIROps.td",
    ],
    includes = ["@heir//include"],
    deps = [
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "ops_inc_gen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "HEIROps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "HEIROps.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "HEIROps.td",
    deps = [
        ":dialect_inc_gen",
        ":td_files",
    ],
)

gentbl_cc_library(
    name = "dialect_inc_gen",
    tbl_outs = [
        (
            ["-gen-dialect-decls"],
            "HEIRDialect.h.inc",
        ),
        (
            ["-gen-dialect-defs"],
            "HEIRDialect.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "HEIRDialect.td",
    deps = [
        ":td_files",
    ],
)
