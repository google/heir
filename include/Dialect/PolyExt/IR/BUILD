# PolyExt, a dialect defining "external" operations on polynomials in a polynomial ring.

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

exports_files(
    [
        "PolyExtDialect.h",
        "PolyExtOps.h",
    ],
)

td_library(
    name = "td_files",
    srcs = [
        "PolyExtDialect.td",
        "PolyExtOps.td",
    ],
    deps = [
        "@heir//include/Dialect/Poly/IR:td_files",
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
            "PolyExtDialect.h.inc",
        ),
        (
            ["-gen-dialect-defs"],
            "PolyExtDialect.cpp.inc",
        ),
        (
            ["-gen-dialect-doc"],
            "PolyExtDialect.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "PolyExtDialect.td",
    deps = [
        ":td_files",
    ],
)

gentbl_cc_library(
    name = "ops_inc_gen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "PolyExtOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "PolyExtOps.cpp.inc",
        ),
        (
            ["-gen-op-doc"],
            "PolyExtOps.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "PolyExtOps.td",
    deps = [
        ":dialect_inc_gen",
        ":td_files",
    ],
)
