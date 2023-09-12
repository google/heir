# Poly, a dialect defining polynomials in a polynomial ring and their operations.

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

exports_files(
    [
        "PolyAttributes.h",
        "PolyDialect.h",
        "PolyOps.h",
        "PolyTypes.h",
        "Polynomial.h",
        "PolynomialDetail.h",
    ],
)

td_library(
    name = "td_files",
    srcs = [
        "PolyAttributes.td",
        "PolyDialect.td",
        "PolyOps.td",
        "PolyTypes.td",
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
            "PolyDialect.h.inc",
        ),
        (
            ["-gen-dialect-defs"],
            "PolyDialect.cpp.inc",
        ),
        (
            ["-gen-dialect-doc"],
            "PolyDialect.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "PolyDialect.td",
    deps = [
        ":td_files",
    ],
)

gentbl_cc_library(
    name = "attributes_inc_gen",
    tbl_outs = [
        (
            ["-gen-attrdef-decls"],
            "PolyAttributes.h.inc",
        ),
        (
            ["-gen-attrdef-defs"],
            "PolyAttributes.cpp.inc",
        ),
        (
            ["-gen-attrdef-doc"],
            "PolyAttributes.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "PolyAttributes.td",
    deps = [
        ":dialect_inc_gen",
        ":td_files",
    ],
)

gentbl_cc_library(
    name = "types_inc_gen",
    tbl_outs = [
        (
            ["-gen-typedef-decls"],
            "PolyTypes.h.inc",
        ),
        (
            ["-gen-typedef-defs"],
            "PolyTypes.cpp.inc",
        ),
        (
            ["-gen-typedef-doc"],
            "PolyTypes.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "PolyTypes.td",
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
            "PolyOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "PolyOps.cpp.inc",
        ),
        (
            ["-gen-op-doc"],
            "PolyOps.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "PolyOps.td",
    deps = [
        ":dialect_inc_gen",
        ":td_files",
        ":types_inc_gen",
    ],
)
