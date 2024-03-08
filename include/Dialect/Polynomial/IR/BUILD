# Polynomial, a dialect defining polynomials in a polynomial ring and their operations.

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

exports_files(
    [
        "PolynomialAttributes.h",
        "PolynomialDialect.h",
        "PolynomialOps.h",
        "PolynomialTypes.h",
        "Polynomial.h",
        "PolynomialDetail.h",
    ],
)

td_library(
    name = "td_files",
    srcs = [
        "PolynomialAttributes.td",
        "PolynomialDialect.td",
        "PolynomialOps.td",
        "PolynomialPatterns.td",
        "PolynomialTypes.td",
    ],
    # include from the heir-root to enable fully-qualified include-paths
    includes = ["../../../.."],
    deps = [
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
            ["-gen-dialect-decls"],
            "PolynomialDialect.h.inc",
        ),
        (
            ["-gen-dialect-defs"],
            "PolynomialDialect.cpp.inc",
        ),
        (
            ["-gen-dialect-doc"],
            "PolynomialDialect.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "PolynomialDialect.td",
    deps = [
        ":td_files",
    ],
)

gentbl_cc_library(
    name = "attributes_inc_gen",
    tbl_outs = [
        (
            ["-gen-attrdef-decls"],
            "PolynomialAttributes.h.inc",
        ),
        (
            ["-gen-attrdef-defs"],
            "PolynomialAttributes.cpp.inc",
        ),
        (
            ["-gen-attrdef-doc"],
            "PolynomialAttributes.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "PolynomialAttributes.td",
    deps = [
        ":dialect_inc_gen",
        ":td_files",
    ],
)

gentbl_cc_library(
    name = "types_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-typedef-decls",
                "-typedefs-dialect=polynomial",
            ],
            "PolynomialTypes.h.inc",
        ),
        (
            [
                "-gen-typedef-defs",
                "-typedefs-dialect=polynomial",
            ],
            "PolynomialTypes.cpp.inc",
        ),
        (
            [
                "-gen-typedef-doc",
                "-typedefs-dialect=polynomial",
            ],
            "PolynomialTypes.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "PolynomialTypes.td",
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
            "PolynomialOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "PolynomialOps.cpp.inc",
        ),
        (
            ["-gen-op-doc"],
            "PolynomialOps.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "PolynomialOps.td",
    deps = [
        ":dialect_inc_gen",
        ":td_files",
        ":types_inc_gen",
    ],
)

gentbl_cc_library(
    name = "canonicalize_inc_gen",
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "PolynomialCanonicalize.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "PolynomialPatterns.td",
    deps = [
        ":td_files",
        ":types_inc_gen",
        "@llvm-project//mlir:ArithOpsTdFiles",
    ],
)
