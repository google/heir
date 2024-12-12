load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Dialect",
    srcs = glob(["*.cpp"]),
    hdrs = glob(["*.h"]),
    deps = [
        ":attributes_inc_gen",
        ":canonicalization_inc_gen",
        ":dialect_inc_gen",
        ":ops_inc_gen",
        ":types_inc_gen",
        "@heir//lib/Dialect/ModArith/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Support",
    ],
)

td_library(
    name = "td_files",
    srcs = [
        "PolynomialAttributes.td",
        "PolynomialCanonicalization.td",
        "PolynomialDialect.td",
        "PolynomialOps.td",
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
            [
                "-gen-dialect-decls",
            ],
            "PolynomialDialect.h.inc",
        ),
        (
            [
                "-gen-dialect-defs",
            ],
            "PolynomialDialect.cpp.inc",
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
            [
                "-gen-attrdef-decls",
                "-attrdefs-dialect=polynomial",
            ],
            "PolynomialAttributes.h.inc",
        ),
        (
            [
                "-gen-attrdef-defs",
                "-attrdefs-dialect=polynomial",
            ],
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
        ":attributes_inc_gen",
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
        "@heir//lib/Dialect/ModArith/IR:td_files",
    ],
)

gentbl_cc_library(
    name = "canonicalization_inc_gen",
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "PolynomialCanonicalization.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "PolynomialCanonicalization.td",
    deps = [
        ":td_files",
        "@heir//lib/Dialect/ModArith/IR:td_files",
        "@llvm-project//mlir:ArithOpsTdFiles",
    ],
)
