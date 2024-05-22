# Polynomial dialect

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Dialect",
    srcs = [
        "PolynomialDialect.cpp",
    ],
    hdrs = [
        "PolynomialAttributes.h",
        "PolynomialDialect.h",
        "PolynomialOps.h",
        "PolynomialTypes.h",
    ],
    deps = [
        ":Polynomial",
        ":PolynomialAttributes",
        ":PolynomialOps",
        ":PolynomialTypes",
        ":attributes_inc_gen",
        ":dialect_inc_gen",
        ":ops_inc_gen",
        ":types_inc_gen",
        "@heir//lib/Dialect/RNS/IR:RNSTypes",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
    ],
)

cc_library(
    name = "PolynomialAttributes",
    srcs = [
        "PolynomialAttributes.cpp",
    ],
    hdrs = [
        "PolynomialAttributes.h",
        "PolynomialDialect.h",
    ],
    deps = [
        ":NumberTheory",
        ":Polynomial",
        ":attributes_inc_gen",
        ":dialect_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "PolynomialOps",
    srcs = [
        "PolynomialOps.cpp",
    ],
    hdrs = [
        "PolynomialDialect.h",
        "PolynomialOps.h",
        "PolynomialTypes.h",
    ],
    deps = [
        ":PolynomialAttributes",
        ":canonicalize_inc_gen",
        ":dialect_inc_gen",
        ":ops_inc_gen",
        ":types_inc_gen",
        "@heir//lib/Dialect/RNS/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "PolynomialTypes",
    srcs = ["PolynomialTypes.cpp"],
    hdrs = ["PolynomialTypes.h"],
    deps = [
        ":PolynomialAttributes",
        ":types_inc_gen",
        "@heir//lib/Dialect/RNS/IR:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "Polynomial",
    srcs = ["Polynomial.cpp"],
    hdrs = [
        "Polynomial.h",
        "PolynomialDetail.h",
    ],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_test(
    name = "PolynomialTest",
    size = "small",
    srcs = ["PolynomialTest.cpp"],
    deps = [
        ":Polynomial",
        "@googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "NumberTheory",
    srcs = ["NumberTheory.cpp"],
    hdrs = ["NumberTheory.h"],
    deps = [
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "NumberTheoryTest",
    size = "small",
    srcs = ["NumberTheoryTest.cpp"],
    deps = [
        ":NumberTheory",
        "@googletest//:gtest_main",
        "@llvm-project//llvm:Support",
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
        "@heir//lib/Dialect/RNS/IR:td_files",
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
                "-typedefs-dialect=_polynomial",
            ],
            "PolynomialTypes.h.inc",
        ),
        (
            [
                "-gen-typedef-defs",
                "-typedefs-dialect=_polynomial",
            ],
            "PolynomialTypes.cpp.inc",
        ),
        (
            [
                "-gen-typedef-doc",
                "-typedefs-dialect=_polynomial",
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
