load("@heir//lib/Dialect:dialect.bzl", "add_heir_dialect_library")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

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
        "@heir//lib/Dialect/RNS/IR:Dialect",
        "@heir//lib/Dialect/RNS/IR:RNSOps",
        "@heir//lib/Dialect/RNS/IR:RNSTypes",
        "@heir//lib/Utils/Polynomial",
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
        "@heir//lib/Dialect/ModArith/IR:td_files",
        "@heir//lib/Dialect/RNS/IR:td_files",
        "@llvm-project//mlir:ArithOpsTdFiles",
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

add_heir_dialect_library(
    name = "dialect_inc_gen",
    dialect = "Polynomial",
    kind = "dialect",
    td_file = "PolynomialDialect.td",
    deps = [
        ":td_files",
    ],
)

add_heir_dialect_library(
    name = "attributes_inc_gen",
    dialect = "Polynomial",
    kind = "attribute",
    td_file = "PolynomialAttributes.td",
    deps = [
        ":td_files",
    ],
)

add_heir_dialect_library(
    name = "types_inc_gen",
    dialect = "Polynomial",
    kind = "type",
    td_file = "PolynomialTypes.td",
    deps = [
        ":td_files",
    ],
)

add_heir_dialect_library(
    name = "ops_inc_gen",
    dialect = "Polynomial",
    kind = "op",
    td_file = "PolynomialOps.td",
    deps = [
        ":td_files",
    ],
)

gentbl_cc_library(
    name = "canonicalization_inc_gen",
    tbl_outs = {"PolynomialCanonicalization.cpp.inc": ["-gen-rewriters"]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "PolynomialCanonicalization.td",
    deps = [
        ":td_files",
    ],
)
