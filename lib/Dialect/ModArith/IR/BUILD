# ModArith dialect

load("@heir//lib/Dialect:dialect.bzl", "add_heir_dialect_library")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Dialect",
    srcs = [
        "ModArithDialect.cpp",
        "ModArithOps.cpp",
        "ModArithTypes.cpp",
    ],
    hdrs = [
        "ModArithDialect.h",
        "ModArithOps.h",
        "ModArithTypes.h",
    ],
    deps = [
        ":canonicalization_inc_gen",
        ":dialect_inc_gen",
        ":ops_inc_gen",
        ":types_inc_gen",
        "@heir//lib/Dialect/RNS/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:CommonFolders",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Support",
    ],
)

td_library(
    name = "td_files",
    srcs = [
        "ModArithCanonicalization.td",
        "ModArithDialect.td",
        "ModArithOps.td",
        "ModArithTypes.td",
    ],
    # include from the heir-root to enable fully-qualified include-paths
    includes = ["../../../.."],
    deps = [
        "@heir//lib/Dialect/RNS/IR:td_files",
        "@heir//lib/Utils/DRR",
        "@llvm-project//mlir:ArithOpsTdFiles",
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

add_heir_dialect_library(
    name = "dialect_inc_gen",
    dialect = "ModArith",
    kind = "dialect",
    td_file = "ModArithDialect.td",
    deps = [
        ":td_files",
    ],
)

add_heir_dialect_library(
    name = "types_inc_gen",
    dialect = "ModArith",
    kind = "type",
    td_file = "ModArithTypes.td",
    deps = [
        ":td_files",
    ],
)

add_heir_dialect_library(
    name = "ops_inc_gen",
    dialect = "ModArith",
    kind = "op",
    td_file = "ModArithOps.td",
    deps = [
        ":td_files",
    ],
)

gentbl_cc_library(
    name = "canonicalization_inc_gen",
    tbl_outs = {"ModArithCanonicalization.cpp.inc": ["-gen-rewriters"]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ModArithCanonicalization.td",
    deps = [
        ":td_files",
    ],
)
