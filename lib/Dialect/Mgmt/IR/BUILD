# Mgmt dialect implementation

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
        "MgmtDialect.cpp",
    ],
    hdrs = [
        "MgmtAttributes.h",
        "MgmtDialect.h",
        "MgmtOps.h",
    ],
    deps = [
        "attributes_inc_gen",
        "dialect_inc_gen",
        "ops_inc_gen",
        ":MgmtAttributes",
        ":MgmtOps",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
    ],
)

cc_library(
    name = "MgmtAttributes",
    srcs = [
        "MgmtAttributes.cpp",
    ],
    hdrs = [
        "MgmtAttributes.h",
        "MgmtDialect.h",
    ],
    deps = [
        ":MgmtOps",
        ":attributes_inc_gen",
        ":dialect_inc_gen",
        "@heir//lib/Analysis/SecretnessAnalysis",
        "@heir//lib/Utils:AttributeUtils",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "MgmtOps",
    srcs = [
        "MgmtOps.cpp",
    ],
    hdrs = [
        "MgmtDialect.h",
        "MgmtOps.h",
    ],
    deps = [
        ":canonicalize_inc_gen",
        ":dialect_inc_gen",
        ":ops_inc_gen",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
    ],
)

td_library(
    name = "td_files",
    srcs = [
        "MgmtAttributes.td",
        "MgmtCanonicalization.td",
        "MgmtDialect.td",
        "MgmtOps.td",
    ],
    # include from the heir-root to enable fully-qualified include-paths
    includes = ["../../../.."],
    deps = [
        "@heir//lib/Utils/DRR",
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

add_heir_dialect_library(
    name = "dialect_inc_gen",
    dialect = "Mgmt",
    kind = "dialect",
    td_file = "MgmtDialect.td",
    deps = [
        ":td_files",
    ],
)

add_heir_dialect_library(
    name = "attributes_inc_gen",
    dialect = "Mgmt",
    kind = "attribute",
    td_file = "MgmtAttributes.td",
    deps = [
        ":td_files",
    ],
)

add_heir_dialect_library(
    name = "ops_inc_gen",
    dialect = "Mgmt",
    kind = "op",
    td_file = "MgmtOps.td",
    deps = [
        ":td_files",
    ],
)

gentbl_cc_library(
    name = "canonicalize_inc_gen",
    tbl_outs = {"MgmtCanonicalization.cpp.inc": ["-gen-rewriters"]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "MgmtCanonicalization.td",
    deps = [
        ":ops_inc_gen",
        ":td_files",
        "@llvm-project//mlir:ArithOpsTdFiles",
    ],
)
