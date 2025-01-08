# Mgmt dialect implementation

load("@heir//lib/Dialect:dialect.bzl", "add_heir_dialect_library")
load("@llvm-project//mlir:tblgen.bzl", "td_library")

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
        ":MgmtOps",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
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
        "MgmtDialect.td",
        "MgmtOps.td",
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
