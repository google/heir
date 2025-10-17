# Orion dialect implementation

load("@heir//lib/Dialect:dialect.bzl", "add_heir_dialect_library")
load("@llvm-project//mlir:tblgen.bzl", "td_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Dialect",
    srcs = [
        "OrionDialect.cpp",
    ],
    hdrs = [
        "OrionDialect.h",
        "OrionOps.h",
    ],
    deps = [
        "dialect_inc_gen",
        "ops_inc_gen",
        ":OrionOps",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "OrionOps",
    hdrs = [
        "OrionDialect.h",
        "OrionOps.h",
    ],
    deps = [
        ":dialect_inc_gen",
        ":ops_inc_gen",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@llvm-project//mlir:IR",
    ],
)

td_library(
    name = "td_files",
    srcs = [
        "OrionDialect.td",
        "OrionOps.td",
    ],
    # include from the heir-root to enable fully-qualified include-paths
    includes = ["../../../.."],
    deps = [
        "@heir//lib/Dialect/LWE/IR:td_files",
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

add_heir_dialect_library(
    name = "dialect_inc_gen",
    dialect = "Orion",
    kind = "dialect",
    td_file = "OrionDialect.td",
    deps = [
        ":td_files",
    ],
)

add_heir_dialect_library(
    name = "ops_inc_gen",
    dialect = "Orion",
    kind = "op",
    td_file = "OrionOps.td",
    deps = [
        ":td_files",
    ],
)
