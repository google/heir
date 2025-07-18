# MathExt dialect implementation

load("@heir//lib/Dialect:dialect.bzl", "add_heir_dialect_library")
load("@llvm-project//mlir:tblgen.bzl", "td_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Dialect",
    srcs = [
        "MathExtDialect.cpp",
    ],
    hdrs = [
        "MathExtDialect.h",
        "MathExtOps.h",
    ],
    deps = [
        "dialect_inc_gen",
        "ops_inc_gen",
        ":MathExtOps",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "MathExtOps",
    srcs = [
        "MathExtOps.cpp",
    ],
    hdrs = [
        "MathExtDialect.h",
        "MathExtOps.h",
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
        "MathExtDialect.td",
        "MathExtOps.td",
    ],
    # include from the heir-root to enable fully-qualified include-paths
    includes = ["../../../.."],
    deps = [
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

add_heir_dialect_library(
    name = "dialect_inc_gen",
    dialect = "MathExt",
    kind = "dialect",
    td_file = "MathExtDialect.td",
    deps = [
        ":td_files",
    ],
)

add_heir_dialect_library(
    name = "ops_inc_gen",
    dialect = "MathExt",
    kind = "op",
    td_file = "MathExtOps.td",
    deps = [
        ":td_files",
    ],
)
