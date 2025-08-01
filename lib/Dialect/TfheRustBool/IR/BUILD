load("@heir//lib/Dialect:dialect.bzl", "add_heir_dialect_library")
load("@llvm-project//mlir:tblgen.bzl", "td_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Dialect",
    srcs = [
        "TfheRustBoolDialect.cpp",
    ],
    hdrs = [
        "TfheRustBoolAttributes.h",
        "TfheRustBoolDialect.h",
        "TfheRustBoolEnums.h",
        "TfheRustBoolOps.h",
        "TfheRustBoolTypes.h",
    ],
    deps = [
        ":attributes_inc_gen",
        ":dialect_inc_gen",
        ":enums_inc_gen",
        ":ops_inc_gen",
        ":types_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
    ],
)
# TfheRustBool, an exit dialect to the boolean tfhe-rs API

td_library(
    name = "td_files",
    srcs = [
        "TfheRustBoolAttributes.td",
        "TfheRustBoolDialect.td",
        "TfheRustBoolEnums.td",
        "TfheRustBoolOps.td",
        "TfheRustBoolTypes.td",
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
    dialect = "TfheRustBool",
    kind = "dialect",
    td_file = "TfheRustBoolDialect.td",
    deps = [
        ":td_files",
    ],
)

add_heir_dialect_library(
    name = "attributes_inc_gen",
    dialect = "TfheRustBool",
    kind = "attribute",
    td_file = "TfheRustBoolAttributes.td",
    deps = [
        ":td_files",
        "@heir//lib/Dialect/CGGI/IR:td_files",
    ],
)

add_heir_dialect_library(
    name = "enums_inc_gen",
    dialect = "TfheRustBool",
    kind = "enum",
    td_file = "TfheRustBoolEnums.td",
    deps = [
        ":td_files",
        "@heir//lib/Dialect/CGGI/IR:td_files",
    ],
)

add_heir_dialect_library(
    name = "types_inc_gen",
    dialect = "TfheRustBool",
    kind = "type",
    td_file = "TfheRustBoolTypes.td",
    deps = [
        ":td_files",
    ],
)

add_heir_dialect_library(
    name = "ops_inc_gen",
    dialect = "TfheRustBool",
    kind = "op",
    td_file = "TfheRustBoolOps.td",
    deps = [
        ":td_files",
    ],
)
