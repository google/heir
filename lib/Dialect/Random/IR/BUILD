# Random dialect

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
        "RandomDialect.cpp",
    ],
    hdrs = [
        "RandomDialect.h",
        "RandomEnums.h",
        "RandomOps.h",
        "RandomTypes.h",
    ],
    deps = [
        ":dialect_inc_gen",
        ":enums_inc_gen",
        ":ops_inc_gen",
        ":types_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Support",
    ],
)

td_library(
    name = "td_files",
    srcs = [
        "RandomDialect.td",
        "RandomEnums.td",
        "RandomOps.td",
        "RandomTypes.td",
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
    dialect = "Random",
    kind = "dialect",
    td_file = "RandomDialect.td",
    deps = [
        ":td_files",
    ],
)

add_heir_dialect_library(
    name = "enums_inc_gen",
    dialect = "Random",
    kind = "enum",
    td_file = "RandomEnums.td",
    deps = [
        ":td_files",
    ],
)

add_heir_dialect_library(
    name = "types_inc_gen",
    dialect = "Random",
    kind = "type",
    td_file = "RandomTypes.td",
    deps = [
        ":td_files",
    ],
)

add_heir_dialect_library(
    name = "ops_inc_gen",
    dialect = "Random",
    kind = "op",
    td_file = "RandomOps.td",
    deps = [
        ":td_files",
    ],
)
