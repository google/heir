# Poulpy dialect implementation.

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
        "PoulpyDialect.cpp",
    ],
    hdrs = [
        "PoulpyDialect.h",
        "PoulpyOps.h",
        "PoulpyTypes.h",
    ],
    deps = [
        ":Enums",
        ":PoulpyOps",
        ":PoulpyTypes",
        ":dialect_inc_gen",
        ":ops_inc_gen",
        ":types_inc_gen",
        "@heir//lib/Dialect:HEIRInterfaces",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "Enums",
    srcs = ["PoulpyEnums.cpp"],
    hdrs = ["PoulpyEnums.h"],
    deps = [
        ":enums_inc_gen",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "PoulpyTypes",
    srcs = [
        "PoulpyTypes.cpp",
    ],
    hdrs = [
        "PoulpyDialect.h",
        "PoulpyTypes.h",
    ],
    deps = [
        ":Enums",
        ":dialect_inc_gen",
        ":types_inc_gen",
        "@heir//lib/Dialect:HEIRInterfaces",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "PoulpyOps",
    srcs = [
        "PoulpyOps.cpp",
    ],
    hdrs = [
        "PoulpyDialect.h",
        "PoulpyOps.h",
        "PoulpyTypes.h",
    ],
    deps = [
        ":Enums",
        ":PoulpyTypes",
        ":dialect_inc_gen",
        ":ops_inc_gen",
        ":types_inc_gen",
        "@heir//lib/Dialect:HEIRInterfaces",
        "@llvm-project//mlir:IR",
    ],
)

td_library(
    name = "td_files",
    srcs = [
        "PoulpyDialect.td",
        "PoulpyEnums.td",
        "PoulpyOps.td",
        "PoulpyTypes.td",
    ],
    # include from the heir-root to enable fully-qualified include-paths
    includes = ["../../../.."],
    deps = [
        "@heir//lib/Dialect:td_files",
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

add_heir_dialect_library(
    name = "dialect_inc_gen",
    dialect = "Poulpy",
    kind = "dialect",
    td_file = "PoulpyDialect.td",
    deps = [
        ":td_files",
    ],
)

add_heir_dialect_library(
    name = "enums_inc_gen",
    dialect = "Poulpy",
    kind = "enum",
    td_file = "PoulpyEnums.td",
    deps = [
        ":td_files",
    ],
)

add_heir_dialect_library(
    name = "types_inc_gen",
    dialect = "Poulpy",
    kind = "type",
    td_file = "PoulpyTypes.td",
    deps = [
        ":td_files",
    ],
)

add_heir_dialect_library(
    name = "ops_inc_gen",
    dialect = "Poulpy",
    kind = "op",
    td_file = "PoulpyOps.td",
    deps = [
        ":td_files",
    ],
)
