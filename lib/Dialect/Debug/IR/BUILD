# Debug dialect

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
        "DebugDialect.cpp",
        "DebugOps.cpp",
    ],
    hdrs = [
        "DebugDialect.h",
        "DebugOps.h",
    ],
    deps = [
        ":dialect_inc_gen",
        ":ops_inc_gen",
        "@llvm-project//mlir:IR",
    ],
)

td_library(
    name = "td_files",
    srcs = [
        "DebugDialect.td",
        "DebugOps.td",
    ],
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

add_heir_dialect_library(
    name = "dialect_inc_gen",
    dialect = "Debug",
    kind = "dialect",
    td_file = "DebugDialect.td",
    deps = [
        ":td_files",
    ],
)

add_heir_dialect_library(
    name = "ops_inc_gen",
    dialect = "Debug",
    kind = "op",
    td_file = "DebugOps.td",
    deps = [
        ":td_files",
    ],
)
