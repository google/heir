load("@heir//lib/Dialect:dialect.bzl", "add_heir_dialect_library")
load("@llvm-project//mlir:tblgen.bzl", "td_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

td_library(
    name = "td_files",
    srcs = [
        "KeyMemRuntimeDialect.td",
        "KeyMemRuntimeOps.td",
        "KeyMemRuntimeTypes.td",
    ],
    includes = ["../../../.."],
    deps = [
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

add_heir_dialect_library(
    name = "dialect_inc_gen",
    dialect = "KeyMemRuntime",
    kind = "dialect",
    td_file = "KeyMemRuntimeDialect.td",
    deps = [
        ":td_files",
    ],
)

add_heir_dialect_library(
    name = "types_inc_gen",
    dialect = "KeyMemRuntime",
    kind = "type",
    td_file = "KeyMemRuntimeTypes.td",
    deps = [
        ":dialect_inc_gen",
        ":td_files",
    ],
)

add_heir_dialect_library(
    name = "ops_inc_gen",
    dialect = "KeyMemRuntime",
    kind = "op",
    td_file = "KeyMemRuntimeOps.td",
    deps = [
        ":dialect_inc_gen",
        ":td_files",
        ":types_inc_gen",
    ],
)

cc_library(
    name = "Dialect",
    srcs = [
        "KeyMemRuntimeDialect.cpp",
        "KeyMemRuntimeOps.cpp",
    ],
    hdrs = [
        "KeyMemRuntimeDialect.h",
        "KeyMemRuntimeOps.h",
        "KeyMemRuntimeTypes.h",
    ],
    deps = [
        ":dialect_inc_gen",
        ":ops_inc_gen",
        ":types_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)
