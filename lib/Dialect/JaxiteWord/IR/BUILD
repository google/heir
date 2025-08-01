# JaxiteWord, an exit dialect to JaxiteWord API

load("@heir//lib/Dialect:dialect.bzl", "add_heir_dialect_library")
load("@llvm-project//mlir:tblgen.bzl", "td_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Dialect",
    srcs = ["JaxiteWordDialect.cpp"],
    hdrs = [
        "JaxiteWordDialect.h",
        "JaxiteWordOps.h",
        "JaxiteWordTypes.h",
    ],
    deps = [
        ":dialect_inc_gen",
        ":ops_inc_gen",
        ":types_inc_gen",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Support",
    ],
)

td_library(
    name = "td_files",
    srcs = [
        "JaxiteWordDialect.td",
        "JaxiteWordOps.td",
        "JaxiteWordTypes.td",
    ],
    deps = [
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

add_heir_dialect_library(
    name = "dialect_inc_gen",
    dialect = "JaxiteWord",
    kind = "dialect",
    td_file = "JaxiteWordDialect.td",
    deps = [
        ":td_files",
    ],
)

add_heir_dialect_library(
    name = "types_inc_gen",
    dialect = "JaxiteWord",
    kind = "type",
    td_file = "JaxiteWordTypes.td",
    deps = [
        ":td_files",
    ],
)

add_heir_dialect_library(
    name = "ops_inc_gen",
    dialect = "JaxiteWord",
    kind = "op",
    td_file = "JaxiteWordOps.td",
    deps = [
        ":td_files",
        "@heir//lib/Dialect/LWE/IR:td_files",
    ],
)
