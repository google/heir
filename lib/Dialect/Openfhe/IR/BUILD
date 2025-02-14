load("@heir//lib/Dialect:dialect.bzl", "add_heir_dialect_library")
load("@llvm-project//mlir:tblgen.bzl", "td_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Dialect",
    srcs = [
        "OpenfheDialect.cpp",
    ],
    hdrs = [
        "OpenfheDialect.h",
        "OpenfheOps.h",
        "OpenfheTypes.h",
    ],
    deps = [
        ":dialect_inc_gen",
        ":ops_inc_gen",
        ":types_inc_gen",
        "@heir//lib/Dialect:FHEHelpers",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@heir//lib/Utils/Tablegen:AsmInterfaces",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Support",
    ],
)

td_library(
    name = "td_files",
    srcs = [
        "OpenfheDialect.td",
        "OpenfheOps.td",
        "OpenfheTypes.td",
    ],
    includes = ["../../../.."],
    deps = [
        "@heir//lib/Utils/Tablegen:td_files",
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

add_heir_dialect_library(
    name = "dialect_inc_gen",
    dialect = "Openfhe",
    kind = "dialect",
    td_file = "OpenfheDialect.td",
    deps = [
        ":td_files",
    ],
)

add_heir_dialect_library(
    name = "types_inc_gen",
    dialect = "Openfhe",
    kind = "type",
    td_file = "OpenfheTypes.td",
    deps = [
        ":td_files",
    ],
)

add_heir_dialect_library(
    name = "ops_inc_gen",
    dialect = "Openfhe",
    kind = "op",
    td_file = "OpenfheOps.td",
    deps = [
        ":td_files",
        "@heir//lib/Dialect/LWE/IR:td_files",
    ],
)
