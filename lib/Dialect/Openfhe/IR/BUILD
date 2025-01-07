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
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@heir//lib/Utils/Tablegen:AsmInterfaces",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
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
    dialect = "Openfhe",
    kind = "dialect",
)

add_heir_dialect_library(
    dialect = "Openfhe",
    kind = "type",
)

add_heir_dialect_library(
    dialect = "Openfhe",
    kind = "op",
    deps = [
        ":types_inc_gen",
        "@heir//lib/Dialect/LWE/IR:td_files",
    ],
)
