# CKKS dialect

load("@heir//lib/Dialect:dialect.bzl", "add_heir_dialect_library")
load("@llvm-project//mlir:tblgen.bzl", "td_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Dialect",
    srcs = [
        "CKKSDialect.cpp",
        "CKKSOps.cpp",
    ],
    hdrs = [
        "CKKSAttributes.h",
        "CKKSDialect.h",
        "CKKSEnums.h",
        "CKKSOps.h",
    ],
    deps = [
        ":attributes_inc_gen",
        ":dialect_inc_gen",
        ":enums_inc_gen",
        ":ops_inc_gen",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@heir//lib/Dialect/LWE/IR:Patterns",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Support",
    ],
)

td_library(
    name = "td_files",
    srcs = [
        "CKKSAttributes.td",
        "CKKSDialect.td",
        "CKKSEnums.td",
        "CKKSOps.td",
    ],
    deps = [
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

add_heir_dialect_library(
    name = "dialect_inc_gen",
    dialect = "CKKS",
    kind = "dialect",
    td_file = "CKKSDialect.td",
    deps = [
        ":td_files",
    ],
)

add_heir_dialect_library(
    name = "ops_inc_gen",
    dialect = "CKKS",
    kind = "op",
    td_file = "CKKSOps.td",
    deps = [
        ":td_files",
        "@heir//lib/Dialect/LWE/IR:td_files",
        "@heir//lib/Dialect/Polynomial/IR:td_files",
    ],
)

add_heir_dialect_library(
    name = "attributes_inc_gen",
    dialect = "CKKS",
    kind = "attribute",
    td_file = "CKKSAttributes.td",
    deps = [
        ":td_files",
    ],
)

add_heir_dialect_library(
    name = "enums_inc_gen",
    dialect = "CKKS",
    kind = "enum",
    td_file = "CKKSEnums.td",
    deps = [
        ":td_files",
    ],
)
