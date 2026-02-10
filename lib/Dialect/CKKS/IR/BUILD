# CKKS dialect

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
        "CKKSAttributes.cpp",
        "CKKSDialect.cpp",
        "CKKSOps.cpp",
    ],
    hdrs = [
        "CKKSAttributes.h",
        "CKKSDialect.h",
        "CKKSOps.h",
    ],
    deps = [
        ":Enums",
        ":attributes_inc_gen",
        ":dialect_inc_gen",
        ":ops_inc_gen",
        "@heir//lib/Dialect:HEIRInterfaces",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@heir//lib/Dialect/LWE/IR:Patterns",
        "@heir//lib/Dialect/ModArith/IR:Dialect",
        "@heir//lib/Dialect/Polynomial/IR:Dialect",
        "@heir//lib/Dialect/RNS/IR:RNSTypes",
        "@heir//lib/Parameters:RLWEParams",
        "@heir//lib/Parameters/CKKS:Params",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "Enums",
    srcs = ["CKKSEnums.cpp"],
    hdrs = ["CKKSEnums.h"],
    deps = [
        ":enums_inc_gen",
        "@llvm-project//mlir:IR",
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
        "@heir//lib/Dialect:td_files",
        "@heir//lib/Dialect/LWE/IR:td_files",
        "@heir//lib/Dialect/Polynomial/IR:td_files",
        "@heir//lib/Parameters/CKKS:Params",
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
