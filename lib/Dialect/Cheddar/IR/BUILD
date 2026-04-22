# Cheddar dialect implementation

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
        "CheddarDialect.cpp",
    ],
    hdrs = [
        "CheddarAttributes.h",
        "CheddarDialect.h",
        "CheddarOps.h",
        "CheddarTypes.h",
    ],
    deps = [
        ":CheddarAttributes",
        ":CheddarOps",
        ":CheddarTypes",
        ":attributes_inc_gen",
        ":dialect_inc_gen",
        ":ops_inc_gen",
        ":types_inc_gen",
        "@heir//lib/Dialect:HEIRInterfaces",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
    ],
)

cc_library(
    name = "CheddarAttributes",
    srcs = [
        "CheddarAttributes.cpp",
    ],
    hdrs = [
        "CheddarAttributes.h",
        "CheddarDialect.h",
    ],
    deps = [
        ":attributes_inc_gen",
        ":dialect_inc_gen",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "CheddarTypes",
    srcs = [
        "CheddarTypes.cpp",
    ],
    hdrs = [
        "CheddarAttributes.h",
        "CheddarDialect.h",
        "CheddarTypes.h",
    ],
    deps = [
        ":CheddarAttributes",
        ":attributes_inc_gen",
        ":dialect_inc_gen",
        ":types_inc_gen",
        "@heir//lib/Dialect:HEIRInterfaces",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "CheddarOps",
    srcs = [
        "CheddarOps.cpp",
    ],
    hdrs = [
        "CheddarDialect.h",
        "CheddarOps.h",
        "CheddarTypes.h",
    ],
    deps = [
        ":CheddarAttributes",
        ":CheddarTypes",
        ":dialect_inc_gen",
        ":ops_inc_gen",
        ":types_inc_gen",
        "@heir//lib/Dialect:HEIRInterfaces",
        "@heir//lib/Utils",
        "@heir//lib/Utils:RotationUtils",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Support",
    ],
)

td_library(
    name = "td_files",
    srcs = [
        "CheddarAttributes.td",
        "CheddarDialect.td",
        "CheddarOps.td",
        "CheddarTypes.td",
    ],
    # include from the heir-root to enable fully-qualified include-paths
    includes = ["../../../.."],
    deps = [
        "@heir//lib/Dialect:td_files",
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

add_heir_dialect_library(
    name = "dialect_inc_gen",
    dialect = "Cheddar",
    kind = "dialect",
    td_file = "CheddarDialect.td",
    deps = [
        ":td_files",
    ],
)

add_heir_dialect_library(
    name = "attributes_inc_gen",
    dialect = "Cheddar",
    kind = "attribute",
    td_file = "CheddarAttributes.td",
    deps = [
        ":td_files",
    ],
)

add_heir_dialect_library(
    name = "types_inc_gen",
    dialect = "Cheddar",
    kind = "type",
    td_file = "CheddarTypes.td",
    deps = [
        ":td_files",
    ],
)

add_heir_dialect_library(
    name = "ops_inc_gen",
    dialect = "Cheddar",
    kind = "op",
    td_file = "CheddarOps.td",
    deps = [
        ":td_files",
        "@heir//lib/Dialect:td_files",
    ],
)
