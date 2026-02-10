# RNS dialect implementation

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
        "RNSAttributes.cpp",
        "RNSDialect.cpp",
    ],
    hdrs = [
        "RNSAttributes.h",
        "RNSDialect.h",
        "RNSOps.h",
        "RNSTypeInterfaces.h",
        "RNSTypes.h",
    ],
    deps = [
        ":RNSOps",
        ":RNSTypes",
        ":attributes_inc_gen",
        ":dialect_inc_gen",
        ":ops_inc_gen",
        ":type_interfaces_inc_gen",
        ":types_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "RNSTypes",
    srcs = [
        "RNSTypes.cpp",
    ],
    hdrs = [
        "RNSDialect.h",
        "RNSTypeInterfaces.h",
        "RNSTypes.h",
    ],
    deps = [
        ":dialect_inc_gen",
        ":type_interfaces_inc_gen",
        ":types_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

# to avoid cyclic dependency among mod_arith and rns
cc_library(
    name = "RNSTypeInterfaces",
    srcs = [
        "RNSTypeInterfaces.cpp",
    ],
    hdrs = [
        "RNSDialect.h",
        "RNSTypeInterfaces.h",
    ],
    deps = [
        ":dialect_inc_gen",
        ":type_interfaces_inc_gen",
        ":types_inc_gen",
        "@heir//lib/Dialect/ModArith/IR:Dialect",
        "@heir//lib/Dialect/Polynomial/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "RNSOps",
    srcs = [
        "RNSOps.cpp",
    ],
    hdrs = [
        "RNSDialect.h",
        "RNSOps.h",
        "RNSTypes.h",
    ],
    deps = [
        ":RNSTypes",
        ":dialect_inc_gen",
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
        "RNSAttributes.td",
        "RNSDialect.td",
        "RNSOps.td",
        "RNSTypeInterfaces.td",
        "RNSTypes.td",
    ],
    # include from the heir-root to enable fully-qualified include-paths
    includes = ["../../../.."],
    deps = [
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

add_heir_dialect_library(
    name = "dialect_inc_gen",
    dialect = "RNS",
    kind = "dialect",
    td_file = "RNSDialect.td",
    deps = [
        ":td_files",
    ],
)

add_heir_dialect_library(
    name = "attributes_inc_gen",
    dialect = "RNS",
    kind = "attribute",
    td_file = "RNSAttributes.td",
    deps = [
        ":td_files",
    ],
)

add_heir_dialect_library(
    name = "types_inc_gen",
    dialect = "RNS",
    kind = "type",
    td_file = "RNSTypes.td",
    deps = [
        ":td_files",
    ],
)

add_heir_dialect_library(
    name = "ops_inc_gen",
    dialect = "RNS",
    kind = "op",
    td_file = "RNSOps.td",
    deps = [
        ":td_files",
    ],
)

add_heir_dialect_library(
    name = "type_interfaces_inc_gen",
    dialect = "RNS",
    kind = "type_interface",
    td_file = "RNSTypeInterfaces.td",
    deps = [
        "@llvm-project//mlir:BuiltinDialectTdFiles",
    ],
)
