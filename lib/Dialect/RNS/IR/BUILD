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
        "RNSOps.cpp",
        "RNSTypes.cpp",
    ],
    hdrs = [
        "RNSAttributes.h",
        "RNSDialect.h",
        "RNSOps.h",
        "RNSTypeInterfaces.h",
        "RNSTypes.h",
    ],
    deps = [
        ":attributes_inc_gen",
        ":dialect_inc_gen",
        ":ops_inc_gen",
        ":type_interfaces_inc_gen",
        ":types_inc_gen",
        "@heir//lib/Dialect/ModArith/IR:Dialect",
        "@heir//lib/Dialect/ModArith/IR:TypeInterfaces",
        "@heir//lib/Utils:APIntUtils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "TypeInterfaces",
    hdrs = [
        "RNSDialect.h",
        "RNSTypeInterfaces.h",
    ],
    deps = [
        ":dialect_inc_gen",
        ":type_interfaces_inc_gen",
        "@llvm-project//mlir:IR",
    ],
)

td_library(
    name = "td_files",
    srcs = [
        "RNSAttributes.td",
        "RNSDialect.td",
        "RNSOps.td",
        "RNSTypes.td",
    ],
    # include from the heir-root to enable fully-qualified include-paths
    includes = ["../../../.."],
    deps = [
        ":type_interfaces_td",
        "@heir//lib/Dialect/ModArith/IR:type_interfaces_td",
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

td_library(
    name = "type_interfaces_td",
    srcs = [
        "RNSTypeInterfaces.td",
    ],
    includes = ["../../../.."],
    deps = [
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
