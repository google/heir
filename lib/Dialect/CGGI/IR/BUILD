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
        "CGGIDialect.cpp",
        "CGGIOps.cpp",
    ],
    hdrs = [
        "CGGIAttributes.h",
        "CGGIDialect.h",
        "CGGIEnums.h",
        "CGGIOps.h",
    ],
    deps = [
        ":attributes_inc_gen",
        ":dialect_inc_gen",
        ":enums_inc_gen",
        ":ops_inc_gen",
        "@heir//lib/Dialect:HEIRInterfaces",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Support",
    ],
)

td_library(
    name = "td_files",
    srcs = [
        "BooleanGates.td",
        "CGGIAttributes.td",
        "CGGIBinOps.td",
        "CGGIDialect.td",
        "CGGIEnums.td",
        "CGGIOps.td",
        "CGGIPBSOps.td",
    ],
    # include from the heir-root to enable fully-qualified include-paths
    includes = ["../../../.."],
    deps = [
        "@llvm-project//mlir:ArithOpsTdFiles",
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

add_heir_dialect_library(
    name = "dialect_inc_gen",
    dialect = "CGGI",
    kind = "dialect",
    td_file = "CGGIDialect.td",
    deps = [
        ":td_files",
    ],
)

add_heir_dialect_library(
    name = "attributes_inc_gen",
    dialect = "CGGI",
    kind = "attribute",
    td_file = "CGGIAttributes.td",
    deps = [
        ":td_files",
    ],
)

add_heir_dialect_library(
    name = "enums_inc_gen",
    dialect = "CGGI",
    kind = "enum",
    td_file = "CGGIEnums.td",
    deps = [
        ":td_files",
    ],
)

add_heir_dialect_library(
    name = "ops_inc_gen",
    dialect = "CGGI",
    kind = "op",
    td_file = "CGGIOps.td",
    deps = [
        ":td_files",
        "@heir//lib/Dialect:td_files",
        "@heir//lib/Dialect/LWE/IR:td_files",
        "@heir//lib/Dialect/Polynomial/IR:td_files",
    ],
)
