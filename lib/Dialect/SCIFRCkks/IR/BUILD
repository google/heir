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
        "SCIFRCkksDialect.cpp",
    ],
    hdrs = [
        "SCIFRCkksDialect.h",
        "SCIFRCkksOps.h",
        "SCIFRCkksTypes.h",
    ],
    deps = [
        ":dialect_inc_gen",
        ":ops_inc_gen",
        ":types_inc_gen",
        "@heir//lib/Dialect:HEIRInterfaces",
        "@heir//lib/Dialect/CKKS/IR:Dialect",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:BytecodeOpInterface",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "scifr_ckks_ops",
    srcs = ["SCIFRCkksOps.cpp"],
    hdrs = ["SCIFRCkksOps.h"],
    deps = [
        ":Dialect",
        ":ops_inc_gen",
        "@heir//lib/Dialect:HEIRInterfaces",
        "@heir//lib/Dialect/CKKS/IR:Dialect",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@llvm-project//mlir:BytecodeOpInterface",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:SideEffectInterfaces",
    ],
)

td_library(
    name = "td_files",
    srcs = [
        "SCIFRCkksDialect.td",
        "SCIFRCkksOps.td",
        "SCIFRCkksTypes.td",
    ],
    # include from the heir-root to enable fully-qualified include-paths
    includes = ["../../../.."],
    deps = [
        "@llvm-project//mlir:ArithOpsTdFiles",
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:BytecodeOpInterface",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

add_heir_dialect_library(
    name = "dialect_inc_gen",
    dialect = "SCIFRCkks",
    kind = "dialect",
    td_file = "SCIFRCkksDialect.td",
    deps = [
        ":td_files",
    ],
)

add_heir_dialect_library(
    name = "ops_inc_gen",
    dialect = "SCIFRCkks",
    kind = "op",
    td_file = "SCIFRCkksOps.td",
    deps = [
        ":td_files",
        "@heir//lib/Dialect:td_files",
        "@heir//lib/Dialect/LWE/IR:td_files",
        "@heir//lib/Dialect/Polynomial/IR:td_files",
        "@llvm-project//mlir:BytecodeOpInterface",
    ],
)

add_heir_dialect_library(
    name = "types_inc_gen",
    dialect = "SCIFRCkks",
    kind = "type",
    td_file = "SCIFRCkksTypes.td",
    deps = [
        ":td_files",
    ],
)
