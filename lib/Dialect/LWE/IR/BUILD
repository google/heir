load("@heir//lib/Dialect:dialect.bzl", "add_heir_dialect_library")
load("@llvm-project//mlir:tblgen.bzl", "td_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Dialect",
    srcs = [
        "LWEDialect.cpp",
    ],
    hdrs = [
        "LWEAttributes.h",
        "LWEDialect.h",
        "LWEOps.h",
        "LWETraits.h",
        "LWETypes.h",
    ],
    deps = [
        ":attributes_inc_gen",
        ":dialect_inc_gen",
        ":enums_inc_gen",
        ":ops_inc_gen",
        ":types_inc_gen",
        "@heir//lib/Dialect/ModArith/IR:Dialect",
        "@heir//lib/Dialect/Polynomial/IR:Dialect",
        "@heir//lib/Dialect/RNS/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "Patterns",
    hdrs = [
        "LWEPatterns.h",
    ],
    deps = [
        ":Dialect",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

td_library(
    name = "td_files",
    srcs = [
        "LWEAttributes.td",
        "LWEDialect.td",
        "LWEOps.td",
        "LWETraits.td",
        "LWETypes.td",
        "NewLWEAttributes.td",
        "NewLWETypes.td",
    ],
    # include from the heir-root to enable fully-qualified include-paths
    includes = ["../../../.."],
    deps = [
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

add_heir_dialect_library(
    dialect = "LWE",
    kind = "dialect",
)

add_heir_dialect_library(
    dialect = "LWE",
    kind = "attribute",
)

add_heir_dialect_library(
    dialect = "LWE",
    kind = "enum",
    td_file = "LWEAttributes.td",
)

add_heir_dialect_library(
    dialect = "LWE",
    kind = "type",
    deps = [
        ":attributes_inc_gen",
        ":enums_inc_gen",
        "@heir//lib/Dialect/Polynomial/IR:td_files",
    ],
)

add_heir_dialect_library(
    dialect = "LWE",
    kind = "op",
    deps = [
        ":types_inc_gen",
        "@heir//lib/Dialect/Polynomial/IR:td_files",
    ],
)
