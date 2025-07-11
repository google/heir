load("@heir//lib/Dialect:dialect.bzl", "add_heir_dialect_library")
load("@llvm-project//mlir:tblgen.bzl", "td_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Dialect",
    srcs = [
        "LWEAttributes.cpp",
        "LWEDialect.cpp",
        "LWEOps.cpp",
        "LWETypes.cpp",
    ],
    hdrs = [
        "LWEAttributes.h",
        "LWEDialect.h",
        "LWEOps.h",
        "LWEPatterns.h",
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
        "@heir//lib/Utils:APIntUtils",
        "@heir//lib/Utils/Polynomial",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
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
    name = "dialect_inc_gen",
    dialect = "LWE",
    kind = "dialect",
    td_file = "LWEDialect.td",
    deps = [
        ":td_files",
    ],
)

add_heir_dialect_library(
    name = "attributes_inc_gen",
    dialect = "LWE",
    kind = "attribute",
    td_file = "LWEAttributes.td",
    deps = [
        ":td_files",
    ],
)

add_heir_dialect_library(
    name = "enums_inc_gen",
    dialect = "LWE",
    kind = "enum",
    td_file = "LWEAttributes.td",
    deps = [
        ":td_files",
    ],
)

add_heir_dialect_library(
    name = "types_inc_gen",
    dialect = "LWE",
    kind = "type",
    td_file = "LWETypes.td",
    deps = [
        ":td_files",
    ],
)

add_heir_dialect_library(
    name = "ops_inc_gen",
    dialect = "LWE",
    kind = "op",
    td_file = "LWEOps.td",
    deps = [
        ":td_files",
        "@heir//lib/Dialect/Polynomial/IR:td_files",
    ],
)
