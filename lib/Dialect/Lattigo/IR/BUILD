# Lattigo dialect implementation

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
        "LattigoDialect.cpp",
    ],
    hdrs = [
        "LattigoAttributes.h",
        "LattigoDialect.h",
        "LattigoOps.h",
        "LattigoTypes.h",
    ],
    deps = [
        "attributes_inc_gen",
        "dialect_inc_gen",
        "ops_inc_gen",
        "types_inc_gen",
        ":LattigoAttributes",
        ":LattigoOps",
        ":LattigoTypes",
        "@heir//lib/Dialect:HEIRInterfaces",
        "@heir//lib/Utils/Tablegen:InPlaceOpInterface",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
    ],
)

cc_library(
    name = "LattigoAttributes",
    srcs = [
        "LattigoAttributes.cpp",
    ],
    hdrs = [
        "LattigoAttributes.h",
        "LattigoDialect.h",
    ],
    deps = [
        ":attributes_inc_gen",
        ":dialect_inc_gen",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "LattigoTypes",
    srcs = [
        "LattigoTypes.cpp",
    ],
    hdrs = [
        "LattigoAttributes.h",
        "LattigoDialect.h",
        "LattigoTypes.h",
    ],
    deps = [
        ":LattigoAttributes",
        ":attributes_inc_gen",
        ":dialect_inc_gen",
        ":types_inc_gen",
        "@heir//lib/Dialect:HEIRInterfaces",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "LattigoOps",
    srcs = [
        "LattigoOps.cpp",
    ],
    hdrs = [
        "LattigoDialect.h",
        "LattigoOps.h",
        "LattigoTypes.h",
    ],
    deps = [
        ":LattigoAttributes",
        ":LattigoTypes",
        ":dialect_inc_gen",
        ":ops_inc_gen",
        ":types_inc_gen",
        "@heir//lib/Dialect:HEIRInterfaces",
        "@heir//lib/Utils/Tablegen:InPlaceOpInterface",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Support",
    ],
)

td_library(
    name = "td_files",
    srcs = [
        "LattigoAttributes.td",
        "LattigoBGVAttributes.td",
        "LattigoBGVOps.td",
        "LattigoBGVTypes.td",
        "LattigoCKKSAttributes.td",
        "LattigoCKKSOps.td",
        "LattigoCKKSTypes.td",
        "LattigoDialect.td",
        "LattigoOps.td",
        "LattigoRLWEOps.td",
        "LattigoRLWETypes.td",
        "LattigoTypes.td",
    ],
    # include from the heir-root to enable fully-qualified include-paths
    includes = ["../../../.."],
    deps = [
        "@heir//lib/Dialect:td_files",
        "@heir//lib/Utils/Tablegen:td_files",
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

add_heir_dialect_library(
    name = "dialect_inc_gen",
    dialect = "Lattigo",
    kind = "dialect",
    td_file = "LattigoDialect.td",
    deps = [
        ":td_files",
    ],
)

add_heir_dialect_library(
    name = "attributes_inc_gen",
    dialect = "Lattigo",
    kind = "attribute",
    td_file = "LattigoAttributes.td",
    deps = [
        ":td_files",
    ],
)

add_heir_dialect_library(
    name = "types_inc_gen",
    dialect = "Lattigo",
    kind = "type",
    td_file = "LattigoTypes.td",
    deps = [
        ":td_files",
    ],
)

add_heir_dialect_library(
    name = "ops_inc_gen",
    dialect = "Lattigo",
    kind = "op",
    td_file = "LattigoOps.td",
    deps = [
        ":td_files",
    ],
)
