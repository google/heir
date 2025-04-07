# BGV dialect

load("@heir//lib/Dialect:dialect.bzl", "add_heir_dialect_library")
load("@llvm-project//mlir:tblgen.bzl", "td_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Dialect",
    srcs = [
        "BGVDialect.cpp",
        "BGVOps.cpp",
    ],
    hdrs = [
        "BGVAttributes.h",
        "BGVDialect.h",
        "BGVEnums.h",
        "BGVOps.h",
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
        "BGVAttributes.td",
        "BGVDialect.td",
        "BGVEnums.td",
        "BGVOps.td",
    ],
    deps = [
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

add_heir_dialect_library(
    name = "dialect_inc_gen",
    dialect = "BGV",
    kind = "dialect",
    td_file = "BGVDialect.td",
    deps = [
        ":td_files",
    ],
)

add_heir_dialect_library(
    name = "ops_inc_gen",
    dialect = "BGV",
    kind = "op",
    td_file = "BGVOps.td",
    deps = [
        ":td_files",
        "@heir//lib/Dialect/LWE/IR:td_files",
        "@heir//lib/Dialect/Polynomial/IR:td_files",
    ],
)

add_heir_dialect_library(
    name = "attributes_inc_gen",
    dialect = "BGV",
    kind = "attribute",
    td_file = "BGVAttributes.td",
    deps = [
        ":td_files",
    ],
)

add_heir_dialect_library(
    name = "enums_inc_gen",
    dialect = "BGV",
    kind = "enum",
    td_file = "BGVEnums.td",
    deps = [
        ":td_files",
    ],
)
