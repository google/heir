# Rotom dialect implementation

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
        "RotomAttributes.cpp",
        "RotomDialect.cpp",
    ],
    hdrs = [
        "RotomAttributes.h",
        "RotomDialect.h",
    ],
    deps = [
        ":attributes_inc_gen",
        ":dialect_inc_gen",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

td_library(
    name = "td_files",
    srcs = [
        "RotomAttributes.td",
        "RotomDialect.td",
    ],
    includes = ["../../../.."],
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

add_heir_dialect_library(
    name = "dialect_inc_gen",
    dialect = "Rotom",
    dialect_doc_name = "rotom",
    kind = "dialect",
    td_file = "RotomDialect.td",
    deps = [
        ":td_files",
    ],
)

add_heir_dialect_library(
    name = "attributes_inc_gen",
    dialect = "Rotom",
    dialect_doc_name = "rotom",
    kind = "attribute",
    td_file = "RotomAttributes.td",
    deps = [
        ":td_files",
    ],
)
