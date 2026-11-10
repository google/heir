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
        "CheddarGPUDialect.cpp",
    ],
    hdrs = [
        "CheddarGPUDialect.h",
    ],
    deps = [
        ":dialect_inc_gen",
        "@llvm-project//mlir:IR",
    ],
)

td_library(
    name = "td_files",
    srcs = [
        "CheddarGPUDialect.td",
    ],
    includes = ["../../../.."],
    deps = [
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

add_heir_dialect_library(
    name = "dialect_inc_gen",
    dialect = "CheddarGPU",
    dialect_doc_name = "cheddar_gpu",
    kind = "dialect",
    td_file = "CheddarGPUDialect.td",
    deps = [
        ":td_files",
    ],
)
