load("@heir//lib/Dialect:dialect.bzl", "add_heir_dialect_library")
load("@llvm-project//mlir:tblgen.bzl", "td_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

td_library(
    name = "td_files",
    srcs = [
        "KeyMgmtDialect.td",
        "KeyMgmtOps.td",
        "KeyMgmtTypes.td",
    ],
    includes = ["../../../.."],
    deps = [
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

add_heir_dialect_library(
    name = "dialect_inc_gen",
    dialect = "KeyMgmt",
    dialect_doc_name = "key_mgmt",
    kind = "dialect",
    td_file = "KeyMgmtDialect.td",
    deps = [
        ":td_files",
    ],
)

add_heir_dialect_library(
    name = "types_inc_gen",
    dialect = "KeyMgmt",
    dialect_doc_name = "key_mgmt",
    kind = "type",
    td_file = "KeyMgmtTypes.td",
    deps = [
        ":dialect_inc_gen",
        ":td_files",
    ],
)

add_heir_dialect_library(
    name = "ops_inc_gen",
    dialect = "KeyMgmt",
    dialect_doc_name = "key_mgmt",
    kind = "op",
    td_file = "KeyMgmtOps.td",
    deps = [
        ":dialect_inc_gen",
        ":td_files",
        ":types_inc_gen",
    ],
)

cc_library(
    name = "Dialect",
    srcs = [
        "KeyMgmtDialect.cpp",
        "KeyMgmtOps.cpp",
    ],
    hdrs = [
        "KeyMgmtDialect.h",
        "KeyMgmtOps.h",
        "KeyMgmtTypes.h",
    ],
    deps = [
        ":dialect_inc_gen",
        ":ops_inc_gen",
        ":types_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)
