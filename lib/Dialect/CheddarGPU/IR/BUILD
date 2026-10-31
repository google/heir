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
        "CheddarGPUAttributes.cpp",
        "CheddarGPUDialect.cpp",
        "CheddarGPUOps.cpp",
    ],
    hdrs = [
        "CheddarGPUAttributes.h",
        "CheddarGPUDialect.h",
        "CheddarGPUEnums.h",
        "CheddarGPUOps.h",
    ],
    deps = [
        ":attributes_inc_gen",
        ":dialect_inc_gen",
        ":enums_inc_gen",
        ":ops_inc_gen",
        "@heir//lib/Utils/CheddarRef:params",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:BytecodeOpInterface",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
    ],
)

td_library(
    name = "td_files",
    srcs = [
        "CheddarGPUAttributes.td",
        "CheddarGPUDialect.td",
        "CheddarGPUEnums.td",
        "CheddarGPUOps.td",
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
    dialect = "CheddarGPU",
    dialect_doc_name = "cheddar_gpu",
    kind = "dialect",
    td_file = "CheddarGPUDialect.td",
    deps = [
        ":td_files",
    ],
)

add_heir_dialect_library(
    name = "enums_inc_gen",
    dialect = "CheddarGPU",
    dialect_doc_name = "cheddar_gpu",
    kind = "enum",
    td_file = "CheddarGPUEnums.td",
    deps = [
        ":td_files",
    ],
)

add_heir_dialect_library(
    name = "attributes_inc_gen",
    dialect = "CheddarGPU",
    dialect_doc_name = "cheddar_gpu",
    kind = "attribute",
    td_file = "CheddarGPUAttributes.td",
    deps = [
        ":td_files",
    ],
)

add_heir_dialect_library(
    name = "ops_inc_gen",
    dialect = "CheddarGPU",
    dialect_doc_name = "cheddar_gpu",
    kind = "op",
    td_file = "CheddarGPUOps.td",
    deps = [
        ":td_files",
    ],
)
