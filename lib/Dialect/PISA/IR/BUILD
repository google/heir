# PISA dialect implementation

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Dialect",
    srcs = [
        "PISADialect.cpp",
    ],
    hdrs = [
        "PISADialect.h",
        "PISAOps.h",
    ],
    deps = [
        "dialect_inc_gen",
        "ops_inc_gen",
        ":PISAOps",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "PISAOps",
    srcs = [
        "PISAOps.cpp",
    ],
    hdrs = [
        "PISADialect.h",
        "PISAOps.h",
    ],
    deps = [
        ":dialect_inc_gen",
        ":ops_inc_gen",
        "@heir//lib/Dialect/ModArith/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Support",
    ],
)

td_library(
    name = "td_files",
    srcs = [
        "PISADialect.td",
        "PISAOps.td",
    ],
    # include from the heir - root to enable fully - qualified include - paths
    includes = ["../../../.."],
    deps = [
        "@heir//lib/Utils/DRR",
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "dialect_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-dialect-decls",
            ],
            "PISADialect.h.inc",
        ),
        (
            [
                "-gen-dialect-defs",
            ],
            "PISADialect.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "PISADialect.td",
    deps = [
        ":td_files",
    ],
)

gentbl_cc_library(
    name = "ops_inc_gen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "PISAOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "PISAOps.cpp.inc",
        ),
        (
            ["-gen-op-doc"],
            "PISAOps.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "PISAOps.td",
    deps = [
        ":dialect_inc_gen",
        ":td_files",
    ],
)
