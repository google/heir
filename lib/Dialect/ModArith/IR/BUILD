# ModArith dialect

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Dialect",
    srcs = [
        "ModArithDialect.cpp",
    ],
    hdrs = [
        "ModArithDialect.h",
        "ModArithOps.h",
        "ModArithTypes.h",
    ],
    deps = [
        ":dialect_inc_gen",
        ":ops_inc_gen",
        ":types_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Support",
    ],
)

td_library(
    name = "td_files",
    srcs = [
        "ModArithDialect.td",
        "ModArithOps.td",
        "ModArithTypes.td",
    ],
    # include from the heir-root to enable fully-qualified include-paths
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
            "ModArithDialect.h.inc",
        ),
        (
            [
                "-gen-dialect-defs",
            ],
            "ModArithDialect.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ModArithDialect.td",
    deps = [
        ":td_files",
    ],
)

gentbl_cc_library(
    name = "types_inc_gen",
    tbl_outs = [
        (
            ["-gen-typedef-decls"],
            "ModArithTypes.h.inc",
        ),
        (
            ["-gen-typedef-defs"],
            "ModArithTypes.cpp.inc",
        ),
        (
            ["-gen-typedef-doc"],
            "ModArithTypes.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ModArithTypes.td",
    deps = [
        ":dialect_inc_gen",
        ":td_files",
    ],
)

gentbl_cc_library(
    name = "ops_inc_gen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "ModArithOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "ModArithOps.cpp.inc",
        ),
        (
            ["-gen-op-doc"],
            "ModArithOps.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ModArithOps.td",
    deps = [
        ":dialect_inc_gen",
        ":td_files",
        "@heir//lib/Dialect/Polynomial/IR:td_files",
    ],
)
