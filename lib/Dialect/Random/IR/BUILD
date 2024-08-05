# Random dialect

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Dialect",
    srcs = [
        "RandomDialect.cpp",
    ],
    hdrs = [
        "RandomDialect.h",
        "RandomEnums.h",
        "RandomOps.h",
        "RandomTypes.h",
    ],
    deps = [
        ":dialect_inc_gen",
        ":enums_inc_gen",
        ":ops_inc_gen",
        ":types_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Support",
    ],
)

td_library(
    name = "td_files",
    srcs = [
        "RandomDialect.td",
        "RandomEnums.td",
        "RandomOps.td",
        "RandomTypes.td",
    ],
    # include from the heir-root to enable fully-qualified include-paths
    includes = ["../../../.."],
    deps = [
        "@heir//lib/DRR",
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "enums_inc_gen",
    tbl_outs = [
        (
            ["-gen-enum-decls"],
            "RandomEnums.h.inc",
        ),
        (
            ["--gen-enum-defs"],
            "RandomEnums.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "RandomEnums.td",
    deps = [
        ":td_files",
        "@llvm-project//mlir:PolynomialTdFiles",
    ],
)

gentbl_cc_library(
    name = "dialect_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-dialect-decls",
            ],
            "RandomDialect.h.inc",
        ),
        (
            [
                "-gen-dialect-defs",
            ],
            "RandomDialect.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "RandomDialect.td",
    deps = [
        ":td_files",
    ],
)

gentbl_cc_library(
    name = "ops_inc_gen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "RandomOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "RandomOps.cpp.inc",
        ),
        (
            ["-gen-op-doc"],
            "RandomOps.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "RandomOps.td",
    deps = [
        ":dialect_inc_gen",
        ":td_files",
        "@llvm-project//mlir:PolynomialTdFiles",
    ],
)

gentbl_cc_library(
    name = "types_inc_gen",
    tbl_outs = [
        (
            ["-gen-typedef-decls"],
            "RandomTypes.h.inc",
        ),
        (
            ["--gen-typedef-defs"],
            "RandomTypes.cpp.inc",
        ),
        (
            ["-gen-typedef-doc"],
            "RandomTypes.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "RandomTypes.td",
    deps = [
        ":dialect_inc_gen",
        ":enums_inc_gen",
        ":td_files",
        "@llvm-project//mlir:PolynomialTdFiles",
    ],
)
