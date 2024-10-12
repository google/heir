load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Dialect",
    srcs = [
        "TfheRustBoolDialect.cpp",
    ],
    hdrs = [
        "TfheRustBoolAttributes.h",
        "TfheRustBoolDialect.h",
        "TfheRustBoolEnums.h",
        "TfheRustBoolOps.h",
        "TfheRustBoolTypes.h",
    ],
    deps = [
        ":attributes_inc_gen",
        ":dialect_inc_gen",
        ":enums_inc_gen",
        ":ops_inc_gen",
        ":types_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
    ],
)
# TfheRustBool, an exit dialect to the boolean tfhe-rs API

td_library(
    name = "td_files",
    srcs = [
        "TfheRustBoolAttributes.td",
        "TfheRustBoolDialect.td",
        "TfheRustBoolEnums.td",
        "TfheRustBoolOps.td",
        "TfheRustBoolTypes.td",
    ],
    # include from the heir-root to enable fully-qualified include-paths
    includes = ["../../../.."],
    deps = [
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "dialect_inc_gen",
    tbl_outs = [
        (
            ["-gen-dialect-decls"],
            "TfheRustBoolDialect.h.inc",
        ),
        (
            ["-gen-dialect-defs"],
            "TfheRustBoolDialect.cpp.inc",
        ),
        (
            ["-gen-dialect-doc"],
            "TfheRustBoolDialect.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "TfheRustBoolDialect.td",
    deps = [
        ":td_files",
    ],
)

gentbl_cc_library(
    name = "types_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-typedef-decls",
                "-typedefs-dialect=tfhe_rust_bool",
            ],
            "TfheRustBoolTypes.h.inc",
        ),
        (
            [
                "-gen-typedef-defs",
                "-typedefs-dialect=tfhe_rust_bool",
            ],
            "TfheRustBoolTypes.cpp.inc",
        ),
        (
            [
                "-gen-typedef-doc",
                "-typedefs-dialect=tfhe_rust_bool",
            ],
            "TfheRustBoolTypes.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "TfheRustBoolTypes.td",
    deps = [
        ":dialect_inc_gen",
        ":td_files",
    ],
)

gentbl_cc_library(
    name = "attributes_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-attrdef-decls",
                "-typedefs-dialect=tfhe_rust_bool",
            ],
            "TfheRustBoolAttributes.h.inc",
        ),
        (
            [
                "-gen-attrdef-defs",
                "-typedefs-dialect=tfhe_rust_bool",
            ],
            "TfheRustBoolAttributes.cpp.inc",
        ),
        (
            [
                "-gen-attrdef-doc",
                "-typedefs-dialect=tfhe_rust_bool",
            ],
            "TfheRustBoolAttributes.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "TfheRustBoolAttributes.td",
    deps = [
        ":dialect_inc_gen",
        ":enums_inc_gen",
        ":td_files",
        "@heir//lib/Dialect/CGGI/IR:td_files",
    ],
)

gentbl_cc_library(
    name = "ops_inc_gen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "TfheRustBoolOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "TfheRustBoolOps.cpp.inc",
        ),
        (
            ["-gen-op-doc"],
            "TfheRustBoolOps.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "TfheRustBoolOps.td",
    deps = [
        ":attributes_inc_gen",
        ":dialect_inc_gen",
        ":td_files",
        ":types_inc_gen",
    ],
)

gentbl_cc_library(
    name = "enums_inc_gen",
    tbl_outs = [
        (
            ["-gen-enum-decls"],
            "TfheRustBoolEnums.h.inc",
        ),
        (
            ["--gen-enum-defs"],
            "TfheRustBoolEnums.cpp.inc",
        ),
        (
            ["-gen-attrdef-doc"],
            "TfheRustBoolEnums.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "TfheRustBoolEnums.td",
    deps = [
        ":td_files",
        "@heir//lib/Dialect/CGGI/IR:td_files",
    ],
)
