# Mgmt dialect implementation

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Dialect",
    srcs = [
        "MgmtDialect.cpp",
    ],
    hdrs = [
        "MgmtAttributes.h",
        "MgmtDialect.h",
        "MgmtOps.h",
    ],
    deps = [
        "attributes_inc_gen",
        "dialect_inc_gen",
        "ops_inc_gen",
        ":MgmtOps",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
    ],
)

cc_library(
    name = "MgmtOps",
    srcs = [
        "MgmtOps.cpp",
    ],
    hdrs = [
        "MgmtDialect.h",
        "MgmtOps.h",
    ],
    deps = [
        ":dialect_inc_gen",
        ":ops_inc_gen",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
    ],
)

td_library(
    name = "td_files",
    srcs = [
        "MgmtAttributes.td",
        "MgmtDialect.td",
        "MgmtOps.td",
    ],
    # include from the heir-root to enable fully-qualified include-paths
    includes = ["../../../.."],
    deps = [
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
            "MgmtDialect.h.inc",
        ),
        (
            [
                "-gen-dialect-defs",
            ],
            "MgmtDialect.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "MgmtDialect.td",
    deps = [
        ":td_files",
    ],
)

gentbl_cc_library(
    name = "attributes_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-attrdef-decls",
                "-attrdefs-dialect=mgmt",
            ],
            "MgmtAttributes.h.inc",
        ),
        (
            [
                "-gen-attrdef-defs",
                "-attrdefs-dialect=mgmt",
            ],
            "MgmtAttributes.cpp.inc",
        ),
        (
            ["-gen-attrdef-doc"],
            "MgmtAttributes.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "MgmtAttributes.td",
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
            "MgmtOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "MgmtOps.cpp.inc",
        ),
        (
            ["-gen-op-doc"],
            "MgmtOps.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "MgmtOps.td",
    deps = [
        ":dialect_inc_gen",
        ":td_files",
    ],
)
