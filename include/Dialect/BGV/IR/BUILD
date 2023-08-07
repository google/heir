# BGV, a dialect defining the BGV cryptosystem.

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

exports_files(
    [
        "BGVDialect.h",
        "BGVOps.h",
        "BGVTypes.h",
        "BGVAttributes.h",
    ],
)

td_library(
    name = "td_files",
    srcs = [
        "BGVAttributes.td",
        "BGVDialect.td",
        "BGVOps.td",
        "BGVTypes.td",
    ],
    includes = ["@heir//include"],
    deps = [
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "dialect_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-dialect-decls",
            ],
            "BGVDialect.h.inc",
        ),
        (
            [
                "-gen-dialect-defs",
            ],
            "BGVDialect.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "BGVDialect.td",
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
            ],
            "BGVTypes.h.inc",
        ),
        (
            [
                "-gen-typedef-defs",
            ],
            "BGVTypes.cpp.inc",
        ),
        (
            ["-gen-typedef-doc"],
            "BGVTypes.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "BGVTypes.td",
    deps = [
        ":attributes_inc_gen",
        ":dialect_inc_gen",
        ":td_files",
    ],
)

gentbl_cc_library(
    name = "ops_inc_gen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "BGVOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "BGVOps.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "BGVOps.td",
    deps = [
        ":dialect_inc_gen",
        ":td_files",
        ":types_inc_gen",
    ],
)

gentbl_cc_library(
    name = "attributes_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-attrdef-decls",
            ],
            "BGVAttributes.h.inc",
        ),
        (
            [
                "-gen-attrdef-defs",
            ],
            "BGVAttributes.cpp.inc",
        ),
        (
            ["-gen-attrdef-doc"],
            "BGVAttributes.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "BGVAttributes.td",
    deps = [
        ":dialect_inc_gen",
        ":td_files",
    ],
)
