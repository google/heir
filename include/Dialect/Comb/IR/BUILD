load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

exports_files(
    glob(["*.h"]),
)

td_library(
    name = "td_files",
    srcs = [
        "Comb.td",
        "Combinational.td",
    ],
    includes = ["include"],
    deps = [
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:FunctionInterfacesTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "dialect_inc_gen",
    includes = ["include"],
    tbl_outs = [
        (
            [
                "-gen-dialect-decls",
                "-dialect=comb",
            ],
            "CombDialect.h.inc",
        ),
        (
            [
                "-gen-dialect-defs",
                "-dialect=comb",
            ],
            "CombDialect.cpp.inc",
        ),
        (
            ["-gen-dialect-doc"],
            "CombDialect.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Comb.td",
    deps = [
        ":td_files",
        ":type_inc_gen",
    ],
)

gentbl_cc_library(
    name = "ops_inc_gen",
    includes = ["include"],
    tbl_outs = [
        (
            [
                "-gen-op-decls",
            ],
            "Comb.h.inc",
        ),
        (
            [
                "-gen-op-defs",
            ],
            "Comb.cpp.inc",
        ),
        (
            ["-gen-op-doc"],
            "CombOps.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Comb.td",
    deps = [
        ":dialect_inc_gen",
        ":td_files",
    ],
)

gentbl_cc_library(
    name = "type_inc_gen",
    includes = ["include"],
    tbl_outs = [
        (
            [
                "-gen-typedef-decls",
            ],
            "CombTypes.h.inc",
        ),
        (
            [
                "-gen-typedef-defs",
            ],
            "CombTypes.cpp.inc",
        ),
        (
            ["-gen-typedef-doc"],
            "CombTypes.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Comb.td",
    deps = [
        ":td_files",
    ],
)

gentbl_cc_library(
    name = "enum_inc_gen",
    includes = ["include"],
    tbl_outs = [
        (
            [
                "-gen-enum-decls",
            ],
            "CombEnums.h.inc",
        ),
        (
            [
                "-gen-enum-defs",
            ],
            "CombEnums.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Comb.td",
    deps = [
        ":dialect_inc_gen",
        ":td_files",
    ],
)
