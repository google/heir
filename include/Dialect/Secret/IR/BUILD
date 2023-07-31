# Secret, a dialect defining secretnomials in a secretnomial ring and their operations.

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

exports_files(
    [
        "SecretDialect.h",
        "SecretOps.h",
        "SecretTypes.h",
    ],
)

td_library(
    name = "td_files",
    srcs = [
        "SecretDialect.td",
        "SecretOps.td",
        "SecretTypes.td",
    ],
    includes = ["@heir//include"],
    deps = [
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "dialect_inc_gen",
    tbl_outs = [
        (
            ["-gen-dialect-decls"],
            "SecretDialect.h.inc",
        ),
        (
            ["-gen-dialect-defs"],
            "SecretDialect.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "SecretDialect.td",
    deps = [
        ":td_files",
    ],
)

gentbl_cc_library(
    name = "types_inc_gen",
    tbl_outs = [
        (
            ["-gen-typedef-decls"],
            "SecretTypes.h.inc",
        ),
        (
            ["-gen-typedef-defs"],
            "SecretTypes.cpp.inc",
        ),
        (
            ["-gen-typedef-doc"],
            "SecretTypes.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "SecretTypes.td",
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
            "SecretOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "SecretOps.cpp.inc",
        ),
        (
            ["-gen-op-doc"],
            "SecretOps.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "SecretOps.td",
    deps = [
        ":dialect_inc_gen",
        ":td_files",
        ":types_inc_gen",
    ],
)
