# EncryptedArith, a dialect defining encrypted analogues of builtin types

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

exports_files(
    [
        "EncryptedArithDialect.h",
        "EncryptedArithTypes.h",
    ],
)

td_library(
    name = "td_files",
    srcs = [
        "EncryptedArithDialect.td",
        "EncryptedArithTypes.td",
    ],
    includes = ["@heir//include"],
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "dialect_inc_gen",
    tbl_outs = [
        (
            ["-gen-dialect-decls"],
            "EncryptedArithDialect.h.inc",
        ),
        (
            ["-gen-dialect-defs"],
            "EncryptedArithDialect.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "EncryptedArithDialect.td",
    deps = [
        ":td_files",
    ],
)

gentbl_cc_library(
    name = "types_inc_gen",
    tbl_outs = [
        (
            ["-gen-typedef-decls"],
            "EncryptedArithTypes.h.inc",
        ),
        (
            ["-gen-typedef-defs"],
            "EncryptedArithTypes.cpp.inc",
        ),
        (
            ["-gen-typedef-doc"],
            "EncryptedArithTypes.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "EncryptedArithTypes.td",
    deps = [
        ":td_files",
    ],
)
