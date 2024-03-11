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
        "BGVTraits.h",
    ],
)

td_library(
    name = "td_files",
    srcs = [
        "BGVDialect.td",
        "BGVOps.td",
    ],
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
        (
            [
                "-gen-dialect-doc",
            ],
            "BGVDialect.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "BGVDialect.td",
    deps = [
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
        (
            ["-gen-op-doc"],
            "BGVOps.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "BGVOps.td",
    deps = [
        ":dialect_inc_gen",
        ":td_files",
        "@heir//include/Dialect/LWE/IR:td_files",
        "@heir//include/Dialect/Polynomial/IR:td_files",
    ],
)
