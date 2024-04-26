load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Dialect",
    srcs = [
        "OpenfheDialect.cpp",
    ],
    hdrs = [
        "OpenfheDialect.h",
        "OpenfheOps.h",
        "OpenfheTypes.h",
    ],
    deps = [
        ":dialect_inc_gen",
        ":ops_inc_gen",
        ":types_inc_gen",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
    ],
)

td_library(
    name = "td_files",
    srcs = [
        "OpenfheDialect.td",
        "OpenfheOps.td",
        "OpenfheTypes.td",
    ],
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
            [
                "-gen-dialect-decls",
            ],
            "OpenfheDialect.h.inc",
        ),
        (
            [
                "-gen-dialect-defs",
            ],
            "OpenfheDialect.cpp.inc",
        ),
        (
            [
                "-gen-dialect-doc",
            ],
            "OpenfheDialect.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "OpenfheDialect.td",
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
            "OpenfheTypes.h.inc",
        ),
        (
            [
                "-gen-typedef-defs",
            ],
            "OpenfheTypes.cpp.inc",
        ),
        (
            ["-gen-typedef-doc"],
            "OpenfheTypes.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "OpenfheTypes.td",
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
            "OpenfheOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "OpenfheOps.cpp.inc",
        ),
        (
            ["-gen-op-doc"],
            "OpenfheOps.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "OpenfheOps.td",
    deps = [
        ":dialect_inc_gen",
        ":td_files",
        ":types_inc_gen",
        "@heir//lib/Dialect/LWE/IR:td_files",
    ],
)
