# CKKS dialect

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Dialect",
    srcs = [
        "CKKSDialect.cpp",
    ],
    hdrs = [
        "CKKSDialect.h",
        "CKKSOps.h",
    ],
    deps = [
        ":dialect_inc_gen",
        ":ops_inc_gen",
        "@heir//lib/Dialect:FHEHelpers",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Support",
    ],
)

td_library(
    name = "td_files",
    srcs = [
        "CKKSDialect.td",
        "CKKSOps.td",
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
            "CKKSDialect.h.inc",
        ),
        (
            [
                "-gen-dialect-defs",
            ],
            "CKKSDialect.cpp.inc",
        ),
        (
            [
                "-gen-dialect-doc",
            ],
            "CKKSDialect.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "CKKSDialect.td",
    deps = [
        ":td_files",
    ],
)

gentbl_cc_library(
    name = "ops_inc_gen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "CKKSOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "CKKSOps.cpp.inc",
        ),
        (
            ["-gen-op-doc"],
            "CKKSOps.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "CKKSOps.td",
    deps = [
        ":dialect_inc_gen",
        ":td_files",
        "@heir//lib/Dialect/LWE/IR:td_files",
        "@llvm-project//mlir:PolynomialTdFiles",
    ],
)
