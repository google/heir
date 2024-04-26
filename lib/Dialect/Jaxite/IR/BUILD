# Jaxite, an exit dialect to Jaxite API

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Dialect",
    srcs = ["JaxiteDialect.cpp"],
    hdrs = [
        "JaxiteDialect.h",
        "JaxiteOps.h",
        "JaxiteTypes.h",
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
        "JaxiteDialect.td",
        "JaxiteOps.td",
        "JaxiteTypes.td",
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
            "JaxiteDialect.h.inc",
        ),
        (
            [
                "-gen-dialect-defs",
            ],
            "JaxiteDialect.cpp.inc",
        ),
        (
            [
                "-gen-dialect-doc",
            ],
            "JaxiteDialect.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "JaxiteDialect.td",
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
                "-typedefs-dialect=jaxite",
            ],
            "JaxiteTypes.h.inc",
        ),
        (
            [
                "-gen-typedef-defs",
                "-typedefs-dialect=jaxite",
            ],
            "JaxiteTypes.cpp.inc",
        ),
        (
            [
                "-gen-typedef-doc",
                "-typedefs-dialect=jaxite",
            ],
            "JaxiteTypes.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "JaxiteTypes.td",
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
            "JaxiteOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "JaxiteOps.cpp.inc",
        ),
        (
            ["-gen-op-doc"],
            "JaxiteOps.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "JaxiteOps.td",
    deps = [
        ":dialect_inc_gen",
        ":td_files",
        ":types_inc_gen",
        "@heir//lib/Dialect/LWE/IR:td_files",
    ],
)
