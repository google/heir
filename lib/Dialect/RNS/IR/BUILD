# RNS dialect implementation

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Dialect",
    srcs = [
        "RNSDialect.cpp",
    ],
    hdrs = [
        "RNSDialect.h",
        "RNSOps.h",
        "RNSTypeInterfaces.h",
        "RNSTypes.h",
    ],
    deps = [
        ":RNSOps",
        ":RNSTypes",
        ":dialect_inc_gen",
        ":ops_inc_gen",
        ":type_interfaces_inc_gen",
        ":types_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:PolynomialDialect",
    ],
)

cc_library(
    name = "RNSTypes",
    srcs = [
        "RNSTypes.cpp",
    ],
    hdrs = [
        "RNSDialect.h",
        "RNSTypeInterfaces.h",
        "RNSTypes.h",
    ],
    deps = [
        ":dialect_inc_gen",
        ":type_interfaces_inc_gen",
        ":types_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:PolynomialDialect",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "RNSOps",
    srcs = [
        "RNSOps.cpp",
    ],
    hdrs = [
        "RNSDialect.h",
        "RNSOps.h",
        "RNSTypes.h",
    ],
    deps = [
        ":RNSTypes",
        ":dialect_inc_gen",
        ":ops_inc_gen",
        ":types_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:PolynomialDialect",
    ],
)

td_library(
    name = "td_files",
    srcs = [
        "RNSDialect.td",
        "RNSOps.td",
        "RNSTypeInterfaces.td",
        "RNSTypes.td",
    ],
    # include from the heir-root to enable fully-qualified include-paths
    includes = ["../../../.."],
    deps = [
        "@llvm-project//mlir:BuiltinDialectTdFiles",
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
            "RNSDialect.h.inc",
        ),
        (
            [
                "-gen-dialect-defs",
            ],
            "RNSDialect.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "RNSDialect.td",
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
            "RNSTypes.h.inc",
        ),
        (
            [
                "-gen-typedef-defs",
            ],
            "RNSTypes.cpp.inc",
        ),
        (
            ["-gen-typedef-doc"],
            "RNSTypes.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "RNSTypes.td",
    deps = [
        ":dialect_inc_gen",
        ":td_files",
        "@llvm-project//mlir:PolynomialTdFiles",
    ],
)

gentbl_cc_library(
    name = "ops_inc_gen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "RNSOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "RNSOps.cpp.inc",
        ),
        (
            ["-gen-op-doc"],
            "RNSOps.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "RNSOps.td",
    deps = [
        ":dialect_inc_gen",
        ":td_files",
        ":types_inc_gen",
        "@llvm-project//mlir:PolynomialTdFiles",
    ],
)

gentbl_cc_library(
    name = "type_interfaces_inc_gen",
    tbl_outs = [
        (
            ["--gen-type-interface-decls"],
            "RNSTypeInterfaces.h.inc",
        ),
        (
            ["--gen-type-interface-defs"],
            "RNSTypeInterfaces.cpp.inc",
        ),
        (
            ["-gen-type-interface-docs"],
            "RNSTypeInterfaces.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "RNSTypeInterfaces.td",
    deps = [
        "@llvm-project//mlir:BuiltinDialectTdFiles",
    ],
)
