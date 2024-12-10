# Lattigo dialect implementation

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Dialect",
    srcs = [
        "LattigoDialect.cpp",
    ],
    hdrs = [
        "LattigoAttributes.h",
        "LattigoDialect.h",
        "LattigoOps.h",
        "LattigoTypes.h",
    ],
    deps = [
        "attributes_inc_gen",
        "dialect_inc_gen",
        "ops_inc_gen",
        "types_inc_gen",
        ":LattigoAttributes",
        ":LattigoOps",
        ":LattigoTypes",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "LattigoAttributes",
    srcs = [
        "LattigoAttributes.cpp",
    ],
    hdrs = [
        "LattigoAttributes.h",
        "LattigoDialect.h",
    ],
    deps = [
        ":attributes_inc_gen",
        ":dialect_inc_gen",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "LattigoTypes",
    srcs = [
        "LattigoTypes.cpp",
    ],
    hdrs = [
        "LattigoAttributes.h",
        "LattigoDialect.h",
        "LattigoTypes.h",
    ],
    deps = [
        ":LattigoAttributes",
        ":attributes_inc_gen",
        ":dialect_inc_gen",
        ":types_inc_gen",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "LattigoOps",
    srcs = [
        "LattigoOps.cpp",
    ],
    hdrs = [
        "LattigoDialect.h",
        "LattigoOps.h",
        "LattigoTypes.h",
    ],
    deps = [
        ":LattigoAttributes",
        ":LattigoTypes",
        ":dialect_inc_gen",
        ":ops_inc_gen",
        ":types_inc_gen",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

td_library(
    name = "td_files",
    srcs = [
        "LattigoAttributes.td",
        "LattigoBGVAttributes.td",
        "LattigoBGVOps.td",
        "LattigoBGVTypes.td",
        "LattigoDialect.td",
        "LattigoOps.td",
        "LattigoRLWEOps.td",
        "LattigoRLWETypes.td",
        "LattigoTypes.td",
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
            "LattigoDialect.h.inc",
        ),
        (
            [
                "-gen-dialect-defs",
            ],
            "LattigoDialect.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "LattigoDialect.td",
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
            ],
            "LattigoAttributes.h.inc",
        ),
        (
            [
                "-gen-attrdef-defs",
            ],
            "LattigoAttributes.cpp.inc",
        ),
        (
            ["-gen-attrdef-doc"],
            "LattigoAttributes.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "LattigoAttributes.td",
    deps = [
        ":dialect_inc_gen",
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
            "LattigoTypes.h.inc",
        ),
        (
            [
                "-gen-typedef-defs",
            ],
            "LattigoTypes.cpp.inc",
        ),
        (
            ["-gen-typedef-doc"],
            "LattigoTypes.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "LattigoTypes.td",
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
            "LattigoOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "LattigoOps.cpp.inc",
        ),
        (
            ["-gen-op-doc"],
            "LattigoOps.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "LattigoOps.td",
    deps = [
        ":dialect_inc_gen",
        ":td_files",
        ":types_inc_gen",
    ],
)
