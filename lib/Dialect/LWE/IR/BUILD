load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Dialect",
    srcs = [
        "LWEDialect.cpp",
    ],
    hdrs = [
        "LWEAttributes.h",
        "LWEDialect.h",
        "LWEOps.h",
        "LWETraits.h",
        "LWETypes.h",
    ],
    deps = [
        ":attributes_inc_gen",
        ":dialect_inc_gen",
        ":ops_inc_gen",
        ":types_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:PolynomialDialect",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "Patterns",
    hdrs = [
        "LWEPatterns.h",
    ],
    deps = [
        ":Dialect",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

td_library(
    name = "td_files",
    srcs = [
        "LWEAttributes.td",
        "LWEDialect.td",
        "LWEOps.td",
        "LWETraits.td",
        "LWETypes.td",
    ],
    # include from the heir-root to enable fully-qualified include-paths
    includes = ["../../../.."],
    deps = [
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "dialect_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-dialect-decls",
            ],
            "LWEDialect.h.inc",
        ),
        (
            [
                "-gen-dialect-defs",
            ],
            "LWEDialect.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "LWEDialect.td",
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
            "LWEAttributes.h.inc",
        ),
        (
            [
                "-gen-attrdef-defs",
            ],
            "LWEAttributes.cpp.inc",
        ),
        (
            ["-gen-attrdef-doc"],
            "LWEAttributes.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "LWEAttributes.td",
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
                "-typedefs-dialect=lwe",
            ],
            "LWETypes.h.inc",
        ),
        (
            [
                "-gen-typedef-defs",
                "-typedefs-dialect=lwe",
            ],
            "LWETypes.cpp.inc",
        ),
        (
            [
                "-gen-typedef-doc",
                "-typedefs-dialect=lwe",
            ],
            "LWETypes.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "LWETypes.td",
    deps = [
        ":attributes_inc_gen",
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
            "LWEOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "LWEOps.cpp.inc",
        ),
        (
            ["-gen-op-doc"],
            "LWEOps.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "LWEOps.td",
    deps = [
        ":dialect_inc_gen",
        ":td_files",
        ":types_inc_gen",
        "@llvm-project//mlir:PolynomialTdFiles",
    ],
)
