load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Dialect",
    srcs = [
        "CGGIDialect.cpp",
        "CGGIOps.cpp",
    ],
    hdrs = [
        "CGGIAttributes.h",
        "CGGIDialect.h",
        "CGGIOps.h",
    ],
    deps = [
        ":attributes_inc_gen",
        ":dialect_inc_gen",
        ":ops_inc_gen",
        "@heir//lib/Dialect:HEIRInterfaces",
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
        "CGGIAttributes.td",
        "CGGIDialect.td",
        "CGGIOps.td",
    ],
    # include from the heir-root to enable fully-qualified include-paths
    includes = ["../../../.."],
    deps = [
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
            "CGGIDialect.h.inc",
        ),
        (
            ["-gen-dialect-defs"],
            "CGGIDialect.cpp.inc",
        ),
        (
            ["-gen-dialect-doc"],
            "CGGIDialect.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "CGGIDialect.td",
    deps = [
        ":td_files",
    ],
)

gentbl_cc_library(
    name = "ops_inc_gen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "CGGIOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "CGGIOps.cpp.inc",
        ),
        (
            ["-gen-op-doc"],
            "CGGIOps.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "CGGIOps.td",
    deps = [
        ":dialect_inc_gen",
        ":td_files",
        "@heir//lib/Dialect:td_files",
        "@heir//lib/Dialect/LWE/IR:td_files",
        "@llvm-project//mlir:PolynomialTdFiles",
    ],
)

gentbl_cc_library(
    name = "attributes_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-attrdef-decls",
            ],
            "CGGIAttributes.h.inc",
        ),
        (
            [
                "-gen-attrdef-defs",
            ],
            "CGGIAttributes.cpp.inc",
        ),
        (
            ["-gen-attrdef-doc"],
            "CGGIAttributes.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "CGGIAttributes.td",
    deps = [
        ":dialect_inc_gen",
        ":td_files",
    ],
)
