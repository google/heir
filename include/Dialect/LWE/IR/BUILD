# LWE, a dialect defining the LWE cryptosystem.

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

exports_files(
    [
        "LWEDialect.h",
        "LWEAttributes.h",
    ],
)

td_library(
    name = "td_files",
    srcs = [
        "LWEAttributes.td",
        "LWEDialect.td",
    ],
    includes = ["@heir//include"],
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
