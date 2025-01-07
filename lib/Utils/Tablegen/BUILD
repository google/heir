# Tablegen helpers for dialect and op definitions
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

td_library(
    name = "td_files",
    srcs = [
        "AsmInterfaces.td",
    ],
    # include from the heir-root to enable fully-qualified include-paths
    includes = ["../../../.."],
)

cc_library(
    name = "AsmInterfaces",
    srcs = [
        "AsmInterfaces.cpp",
    ],
    hdrs = [
        "AsmInterfaces.h",
    ],
    deps = [
        ":asm_interfaces_inc_gen",
        "@llvm-project//mlir:IR",
    ],
)

gentbl_cc_library(
    name = "asm_interfaces_inc_gen",
    tbl_outs = [
        (
            ["--gen-type-interface-decls"],
            "AsmInterfaces.h.inc",
        ),
        (
            ["--gen-type-interface-defs"],
            "AsmInterfaces.cpp.inc",
        ),
        (
            ["-gen-type-interface-docs"],
            "AsmInterfaces.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "AsmInterfaces.td",
    deps = [
        "@llvm-project//mlir:BuiltinDialectTdFiles",
    ],
)
