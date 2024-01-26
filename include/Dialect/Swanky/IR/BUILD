# Swanky tablegen and headers

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

exports_files(
    [
        "SwankyDialect.h",
    ],
)

td_library(
    name = "td_files",
    srcs = [
        "SwankyDialect.td",
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
            "SwankyDialect.h.inc",
        ),
        (
            [
                "-gen-dialect-defs",
            ],
            "SwankyDialect.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "SwankyDialect.td",
    deps = [
        ":td_files",
    ],
)
