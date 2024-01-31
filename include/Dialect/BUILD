# Built-in common HEIR definitions

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

exports_files(
    [
        "HEIRInterfaces.h",
    ],
)

td_library(
    name = "td_files",
    srcs = [
        "HEIRInterfaces.td",
    ],
    # include from the heir-root to enable fully-qualified include-paths
    includes = ["../.."],
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "interfaces_inc_gen",
    tbl_outs = [
        (
            ["-gen-op-interface-decls"],
            "HEIRInterfaces.h.inc",
        ),
        (
            ["-gen-op-interface-defs"],
            "HEIRInterfaces.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "HEIRInterfaces.td",
    deps = [
        ":td_files",
    ],
)
