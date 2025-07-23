load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

td_library(
    name = "td_files",
    srcs = [
        "InplaceOpInterface.td",
    ],
    # include from the heir-root to enable fully-qualified include-paths
    includes = ["../../../.."],
)

cc_library(
    name = "InplaceOpInterface",
    srcs = [
        "InplaceOpInterface.cpp",
    ],
    hdrs = [
        "InplaceOpInterface.h",
    ],
    deps = [
        ":inplace_op_interface_inc_gen",
        "@llvm-project//mlir:IR",
    ],
)

gentbl_cc_library(
    name = "inplace_op_interface_inc_gen",
    tbl_outs = {
        "InplaceOpInterface.h.inc": ["--gen-op-interface-decls"],
        "InplaceOpInterface.cpp.inc": ["--gen-op-interface-defs"],
        "InplaceOpInterface.md": ["-gen-op-interface-docs"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "InplaceOpInterface.td",
    deps = [
        "@llvm-project//mlir:BuiltinDialectTdFiles",
    ],
)
