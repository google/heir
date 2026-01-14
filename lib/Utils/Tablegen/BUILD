load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

td_library(
    name = "td_files",
    srcs = [
        "InPlaceOpInterface.td",
    ],
    # include from the heir-root to enable fully-qualified include-paths
    includes = ["../../../.."],
)

cc_library(
    name = "InPlaceOpInterface",
    srcs = [
        "InPlaceOpInterface.cpp",
    ],
    hdrs = [
        "InPlaceOpInterface.h",
    ],
    deps = [
        ":inplace_op_interface_inc_gen",
        "@llvm-project//mlir:IR",
    ],
)

gentbl_cc_library(
    name = "inplace_op_interface_inc_gen",
    tbl_outs = {
        "InPlaceOpInterface.h.inc": ["--gen-op-interface-decls"],
        "InPlaceOpInterface.cpp.inc": ["--gen-op-interface-defs"],
        "InPlaceOpInterface.md": ["-gen-op-interface-docs"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "InPlaceOpInterface.td",
    deps = [
        "@llvm-project//mlir:BuiltinDialectTdFiles",
    ],
)
