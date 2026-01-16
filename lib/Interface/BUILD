load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "HoistingInterfaces",
    srcs = ["HoistingInterfaces.cpp"],
    hdrs = ["HoistingInterfaces.h"],
    deps = [
        ":op_interfaces_inc_gen",
        "@heir//lib/Transforms/LayoutOptimization:Hoisting",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

td_library(
    name = "td_files",
    srcs = ["HoistingInterfaces.td"],
    # include from the heir-root to enable fully-qualified include-paths
    includes = ["../.."],
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "op_interfaces_inc_gen",
    tbl_outs = {
        "HoistingOpInterfaces.h.inc": ["-gen-op-interface-decls"],
        "HoistingOpInterfaces.cpp.inc": ["-gen-op-interface-defs"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "HoistingInterfaces.td",
    deps = [
        ":td_files",
    ],
)
