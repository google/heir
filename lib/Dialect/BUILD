# Built in HEIR declarations

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "HEIRInterfaces",
    srcs = ["HEIRInterfaces.cpp"],
    hdrs = ["HEIRInterfaces.h"],
    deps = [
        ":interfaces_inc_gen",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "Utils",
    hdrs = ["Utils.h"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "FHEHelpers",
    hdrs = ["FHEHelpers.h"],
    deps = [
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

td_library(
    name = "td_files",
    srcs = ["HEIRInterfaces.td"],
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
