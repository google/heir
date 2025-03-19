# Built in HEIR declarations

load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
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
        "@llvm-project//mlir:AffineDialect",
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
    name = "FuncUtils",
    srcs = ["FuncUtils.cpp"],
    hdrs = ["FuncUtils.h"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "ModuleAttributes",
    srcs = ["ModuleAttributes.cpp"],
    hdrs = ["ModuleAttributes.h"],
    deps = [
        "@heir//lib/Dialect/BGV/IR:Dialect",
        "@heir//lib/Dialect/CKKS/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
    alwayslink = 1,
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
    tbl_outs = {
        "HEIRInterfaces.h.inc": ["-gen-op-interface-decls"],
        "HEIRInterfaces.cpp.inc": ["-gen-op-interface-defs"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "HEIRInterfaces.td",
    deps = [
        ":td_files",
    ],
)

bzl_library(
    name = "dialect_bzl",
    srcs = ["dialect.bzl"],
)
