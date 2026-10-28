# Cheddar parameter loading module

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "ParameterFile",
    srcs = ["ParameterFile.cpp"],
    hdrs = ["ParameterFile.h"],
    deps = [
        "@heir//lib/Dialect/CKKS/IR:Dialect",
        "@heir//lib/Dialect/CKKS/IR:Enums",
        "@heir//lib/Dialect/Cheddar/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_test(
    name = "ParameterFileTest",
    srcs = ["ParameterFileTest.cpp"],
    data = ["@cheddar//parameters:param_files"],
    deps = [
        ":ParameterFile",
        "@bazel_tools//tools/cpp/runfiles",
        "@googletest//:gtest_main",
        "@heir//lib/Dialect/CKKS/IR:Dialect",
        "@heir//lib/Dialect/CKKS/IR:Enums",
        "@heir//lib/Dialect/Cheddar/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)
