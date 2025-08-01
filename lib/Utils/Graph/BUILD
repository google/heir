# General utilities for graphs

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Graph",
    hdrs = ["Graph.h"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Support",
    ],
)

cc_test(
    name = "GraphTest",
    srcs = ["GraphTest.cpp"],
    deps = [
        ":Graph",
        "@googletest//:gtest_main",
        "@llvm-project//mlir:Support",
    ],
)
