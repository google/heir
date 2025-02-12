# General utilities for graphs

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Graph",
    hdrs = ["Graph.h"],
    deps = ["@llvm-project//mlir:Support"],
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
