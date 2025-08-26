load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Kernel",
    srcs = ["Kernel.cpp"],
    hdrs = [
        "Kernel.h",
        "KernelName.h",
    ],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "KernelImplementation",
    srcs = ["KernelImplementation.cpp"],
    hdrs = ["KernelImplementation.h"],
    deps = [
        ":Kernel",
        "@heir//lib/Utils:ArithmeticDag",
        "@llvm-project//mlir:IR",
    ],
)

cc_test(
    name = "KernelImplementationTest",
    srcs = ["KernelImplementationTest.cpp"],
    deps = [
        ":Kernel",
        ":KernelImplementation",
        "@googletest//:gtest_main",
        "@heir//lib/Utils:ArithmeticDag",
        "@llvm-project//mlir:IR",
    ],
)
