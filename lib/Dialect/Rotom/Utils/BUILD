# Rotom utilities (testable helpers shared with transforms).

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "RotomTensorExtLayoutLowering",
    srcs = ["RotomTensorExtLayoutLowering.cpp"],
    hdrs = ["RotomTensorExtLayoutLowering.h"],
    deps = [
        "@heir//lib/Dialect/Rotom/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_test(
    name = "RotomTensorExtLayoutLoweringTest",
    srcs = ["RotomTensorExtLayoutLoweringTest.cpp"],
    deps = [
        ":RotomTensorExtLayoutLowering",
        "@googletest//:gtest_main",
        "@heir//lib/Dialect/Rotom/IR:Dialect",
        "@heir//lib/Utils/Layout:Evaluate",
        "@heir//lib/Utils/Layout:IslConversion",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)
