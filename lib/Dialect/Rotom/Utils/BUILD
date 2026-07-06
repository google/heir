# Rotom utilities (testable helpers shared with transforms).

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "ContractionAlignment",
    srcs = ["ContractionAlignment.cpp"],
    hdrs = ["ContractionAlignment.h"],
    deps = [
        ":LayoutAlignment",
        "@heir//lib/Dialect/Rotom/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_test(
    name = "ContractionAlignmentTest",
    srcs = ["ContractionAlignmentTest.cpp"],
    deps = [
        ":ContractionAlignment",
        ":LayoutAlignment",
        "@googletest//:gtest_main",
        "@heir//lib/Dialect/Rotom/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "LayoutAlignment",
    srcs = ["LayoutAlignment.cpp"],
    hdrs = ["LayoutAlignment.h"],
    deps = [
        ":RotomTensorExtLayoutLowering",
        "@heir//lib/Dialect/Rotom/IR:Dialect",
        "@heir//lib/Dialect/TensorExt/IR:TensorExtAttributes",
        "@heir//lib/Transforms/LayoutOptimization:LayoutConversionCost",
        "@heir//lib/Utils/Layout:Utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_test(
    name = "LayoutAlignmentTest",
    srcs = ["LayoutAlignmentTest.cpp"],
    deps = [
        ":LayoutAlignment",
        ":RotomTensorExtLayoutLowering",
        "@googletest//:gtest_main",
        "@heir//lib/Dialect/Rotom/IR:Dialect",
        "@heir//lib/Dialect/TensorExt/IR:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
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
        "@heir//lib/Utils/Layout:Utils",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)
