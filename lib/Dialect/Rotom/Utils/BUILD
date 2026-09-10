# Rotom utilities (testable helpers shared with transforms).

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "RotomLayout",
    srcs = ["RotomLayout.cpp"],
    hdrs = ["RotomLayout.h"],
    deps = [
        ":RotomTensorExtLayoutLowering",
        "@heir//lib/Dialect/Rotom/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_test(
    name = "RotomLayoutTest",
    srcs = ["RotomLayoutTest.cpp"],
    deps = [
        ":RotomLayout",
        "@googletest//:gtest_main",
        "@heir//lib/Dialect/Rotom/IR:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "LayoutConversion",
    srcs = ["LayoutConversion.cpp"],
    hdrs = ["LayoutConversion.h"],
    deps = [
        ":RotomLayout",
        "@heir//lib/Dialect/Rotom/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_test(
    name = "LayoutConversionTest",
    srcs = ["LayoutConversionTest.cpp"],
    deps = [
        ":LayoutConversion",
        ":RotomLayout",
        "@googletest//:gtest_main",
        "@heir//lib/Dialect/Rotom/IR:Dialect",
        "@heir//lib/Dialect/TensorExt/IR:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "LayoutAlignment",
    srcs = ["LayoutAlignment.cpp"],
    hdrs = ["LayoutAlignment.h"],
    deps = [
        ":RotomLayout",
        "@heir//lib/Dialect/Rotom/IR:Dialect",
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
        ":LayoutConversion",
        ":RotomLayout",
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
