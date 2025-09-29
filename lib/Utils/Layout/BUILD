# Utilities for layout manipulation
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Codegen",
    srcs = ["Codegen.cpp"],
    hdrs = ["Codegen.h"],
    deps = [
        ":IslConversion",
        "@isl",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
    ],
)

cc_test(
    name = "CodegenTest",
    srcs = ["CodegenTest.cpp"],
    deps = [
        ":Codegen",
        "@googletest//:gtest_main",
        "@heir//lib/Utils/Layout:IslConversion",
        "@heir//lib/Utils/Layout:Utils",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
    ],
)

cc_library(
    name = "Utils",
    srcs = ["Utils.cpp"],
    hdrs = ["Utils.h"],
    deps = [
        ":IslConversion",
        "@heir//lib/Utils:MathUtils",
        "@isl",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithUtils",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "Evaluate",
    srcs = ["Evaluate.cpp"],
    hdrs = ["Evaluate.h"],
    deps = [
        ":IslConversion",
        ":Utils",
        "@isl",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithUtils",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_test(
    name = "EvaluateTest",
    srcs = ["EvaluateTest.cpp"],
    deps = [
        ":Evaluate",
        ":IslConversion",
        ":Utils",
        "@googletest//:gtest_main",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
    ],
)

cc_test(
    name = "UtilsTest",
    srcs = ["UtilsTest.cpp"],
    deps = [
        ":Evaluate",
        ":IslConversion",
        ":Utils",
        "@googletest//:gtest_main",
        "@heir//lib/Utils:TensorUtils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "Hoisting",
    srcs = ["Hoisting.cpp"],
    hdrs = ["Hoisting.h"],
    deps = [
        ":IslConversion",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
    ],
)

cc_test(
    name = "HoistingTest",
    srcs = ["HoistingTest.cpp"],
    deps = [
        ":Codegen",
        ":Hoisting",
        ":IslConversion",
        "@googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "IslConversion",
    srcs = ["IslConversion.cpp"],
    hdrs = ["IslConversion.h"],
    deps = [
        "@isl",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:Support",
    ],
)

cc_test(
    name = "IslConversionTest",
    srcs = ["IslConversionTest.cpp"],
    deps = [
        ":IslConversion",
        "@googletest//:gtest_main",
        "@isl",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineAnalysis",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:AsmParser",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_binary(
    name = "debug_packing_1d",
    srcs = ["debug_packing_1d.c"],
)

cc_binary(
    name = "debug_packing_2d",
    srcs = ["debug_packing_2d.cpp"],
)
