# Utilities for layout manipulation
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
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "Parser",
    srcs = ["Parser.cpp"],
    hdrs = ["Parser.h"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineAnalysis",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:AsmParser",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_test(
    name = "CodegenTest",
    srcs = ["CodegenTest.cpp"],
    deps = [
        ":Codegen",
        ":Parser",
        "@googletest//:gtest_main",
        "@isl",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "Utils",
    srcs = ["Utils.cpp"],
    hdrs = ["Utils.h"],
    deps = [
        "@heir//lib/Utils:MathUtils",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_test(
    name = "UtilsTest",
    srcs = ["UtilsTest.cpp"],
    deps = [
        ":Parser",
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
    name = "IslConversion",
    srcs = ["IslConversion.cpp"],
    hdrs = ["IslConversion.h"],
    deps = [
        ":Utils",
        "@isl",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_test(
    name = "IslConversionTest",
    srcs = ["IslConversionTest.cpp"],
    deps = [
        ":IslConversion",
        ":Parser",
        ":Utils",
        "@googletest//:gtest_main",
        "@isl",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
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
    srcs = ["debug_packing_2d.c"],
)
