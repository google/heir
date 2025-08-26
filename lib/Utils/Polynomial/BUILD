"""Polynomial data structure."""

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Polynomial",
    srcs = ["Polynomial.cpp"],
    hdrs = ["Polynomial.h"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Support",
    ],
)

cc_test(
    name = "PolynomialTest",
    srcs = ["PolynomialTest.cpp"],
    deps = [
        ":Polynomial",
        "@googletest//:gtest_main",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "ChebyshevDecomposition",
    srcs = ["ChebyshevDecomposition.cpp"],
    hdrs = ["ChebyshevDecomposition.h"],
)

cc_test(
    name = "ChebyshevDecompositionTest",
    srcs = ["ChebyshevDecompositionTest.cpp"],
    deps = [
        ":ChebyshevDecomposition",
        "@googletest//:gtest_main",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "ChebyshevPatersonStockmeyer",
    hdrs = ["ChebyshevPatersonStockmeyer.h"],
    deps = [
        ":ChebyshevDecomposition",
        "@heir//lib/Kernel:ArithmeticDag",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "ChebyshevPatersonStockmeyerTest",
    srcs = ["ChebyshevPatersonStockmeyerTest.cpp"],
    deps = [
        ":ChebyshevPatersonStockmeyer",
        "@googletest//:gtest_main",
        "@heir//lib/Kernel:ArithmeticDag",
    ],
)
