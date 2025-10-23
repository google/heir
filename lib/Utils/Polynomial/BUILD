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
        "@llvm-project//mlir:IR",
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
        "@heir//lib/Kernel:AbstractValue",
        "@heir//lib/Kernel:ArithmeticDag",
        "@heir//lib/Kernel:KernelImplementation",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "ChebyshevPatersonStockmeyerTest",
    srcs = ["ChebyshevPatersonStockmeyerTest.cpp"],
    deps = [
        ":ChebyshevPatersonStockmeyer",
        ":PolynomialTestVisitors",
        "@googletest//:gtest_main",
        "@heir//lib/Kernel:ArithmeticDag",
    ],
)

cc_library(
    name = "Horner",
    hdrs = ["Horner.h"],
    deps = [
        "@heir//lib/Kernel:ArithmeticDag",
    ],
)

cc_library(
    name = "PatersonStockmeyer",
    hdrs = ["PatersonStockmeyer.h"],
    deps = [
        ":ChebyshevPatersonStockmeyer",
        "@heir//lib/Kernel:ArithmeticDag",
    ],
)

cc_library(
    name = "PolynomialTestVisitors",
    hdrs = ["PolynomialTestVisitors.h"],
    deps = [
        "@heir//lib/Kernel:AbstractValue",
        "@heir//lib/Kernel:ArithmeticDag",
    ],
)

cc_test(
    name = "HornerTest",
    srcs = ["HornerTest.cpp"],
    deps = [
        ":Horner",
        ":PolynomialTestVisitors",
        "@googletest//:gtest_main",
        "@heir//lib/Kernel:ArithmeticDag",
    ],
)

cc_test(
    name = "PatersonStockmeyerTest",
    srcs = ["PatersonStockmeyerTest.cpp"],
    deps = [
        ":PatersonStockmeyer",
        ":PolynomialTestVisitors",
        "@googletest//:gtest_main",
        "@heir//lib/Kernel:AbstractValue",
        "@heir//lib/Kernel:ArithmeticDag",
        "@heir//lib/Kernel:KernelImplementation",
    ],
)
