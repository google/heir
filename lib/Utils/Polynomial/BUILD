"""Polynomial data structure."""

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
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
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "ChebyshevPatersonStockmeyerTestUtils",
    srcs = ["ChebyshevPatersonStockmeyerTestUtils.cpp"],
    hdrs = ["ChebyshevPatersonStockmeyerTestUtils.h"],
)

cc_test(
    name = "ChebyshevPatersonStockmeyerTest",
    srcs = ["ChebyshevPatersonStockmeyerTest.cpp"],
    deps = [
        ":ChebyshevPatersonStockmeyer",
        ":ChebyshevPatersonStockmeyerTestUtils",
        ":PolynomialTestVisitors",
        "@googletest//:gtest_main",
        "@heir//lib/Kernel:AbstractValue",
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
        "@heir//lib/Kernel:AbstractValue",
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
    ],
)

cc_test(
    name = "ChebyshevPatersonStockmeyerFuzzTest",
    srcs = ["ChebyshevPatersonStockmeyerFuzzTest.cpp"],
    deps = [
        ":ChebyshevPatersonStockmeyer",
        ":ChebyshevPatersonStockmeyerTestUtils",
        ":PolynomialTestVisitors",
        "@fuzztest//fuzztest",
        "@googletest//:gtest_main",
        "@heir//lib/Kernel:AbstractValue",
    ],
)

cc_library(
    name = "NTT",
    srcs = ["NTT.cpp"],
    hdrs = ["NTT.h"],
    deps = [
        "@com_google_absl//absl/numeric:int128",
        "@heir//lib/Utils:APIntUtils",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "NTTTest",
    srcs = ["NTTTest.cpp"],
    deps = [
        ":NTT",
        "@googletest//:gtest_main",
        "@heir//lib/Utils:MathUtils",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "OpenFHEEquivalenceTest",
    srcs = ["OpenFHEEquivalenceTest.cpp"],
    deps = [
        ":NTT",
        ":Polynomial",
        "@googletest//:gtest_main",
        "@heir//lib/Utils:MathUtils",
        "@llvm-project//llvm:Support",
        "@openfhe//:core",
        "@openfhe//:pke",
    ],
)

cc_test(
    name = "OpenFHEEquivalenceFuzzTest",
    srcs = ["OpenFHEEquivalenceFuzzTest.cpp"],
    deps = [
        ":NTT",
        "@fuzztest//fuzztest",
        "@googletest//:gtest_main",
        "@heir//lib/Utils:MathUtils",
        "@llvm-project//llvm:Support",
        "@openfhe//:core",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "ntt_benchmark",
    srcs = ["ntt_benchmark.cpp"],
    deps = [
        ":NTT",
        "@google_benchmark//:benchmark_main",
        "@heir//lib/Utils:MathUtils",
        "@llvm-project//llvm:Support",
        "@openfhe//:core",
        "@openfhe//:pke",
    ],
)
