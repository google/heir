"""Polynomial data structure."""

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
    deps = ["@llvm-project//llvm:Support"],
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
