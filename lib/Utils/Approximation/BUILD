"""Approximation utilities"""

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Chebyshev",
    srcs = ["Chebyshev.cpp"],
    hdrs = ["Chebyshev.h"],
    deps = [
        "@heir//lib/Utils/Polynomial",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Support",
        "@pocketfft",
    ],
)

cc_test(
    name = "ChebyshevTest",
    srcs = ["ChebyshevTest.cpp"],
    deps = [
        ":Chebyshev",
        "@googletest//:gtest_main",
        "@heir//lib/Utils/Polynomial",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "CaratheodoryFejer",
    srcs = ["CaratheodoryFejer.cpp"],
    hdrs = ["CaratheodoryFejer.h"],
    deps = [
        ":Chebyshev",
        "@eigen",
        "@heir//lib/Utils/Polynomial",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "CaratheodoryFejerTest",
    srcs = ["CaratheodoryFejerTest.cpp"],
    deps = [
        ":CaratheodoryFejer",
        "@googletest//:gtest_main",
        "@heir//lib/Utils/Polynomial",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Support",
    ],
)
