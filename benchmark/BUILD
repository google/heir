load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_binary(
    name = "openfhe_benchmark",
    srcs = ["openfhe_benchmark.cpp"],
    deps = [
        "@google_benchmark//:benchmark_main",
        "@openfhe//:core",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "math_utils_benchmark",
    srcs = ["math_utils_benchmark.cpp"],
    deps = [
        "@google_benchmark//:benchmark_main",
        "@heir//lib/Utils:MathUtils",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "bootstrap_example",
    srcs = ["bootstrap_example.cpp"],
    hdrs = ["bootstrap_example.h"],
    deps = [
        "@openfhe//:core",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "bootstrap_example_main",
    srcs = ["bootstrap_example_main.cpp"],
    deps = [
        ":bootstrap_example",
    ],
)
