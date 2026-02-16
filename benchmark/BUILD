load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_go//go:def.bzl", "go_binary")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

go_binary(
    name = "lattigo_bootstrap_benchmark",
    srcs = ["lattigo_bootstrap_benchmark.go"],
    deps = [
        "@com_github_tuneinsight_lattigo_v6//circuits/ckks/bootstrapping",
        "@com_github_tuneinsight_lattigo_v6//core/rlwe",
        "@com_github_tuneinsight_lattigo_v6//ring",
        "@com_github_tuneinsight_lattigo_v6//schemes/ckks",
        "@com_github_tuneinsight_lattigo_v6//utils",
        "@com_github_tuneinsight_lattigo_v6//utils/sampling",
    ],
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
    name = "openfhe_bootstrap_benchmark",
    srcs = ["openfhe_bootstrap_benchmark.cpp"],
    deps = [
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
