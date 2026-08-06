load("@rules_cc//cc:defs.bzl", "cc_test")
load("@rules_go//go:def.bzl", "go_test")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_test(
    name = "openfhe_test",
    srcs = ["openfhe_test.cpp"],
    deps = [
        "@googletest//:gtest",
        "@googletest//:gtest_main",
        "@openfhe//:core",
        "@openfhe//:pke",
    ],
)

go_test(
    name = "chebyshev_level_consumption_test",
    srcs = ["lattigo_test.go"],
    deps = [
        "@com_github_tuneinsight_lattigo_v6//circuits/ckks/polynomial",
        "@com_github_tuneinsight_lattigo_v6//core/rlwe",
        "@com_github_tuneinsight_lattigo_v6//ring",
        "@com_github_tuneinsight_lattigo_v6//schemes/ckks",
        "@com_github_tuneinsight_lattigo_v6//utils/bignum",
    ],
)
