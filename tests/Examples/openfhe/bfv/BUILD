load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "debug_helper",
    srcs = ["debug_helper.cpp"],
    hdrs = ["debug_helper.h"],
    deps = [
        "@openfhe//:core",
        "@openfhe//:pke",
    ],
)
