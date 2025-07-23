load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_applicable_licenses = ["@heir//:license"])

cc_library(
    name = "memrefCopy",
    srcs = [
        "memrefCopy.cpp",
    ],
    visibility = ["@heir//tests/Examples/plaintext:__subpackages__"],
)

bzl_library(
    name = "test_bzl",
    srcs = ["test.bzl"],
    visibility = ["//visibility:private"],
    deps = [
        "@heir//tools:heir_opt_bzl",
        "@heir//tools:llc_bzl",
        "@heir//tools:mlir_translate_bzl",
    ],
)
