load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load(":test.bzl", "cheddar_end_to_end_test")

package(default_applicable_licenses = ["@heir//:license"])

bzl_library(
    name = "test_bzl",
    srcs = ["test.bzl"],
    visibility = ["//visibility:private"],
    deps = [
        "@bazel_skylib//rules:build_test",
        "@bazel_skylib//rules:write_file",
        "@heir//bazel/cheddar:config_bzl",
        "@heir//tools:heir_opt_bzl",
        "@heir//tools:heir_translate_bzl",
        "@rules_cc//cc:core_rules",
    ],
)

# TODO: these inputs are hand-written Cheddar IR; generate them from
# tests/Examples/common via a scheme-to-cheddar pipeline once one exists.

# The op surface real kernels use.
cheddar_end_to_end_test(
    name = "kernels",
    mlir_src = "kernels.mlir",
)

# Cleartext message bridges (encode/decode) and key lookups on the EvkMap.
cheddar_end_to_end_test(
    name = "encode_decode",
    mlir_src = "encode_decode.mlir",
)
