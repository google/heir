load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

package(default_applicable_licenses = ["@heir//:license"])

bzl_library(
    name = "test_bzl",
    srcs = ["test.bzl"],
    visibility = ["//visibility:private"],
    deps = [
        "//tools/build_defs/go:go_library_bzl",
        "@heir//tools:heir_opt_bzl",
        "@heir//tools:heir_translate_bzl",
    ],
)
