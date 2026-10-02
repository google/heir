# This build file marks the directory as a Bazel subpackage.

load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

exports_files([
    "cheddar.BUILD",
    "libtommath.BUILD",
    "rmm.BUILD",
])

bzl_library(
    name = "config_bzl",
    srcs = ["config.bzl"],
    visibility = ["//visibility:public"],
)
