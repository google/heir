# Helper files for setting up the bazel workspace

load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

bzl_library(
    name = "import_llvm_bzl",
    srcs = ["import_llvm.bzl"],
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "setup_llvm_bzl",
    srcs = ["setup_llvm.bzl"],
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "lit_bzl",
    srcs = ["lit.bzl"],
    visibility = ["//visibility:private"],
)
