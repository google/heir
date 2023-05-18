# Helper files for setting up the bazel workspace

load("//tools/build_defs/testing:bzl_library.bzl", "bzl_library")

bzl_library(
    name = "import_llvm_bzl",
    srcs = ["import_llvm.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "setup_llvm_bzl",
    srcs = ["setup_llvm.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "glob_lit_test_bzl",
    srcs = ["glob_lit_test.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)
