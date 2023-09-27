# Tests for HEIR

load("//bazel:lit.bzl", "glob_lit_tests")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "@heir//tests:lit.cfg.py",
        "@heir//tools:heir-opt",
        "@heir//tools:heir-translate",
        "@heir_pip_deps_lit//:pkg",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//llvm:count",
        "@llvm-project//llvm:not",
        "@llvm-project//mlir:libmlir_c_runner_utils.so",
        "@llvm-project//mlir:libmlir_runner_utils.so",
        "@llvm-project//mlir:mlir-cpu-runner",
        "@llvm-project//mlir:mlir-opt",
        # copybara: run_lit.sh
    ],
)

glob_lit_tests(
    name = "all_tests",
    data = [":test_utilities"],
    # Note: the driver argument is silently ignored for the OSS version of this
    # macro, replacing it with a standard invocation of lit whose config is in
    # lit.cfg.py
    driver = "@heir//tests:run_lit.sh",
    test_file_exts = ["mlir"],
)

exports_files(
    [
        "run_lit.sh",
    ],
)
