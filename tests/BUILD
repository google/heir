# Tests for HEIR

load("//bazel:glob_lit_test.bzl", "glob_lit_tests")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "@heir//tools:heir-opt",
        "@heir//tools:heir-translate",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//llvm:not",
        "@llvm-project//mlir:libmlir_c_runner_utils.so",
        "@llvm-project//mlir:libmlir_runner_utils.so",
        "@llvm-project//mlir:mlir-cpu-runner",
        "@llvm-project//mlir:mlir-opt",
        "@llvm-project//mlir:run_lit.sh",
    ],
)

glob_lit_tests(
    data = [":test_utilities"],
    # Note: the driver argument is silently ignored for the OSS version of this
    # macro
    driver = "@heir//tests:run_lit.sh",
    exclude = [],
    test_file_exts = ["mlir"],
)
