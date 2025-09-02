load("@heir//bazel:lit.bzl", "glob_lit_tests")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_applicable_licenses = ["@heir//:license"])

glob_lit_tests(
    name = "all_tests",
    data = [
        ":quarter_to_llvm",
        "@heir//tests:test_utilities",
        "@llvm-project//clang",
        "@llvm-project//llvm:llc",
        "@llvm-project//mlir:mlir-translate",
    ],
    default_tags = [
        "notap",
        "manual",
    ],
    driver = "@heir//tests:run_lit.sh",
    test_file_exts = ["mlir"],
)

cc_library(
    name = "quarter_to_llvm",
    srcs = ["quarter_to_llvm.c"],
)
