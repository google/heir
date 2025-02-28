load("//bazel:lit.bzl", "glob_lit_tests")

package(default_applicable_licenses = ["@heir//:license"])

glob_lit_tests(
    name = "all_tests",
    data = [
        ":dot_product_8_test",
        "@heir//tests:test_utilities",
        "@llvm-project//clang",
        "@llvm-project//llvm:llc",
        "@llvm-project//mlir:mlir-translate",
    ],
    default_tags = [
        "notap",
    ],
    driver = "@heir//tests:run_lit.sh",
    test_file_exts = ["mlir"],
)

cc_library(
    name = "dot_product_8_test",
    srcs = ["dot_product_8_test.c"],
)
