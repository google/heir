load("//bazel:lit.bzl", "glob_lit_tests")

package(default_applicable_licenses = ["@heir//:license"])

glob_lit_tests(
    name = "all_tests",
    data = [
        ":dot_product_8f_debug_test",
        "@heir//tests:test_utilities",
        "@heir//tests/Examples/plaintext:memrefCopy",
        "@llvm-project//llvm:llc",
        "@llvm-project//mlir:mlir-translate",
    ],
    driver = "@heir//tests:run_lit.sh",
    tags_override = {
        "dot_product_8f.mlir": [
            "notap",
            # It is a failing test on purpose.
            # When failure the debug output could be dumped.
            "manual",
        ],
    },
    test_file_exts = ["mlir"],
)

cc_library(
    name = "dot_product_8f_debug_test",
    srcs = ["dot_product_8f_debug_test.c"],
)
