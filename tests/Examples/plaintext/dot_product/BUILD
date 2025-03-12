load("//bazel:lit.bzl", "glob_lit_tests")

package(default_applicable_licenses = ["@heir//:license"])

glob_lit_tests(
    name = "all_tests",
    data = [
        ":dot_product_8_mod_test",
        ":dot_product_8_test",
        "@heir//tests:test_utilities",
        "@llvm-project//llvm:llc",
        "@llvm-project//mlir:mlir-translate",
    ],
    driver = "@heir//tests:run_lit.sh",
    tags_override = {
        "dot_product_8.mlir": [
            "notap",
        ],
    },
    test_file_exts = ["mlir"],
)

cc_library(
    name = "dot_product_8_test",
    srcs = ["dot_product_8_test.c"],
)

cc_library(
    name = "dot_product_8_mod_test",
    srcs = ["dot_product_8_mod_test.c"],
)
