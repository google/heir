load("//bazel:lit.bzl", "glob_lit_tests")

package(default_applicable_licenses = ["@heir//:license"])

glob_lit_tests(
    name = "all_tests",
    data = [
        ":roberts_cross_mod_test",
        ":roberts_cross_test",
        "@heir//tests:test_utilities",
        "@heir//tests/Examples/plaintext:memrefCopy",
        "@llvm-project//llvm:llc",
        "@llvm-project//mlir:mlir-translate",
    ],
    driver = "@heir//tests:run_lit.sh",
    size_override = {
        "roberts_cross_64x64.mlir": "enormous",
    },
    tags_override = {
        "roberts_cross_64x64.mlir": [
            "manual",
            "nofastbuild",
            "notap",
        ],
    },
    test_file_exts = ["mlir"],
)

cc_library(
    name = "roberts_cross_test",
    srcs = [
        "roberts_cross_test.c",
    ],
)

cc_library(
    name = "roberts_cross_mod_test",
    srcs = [
        "roberts_cross_mod_test.c",
    ],
)
