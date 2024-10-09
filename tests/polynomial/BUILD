load("//bazel:lit.bzl", "glob_lit_tests")

package(default_applicable_licenses = ["@heir//:license"])

glob_lit_tests(
    name = "all_tests",
    data = ["@heir//tests:test_utilities"],
    driver = "@heir//tests:run_lit.sh",
    exclude = [
        "lower_add.mlir",  # b/371961315
        "lower_mul.mlir",  # b/371961315
        "lower_mul_runner.mlir",  # b/371961315
    ],
    test_file_exts = ["mlir"],
)
