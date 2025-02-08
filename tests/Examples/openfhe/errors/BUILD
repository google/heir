load("//bazel:lit.bzl", "glob_lit_tests")

package(default_applicable_licenses = ["@heir//:license"])

glob_lit_tests(
    name = "all_tests",
    data = ["@heir//tests:test_utilities"],
    driver = "@heir//tests:run_lit.sh",
    # TODO(#1397): Move type detection earlier
    exclude = ["dot_product_8f_type_error.mlir"],
    test_file_exts = ["mlir"],
)
