load("//bazel:lit.bzl", "glob_lit_tests")

package(default_applicable_licenses = ["@heir//:license"])

glob_lit_tests(
    name = "all_tests",
    data = ["@heir//tests:test_utilities"],
    driver = "@heir//tests:run_lit.sh",
    # TODO(#1199): support RNS lowering
    exclude = [
        "lower_add_error.mlir",
    ],
    test_file_exts = ["mlir"],
)
