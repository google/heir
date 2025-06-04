load("//bazel:lit.bzl", "glob_lit_tests")

package(default_applicable_licenses = ["@heir//:license"])

glob_lit_tests(
    name = "all_tests",
    data = ["@heir//tests:test_utilities"],
    driver = "@heir//tests:run_lit.sh",
    # TODO(#1642): cross-level operation does not work well with bootstrapping
    # FIXME: todo to re-enable tests
    exclude = [
        "bootstrap_waterline.mlir",
        "matmul.mlir",
        "naive_matmul.mlir",
    ],
    test_file_exts = ["mlir"],
)
