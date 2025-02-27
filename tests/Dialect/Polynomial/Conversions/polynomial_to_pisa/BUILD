load("//bazel:lit.bzl", "glob_lit_tests")

package(default_applicable_licenses = ["@heir//:license"])

glob_lit_tests(
    name = "all_tests",
    data = ["@heir//tests:test_utilities"],
    driver = "@heir//tests:run_lit.sh",
    exclude = ["end_to_end.mlir"],  # TODO (#1199): re-enable after `--lwe-to-polynomial` is fixed
    test_file_exts = ["mlir"],
)
