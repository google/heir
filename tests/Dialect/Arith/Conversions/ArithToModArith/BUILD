load("//bazel:lit.bzl", "glob_lit_tests")

package(default_applicable_licenses = ["@heir//:license"])

glob_lit_tests(
    name = "all_tests",
    data = ["@heir//tests:test_utilities"],
    driver = "@heir//tests:run_lit.sh",
    exclude = [
        "arith-to-mod-arith.mlir",  # TODO(b/386761235): re-enable it
    ],
    test_file_exts = ["mlir"],
)
