load("//bazel:lit.bzl", "glob_lit_tests")

package(default_applicable_licenses = ["@heir//:license"])

glob_lit_tests(
    name = "all_tests",
    data = ["@heir//tests:test_utilities"],
    driver = "@heir//tests:run_lit.sh",
    size_override = {
        "issue_2466.mlir": "large",
    },
    tags_override = {
        "issue_2466.mlir": ["nofastbuild"],
    },
    test_file_exts = ["mlir"],
)
