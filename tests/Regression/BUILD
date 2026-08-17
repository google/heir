load("//bazel:lit.bzl", "glob_lit_tests")
load("//bazel:yosys.bzl", "requires_yosys")

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
    target_compatible_with_override = {
        "issue_1086.mlir": requires_yosys(),
    },
    test_file_exts = ["mlir"],
)
