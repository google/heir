# Tests depending on Yosys

load("//bazel:lit.bzl", "glob_lit_tests")
load("//bazel:yosys.bzl", "requires_yosys")

package(default_applicable_licenses = ["@heir//:license"])

glob_lit_tests(
    name = "all_tests",
    data = [
        "@heir//tests:test_utilities",
        "@yosys",
    ],
    default_tags = ["yosys"],
    driver = "@heir//tests:run_lit.sh",
    target_compatible_with = requires_yosys(),
    test_file_exts = [
        "mlir",
        "v",
    ],
)
