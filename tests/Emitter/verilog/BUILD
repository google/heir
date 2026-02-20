# Tests depending on Yosys

load("//bazel:lit.bzl", "glob_lit_tests")

package(default_applicable_licenses = ["@heir//:license"])

glob_lit_tests(
    name = "all_tests",
    data = [
        "@heir//tests:test_utilities",
        "@yosys",
    ],
    default_tags = ["yosys"],
    driver = "@heir//tests:run_lit.sh",
    test_file_exts = [
        "mlir",
        "v",
    ],
)
