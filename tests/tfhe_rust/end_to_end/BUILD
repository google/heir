# See README.md for setup required to run these tests

load("//bazel:lit.bzl", "glob_lit_tests")

package(default_applicable_licenses = ["@heir//:license"])

glob_lit_tests(
    name = "all_tests",
    data = [
        "Cargo.toml",
        "src/main.rs",
        "src/main_add_one.rs",
        "src/main_fully_connected.rs",
        "@heir//tests:test_utilities",
    ],
    default_tags = [
        "manual",
        "notap",
    ],
    driver = "@heir//tests:run_lit.sh",
    size_override = {
        "test_add.mlir": "large",
        "test_add_one.mlir": "large",
        "test_bitand.mlir": "large",
        "test_simple_lut.mlir": "large",
        "test_fully_connected.mlir": "large",
    },
    test_file_exts = ["mlir"],
)

