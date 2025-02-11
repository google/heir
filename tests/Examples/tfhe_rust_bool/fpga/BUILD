# See README.md for setup required to run these tests

load("//bazel:lit.bzl", "glob_lit_tests")

package(default_applicable_licenses = ["@heir//:license"])

glob_lit_tests(
    name = "all_tests",
    data = [
        "Cargo.toml",
        "src/main.rs",
        "src/main_fully_connected.rs",
        "src/server_key_enum.rs",
        "tfhe-rs",
        "@heir//tests:test_utilities",
    ],
    default_tags = [
        "manual",
        "notap",
    ],
    driver = "@heir//tests:run_lit.sh",
    test_file_exts = ["mlir"],
)
