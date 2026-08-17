load("//bazel:lit.bzl", "glob_lit_tests")
load("//bazel:yosys.bzl", "requires_yosys")

package(default_applicable_licenses = ["@heir//:license"])

glob_lit_tests(
    name = "all_tests",
    data = ["@heir//tests:test_utilities"],
    default_tags = ["yosys"],
    driver = "@heir//tests:run_lit.sh",
    size_override = {
        "fully_connected.mlir": "large",
        "hello_world_small.mlir": "large",
        "variance.mlir": "large",
    },
    tags_override = {
        "fully_connected.mlir": [
            "nofastbuild",
            "notap",
            "manual",
        ],
        "hello_world.mlir": [
            "nofastbuild",
            "notap",
            "manual",
        ],
        "hello_world_clean.mlir": [
            "nofastbuild",
            "notap",
            "manual",
        ],
        "hello_world_small.mlir": [
            "nofastbuild",
            "notap",
            "manual",
        ],
        "hello_world_clean_small.mlir": [
            "nofastbuild",
            "notap",
            "manual",
        ],
        "hello_world_clean_xsmall.mlir": [
            "nofastbuild",
            "notap",
            "manual",
        ],
    },
    target_compatible_with = requires_yosys(),
    test_file_exts = ["mlir"],
)

exports_files([
    "hello_world_small.mlir",
])
