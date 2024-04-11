load("//bazel:lit.bzl", "glob_lit_tests")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

glob_lit_tests(
    name = "all_tests",
    data = ["@heir//tests:test_utilities"],
    default_tags = ["yosys"],
    driver = "@heir//tests:run_lit.sh",
    size_override = {
        "fully_connected.mlir": "large",
    },
    tags_override = {
        "hello_world.mlir": [
            "nofastbuild",
            "notap",
            "manual",
        ],
    },
    test_file_exts = ["mlir"],
)
