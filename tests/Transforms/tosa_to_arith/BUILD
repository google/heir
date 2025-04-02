load("//bazel:lit.bzl", "glob_lit_tests")

package(default_applicable_licenses = ["@heir//:license"])

glob_lit_tests(
    name = "all_tests",
    data = ["@heir//tests:test_utilities"],
    driver = "@heir//tests:run_lit.sh",
    tags_override = {
        "mnist_2fc.mlir": [
            "nofastbuild",
            "notap",
            "manual",
        ],
    },
    test_file_exts = ["mlir"],
)
