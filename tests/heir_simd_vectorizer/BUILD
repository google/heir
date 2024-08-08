load("//bazel:lit.bzl", "glob_lit_tests")

package(default_applicable_licenses = ["@heir//:license"])

glob_lit_tests(
    name = "all_tests",
    data = ["@heir//tests:test_utilities"],
    driver = "@heir//tests:run_lit.sh",
    size_override = {
        "box_blur_64x64.mlir": "large",
        "roberts_cross_64x64.mlir": "enormous",
        "gx_kernel_64x64.mlir": "large",
    },
    tags_override = {
        "gx_kernel_64x64.mlir": [
            "nofastbuild",
            "notap",
            "manual",
        ],
    },
    test_file_exts = ["mlir"],
)
