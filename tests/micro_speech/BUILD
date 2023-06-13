load("//bazel:lit.bzl", "glob_lit_tests")

package(
    default_applicable_licenses = ["@heir//:license"],
)

glob_lit_tests(
    data = ["@heir//tests:test_utilities"],
    driver = "@heir//tests:run_lit.sh",
    size_override = {
        "before_unroll_and_forward.mlir": "large",
    },
    tags_override = {
        "before_unroll_and_forward.mlir": ["nofastbuild"],
    },
    test_file_exts = ["mlir"],
)
