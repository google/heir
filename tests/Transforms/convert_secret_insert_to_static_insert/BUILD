load("//bazel:lit.bzl", "glob_lit_tests")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

glob_lit_tests(
    name = "all_tests",
    data = ["@heir//tests:test_utilities"],
    driver = "@heir//tests:run_lit.sh",
    test_file_exts = ["mlir"],
)
