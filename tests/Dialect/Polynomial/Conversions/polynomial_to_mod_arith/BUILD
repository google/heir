load("//bazel:lit.bzl", "glob_lit_tests")

package(default_applicable_licenses = ["@heir//:license"])

glob_lit_tests(
    name = "all_tests",
    data = ["@heir//tests:test_utilities"],
    driver = "@heir//tests:run_lit.sh",
    # TODO(#2157): re-enable when the rest of key_switch_inner is lowered
    exclude = ["lower_keyswitch_inner.mlir"],
    test_file_exts = ["mlir"],
)
