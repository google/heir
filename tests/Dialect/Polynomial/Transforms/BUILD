load("//bazel:lit.bzl", "glob_lit_tests")

package(default_applicable_licenses = ["@heir//:license"])

glob_lit_tests(
    name = "all_tests",
    data = ["@heir//tests:test_utilities"],
    driver = "@heir//tests:run_lit.sh",
    exclude = ["ntt_rewrites.mlir"],  # TODO(#1095): disabled for mod_arith type migration
    test_file_exts = ["mlir"],
)
