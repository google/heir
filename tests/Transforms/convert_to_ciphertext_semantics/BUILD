load("//bazel:lit.bzl", "glob_lit_tests")

package(default_applicable_licenses = ["@heir//:license"])

glob_lit_tests(
    name = "all_tests",
    data = ["@heir//tests:test_utilities"],
    driver = "@heir//tests:run_lit.sh",
    exclude = [
        # TODO(#2047): Remove these tests after migrating to new layout
        "linalg_reduce.mlir",
        "matvec.mlir",
        "linalg_matvec_1024.mlir",
        "linalg_matvec_ckks.mlir",
    ],
    test_file_exts = ["mlir"],
)
