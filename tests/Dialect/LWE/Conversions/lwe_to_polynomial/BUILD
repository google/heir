load("//bazel:lit.bzl", "glob_lit_tests")

package(default_applicable_licenses = ["@heir//:license"])

glob_lit_tests(
    name = "all_tests",
    data = ["@heir//tests:test_utilities"],
    driver = "@heir//tests:run_lit.sh",
    # TODO(#1199): support RNS lowering
    exclude = [
        "test_rlwe_sk_encrypt.mlir",
        "test_rlwe_pk_encrypt.mlir",
        "ops_dimension_error.mlir",
        "decrypt_ops.mlir",
    ],
    test_file_exts = ["mlir"],
)
