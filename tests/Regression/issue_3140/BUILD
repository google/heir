load("@heir//tests/Examples/openfhe:test.bzl", "openfhe_end_to_end_test")

package(default_applicable_licenses = ["@heir//:license"])

openfhe_end_to_end_test(
    name = "issue_3140_test",
    generated_lib_header = "issue_3140.h",
    heir_opt_flags = [
        "--annotate-module=backend=openfhe scheme=bfv",
        "--mlir-to-bfv",
        "--scheme-to-openfhe",
    ],
    mlir_src = "issue_3140.mlir",
    test_src = "issue_3140_test.cpp",
)
