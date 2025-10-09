# See README.md for setup required to run these tests

load("@heir//tests/Examples/openfhe:test.bzl", "openfhe_end_to_end_test")

package(default_applicable_licenses = ["@heir//:license"])

openfhe_end_to_end_test(
    name = "ciphertext_plaintext_ops_test",
    generated_lib_header = "ciphertext_plaintext_ops_lib.h",
    mlir_src = "ciphertext_plaintext_ops.mlir",
    test_src = "ciphertext_plaintext_ops_test.cpp",
)
