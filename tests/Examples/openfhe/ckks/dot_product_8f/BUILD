# See README.md for setup required to run these tests

load("@heir//tests/Examples/openfhe:test.bzl", "openfhe_end_to_end_test")

package(default_applicable_licenses = ["@heir//:license"])

openfhe_end_to_end_test(
    name = "dot_product_8f_test",
    generated_lib_header = "dot_product_8f_lib.h",
    heir_opt_flags = [
        "--mlir-to-ckks=ciphertext-degree=8",
        "--scheme-to-openfhe",
    ],
    heir_translate_flags = [],
    mlir_src = "dot_product_8f.mlir",
    tags = ["notap"],
    test_src = "dot_product_8f_test.cpp",
)
