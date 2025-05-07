# See README.md for setup required to run these tests

load("@heir//tests/Examples/openfhe:test.bzl", "openfhe_end_to_end_test")

package(default_applicable_licenses = ["@heir//:license"])

openfhe_end_to_end_test(
    name = "dot_product_128_debug_test",
    generated_lib_header = "dot_product_128_debug_lib.h",
    heir_opt_flags = [
        "--mlir-to-bgv=ciphertext-degree=128 annotate-noise-bound=true plaintext-modulus=4194305",
        "--scheme-to-openfhe=insert-debug-handler-calls=true",
    ],
    mlir_src = "dot_product_128.mlir",
    tags = ["notap"],
    test_src = "dot_product_128_debug_test.cpp",
)
