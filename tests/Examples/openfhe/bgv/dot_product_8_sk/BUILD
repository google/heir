# See README.md for setup required to run these tests

load("@heir//tests/Examples/openfhe:test.bzl", "openfhe_end_to_end_test")

package(default_applicable_licenses = ["@heir//:license"])

openfhe_end_to_end_test(
    name = "dot_product_8_sk_test",
    generated_lib_header = "dot_product_8_sk_lib.h",
    heir_opt_flags = [
        "--annotate-module=backend=openfhe scheme=bgv",
        "--mlir-to-bgv=ciphertext-degree=8192 use-public-key=false",
        "--scheme-to-openfhe",
    ],
    mlir_src = "@heir//tests/Examples/common:dot_product_8.mlir",
    tags = ["notap"],
    test_src = "dot_product_8_sk_test.cpp",
)
