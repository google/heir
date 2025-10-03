# See README.md for setup required to run these tests

load("@heir//tests/Examples/openfhe:test.bzl", "openfhe_end_to_end_test")

package(default_applicable_licenses = ["@heir//:license"])

openfhe_end_to_end_test(
    name = "custom_arithmetization_test",
    generated_lib_header = "custom_arithmetization_lib.h",
    heir_opt_flags = [
        "--annotate-module=backend=openfhe scheme=ckks",
        "--mlir-to-ckks=ciphertext-degree=16 enable-arithmetization=false",
        "--scheme-to-openfhe",
    ],
    heir_translate_flags = [],
    mlir_src = "@heir//tests/Examples/openfhe/ckks/custom_arithmetization:custom_arithmetization.mlir",
    test_src = "custom_arithmetization_test.cpp",
)
