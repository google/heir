# See README.md for setup required to run these tests

load("@heir//tests/Examples/openfhe:test.bzl", "openfhe_end_to_end_test")

package(default_applicable_licenses = ["@heir//:license"])

openfhe_end_to_end_test(
    name = "conv_2d_test",
    generated_lib_header = "conv_2d_lib.h",
    heir_opt_flags = [
        "--annotate-module=backend=openfhe scheme=ckks",
        "--mlir-to-ckks=ciphertext-degree=32",
        "--scheme-to-openfhe",
    ],
    heir_translate_flags = [],
    mlir_src = "@heir//tests/Examples/openfhe/ckks/conv_2d:conv_2d.mlir",
    tags = ["notap"],
    test_src = "conv_2d_test.cpp",
)
