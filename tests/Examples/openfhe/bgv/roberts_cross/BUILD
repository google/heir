# See README.md for setup required to run these tests

load("@heir//tests/Examples/openfhe:test.bzl", "openfhe_end_to_end_test")

package(default_applicable_licenses = ["@heir//:license"])

openfhe_end_to_end_test(
    name = "roberts_cross_64x64_test",
    data = ["@heir//tests/Examples/common"],
    generated_lib_header = "roberts_cross_64x64_lib.h",
    heir_opt_flags = [
        "--annotate-module=backend=openfhe scheme=bgv",
        "--mlir-to-bgv=ciphertext-degree=4096 plaintext-modulus=536903681",
        "--scheme-to-openfhe",
    ],
    mlir_src = "@heir//tests/Examples/common:roberts_cross_64x64.mlir",
    tags = ["notap"],
    test_src = "roberts_cross_test.cpp",
)
