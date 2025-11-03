# See README.md for setup required to run these tests

load("@heir//tests/Examples/openfhe:test.bzl", "openfhe_end_to_end_test")

package(default_applicable_licenses = ["@heir//:license"])

openfhe_end_to_end_test(
    name = "convolution_test",
    generated_lib_header = "convolution_testlib.h",
    heir_opt_flags = [
        "--annotate-module=backend=openfhe scheme=ckks",
        "--torch-linalg-to-ckks=ciphertext-degree=1024",
        "--scheme-to-openfhe",
    ],
    heir_translate_flags = [],
    mlir_src = "convolution.mlir",
    tags = ["notap"],
    test_src = "convolution_test.cpp",
)
