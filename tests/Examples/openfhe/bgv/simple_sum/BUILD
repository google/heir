# See README.md for setup required to run these tests

load("@heir//tests/Examples/openfhe:test.bzl", "openfhe_end_to_end_test")

package(default_applicable_licenses = ["@heir//:license"])

openfhe_end_to_end_test(
    name = "simple_sum_test",
    generated_lib_header = "simple_sum_lib.h",
    heir_opt_flags = [
        "--annotate-module=backend=openfhe scheme=bgv",
        "--mlir-to-bgv=ciphertext-degree=8192",
        "--scheme-to-openfhe",
    ],
    mlir_src = "@heir//tests/Examples/common:simple_sum.mlir",
    tags = ["notap"],
    test_src = "simple_sum_test.cpp",
)
