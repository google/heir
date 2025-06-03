# See README.md for setup required to run these tests

load("@heir//tests/Examples/openfhe:test.bzl", "openfhe_end_to_end_test")

package(default_applicable_licenses = ["@heir//:license"])

openfhe_end_to_end_test(
    name = "naive_matmul_test",
    generated_lib_header = "naive_matmul_lib.h",
    # We manually specify the passes since the test file is already expressed at the CKKS dialect level.
    heir_opt_flags = [
        "--annotate-module=backend=openfhe scheme=ckks",
        "--mlir-to-ckks=ciphertext-degree=32",
        "--scheme-to-openfhe",
    ],
    mlir_src = "naive_matmul.mlir",
    tags = ["notap"],
    test_src = "naive_matmul_test.cpp",
)
