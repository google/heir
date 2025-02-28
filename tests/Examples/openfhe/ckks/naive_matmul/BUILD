# See README.md for setup required to run these tests

load("@heir//tests/Examples/openfhe:test.bzl", "openfhe_end_to_end_test")

package(default_applicable_licenses = ["@heir//:license"])

openfhe_end_to_end_test(
    name = "naive_matmul_test",
    generated_lib_header = "naive_matmul_lib.h",
    # We manually specify the passes since the test file is already expressed at the CKKS dialect level.
    heir_opt_flags = [
        "--full-loop-unroll",
        "--ckks-to-lwe",
        "--lwe-to-openfhe",
        "--canonicalize",
        "--cse",
        "--openfhe-configure-crypto-context",
    ],
    heir_translate_flags = [
        "--openfhe-include-type=source-relative",
    ],
    mlir_src = "naive_matmul.mlir",
    tags = ["notap"],
    test_src = "naive_matmul_test.cpp",
)
