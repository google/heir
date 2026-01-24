load("@heir//tests/Examples/openfhe:test.bzl", "openfhe_end_to_end_test")

package(default_applicable_licenses = ["@heir//:license"])

openfhe_end_to_end_test(
    name = "loop_test",
    generated_lib_header = "loop_lib.h",
    heir_opt_flags = [
        "--annotate-module=backend=openfhe scheme=ckks",
        "--mlir-to-ckks=ciphertext-degree=8192 level-budget=3 experimental-disable-loop-unroll=true",
        "--scheme-to-openfhe",
    ],
    mlir_src = "@heir//tests/Examples/common:loop.mlir",
    test_src = "loop_test.cpp",
)
