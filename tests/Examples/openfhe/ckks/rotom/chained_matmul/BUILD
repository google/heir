# See README.md for setup required to run these tests

load("@heir//tests/Examples/openfhe:test.bzl", "openfhe_end_to_end_test")

package(default_applicable_licenses = ["@heir//:license"])

openfhe_end_to_end_test(
    name = "chained_matmul_test",
    generated_lib_header = "chained_matmul_lib.h",
    heir_opt_flags = [
        "--annotate-module=backend=openfhe scheme=ckks",
        "--mlir-to-ckks=ciphertext-degree=4096 enable-arithmetization=false",
        "--scheme-to-openfhe",
    ],
    heir_translate_flags = [],
    mlir_src = "@heir//tests/Examples/openfhe/ckks/rotom/chained_matmul:chained_matmul.mlir",
    tags = ["notap"],
    test_src = "chained_matmul_test.cpp",
    data = glob(["inputs/*.txt"]) + glob(["results/*.txt"]),
)
