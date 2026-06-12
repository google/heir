load("@heir//tests/Examples/openfhe:test.bzl", "openfhe_end_to_end_test")

package(default_applicable_licenses = ["@heir//:license"])

openfhe_end_to_end_test(
    name = "bicyclic_matmul_test",
    generated_lib_header = "bicyclic_matmul_lib.h",
    heir_opt_flags = [
        "--annotate-module=backend=openfhe scheme=ckks",
        "--mlir-to-ckks=ciphertext-degree=2048",
        "--scheme-to-openfhe",
    ],
    mlir_src = "@heir//tests/Examples/common:bicyclic_matmul.mlir",
    test_src = "bicyclic_matmul_test.cpp",
)
